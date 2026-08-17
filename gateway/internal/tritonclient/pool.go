package tritonclient

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Pool spreads sessions across several Triton endpoints.
//
// Why this exists: a single RTX 6000 Ada holds 1,067 sessions at 10% duty with p99
// time-to-first-audio at 149 ms, and breaks when aggregate real-time factor reaches ~205
//
// Routing is least-in-flight rather than round-robin. Round-robin is only fair when
// requests cost the same, and TTS requests do not: a 57-word utterance occupies a
//
// A request is pinned to one endpoint for all three hops. Nothing forces that — the
// endpoints are identical replicas and each hop is stateless — but splitting a request
type Pool struct {
	entries []*poolEntry
	mu      sync.RWMutex
}

type poolEntry struct {
	addr     string
	client   *Client
	inFlight atomic.Int64
	healthy  atomic.Bool
	// Consecutive failures. One transient error should not evict an endpoint; a
	// persistent one should stop receiving traffic.
	failures atomic.Int64
}

const unhealthyAfterFailures = 3

// DialPool connects to a comma-separated list of endpoints.
//
//	"localhost:8001"                              single GPU
//	"localhost:8001,localhost:8011,localhost:8021" three Tritons, one GPU each
func DialPool(addrs string) (*Pool, error) {
	parts := strings.Split(addrs, ",")
	p := &Pool{}
	for _, raw := range parts {
		addr := strings.TrimSpace(raw)
		if addr == "" {
			continue
		}
		c, err := Dial(addr)
		if err != nil {
			return nil, fmt.Errorf("dial %s: %w", addr, err)
		}
		e := &poolEntry{addr: addr, client: c}
		e.healthy.Store(true)
		p.entries = append(p.entries, e)
	}
	if len(p.entries) == 0 {
		return nil, fmt.Errorf("no triton endpoints in %q", addrs)
	}
	return p, nil
}

func (p *Pool) Size() int { return len(p.entries) }

func (p *Pool) Addrs() []string {
	out := make([]string, 0, len(p.entries))
	for _, e := range p.entries {
		out = append(out, e.addr)
	}
	return out
}

// Lease is a checked-out endpoint. Release must be called exactly once.
type Lease struct {
	Client *Client
	Addr   string
	entry  *poolEntry
}

// Release returns the endpoint to the pool and records whether the request succeeded.
func (l *Lease) Release(err error) {
	if l == nil || l.entry == nil {
		return
	}
	l.entry.inFlight.Add(-1)
	if err != nil {
		if l.entry.failures.Add(1) >= unhealthyAfterFailures {
			l.entry.healthy.Store(false)
		}
		return
	}
	// Any success clears the streak: the counter tracks *consecutive* failures, so a
	// single recovery is enough to keep an endpoint in rotation.
	l.entry.failures.Store(0)
	l.entry.healthy.Store(true)
}

// Acquire picks the healthy endpoint with the fewest in-flight requests.
//
// Linear scan: with a handful of endpoints this is a few atomic loads and beats any
// structure that needs locking. It would need revisiting past a few dozen.
func (p *Pool) Acquire() (*Lease, error) {
	var best *poolEntry
	var bestLoad int64

	for _, e := range p.entries {
		if !e.healthy.Load() {
			continue
		}
		load := e.inFlight.Load()
		if best == nil || load < bestLoad {
			best, bestLoad = e, load
		}
	}

	// Every endpoint marked unhealthy: fall back to the least loaded one anyway rather
	// than refusing service. Health here is a heuristic from consecutive failures, and
	// a wrong heuristic should degrade the service, not stop it.
	if best == nil {
		for _, e := range p.entries {
			load := e.inFlight.Load()
			if best == nil || load < bestLoad {
				best, bestLoad = e, load
			}
		}
	}
	if best == nil {
		return nil, fmt.Errorf("no triton endpoints available")
	}

	best.inFlight.Add(1)
	return &Lease{Client: best.client, Addr: best.addr, entry: best}, nil
}

// Ready reports how many endpoints answer, and errors only if none do.
func (p *Pool) Ready(ctx context.Context) (int, error) {
	ok := 0
	var lastErr error
	for _, e := range p.entries {
		c, cancel := context.WithTimeout(ctx, 3*time.Second)
		err := e.client.Ready(c)
		cancel()
		if err == nil {
			ok++
			e.healthy.Store(true)
			e.failures.Store(0)
		} else {
			lastErr = err
			e.healthy.Store(false)
		}
	}
	if ok == 0 {
		return 0, fmt.Errorf("no triton endpoint ready: %w", lastErr)
	}
	return ok, nil
}

// Stats reports per-endpoint load, for /healthz and for debugging skew.
func (p *Pool) Stats() []map[string]any {
	out := make([]map[string]any, 0, len(p.entries))
	for _, e := range p.entries {
		out = append(out, map[string]any{
			"addr":      e.addr,
			"in_flight": e.inFlight.Load(),
			"healthy":   e.healthy.Load(),
			"failures":  e.failures.Load(),
		})
	}
	return out
}

func (p *Pool) Close() error {
	var firstErr error
	for _, e := range p.entries {
		if err := e.client.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
