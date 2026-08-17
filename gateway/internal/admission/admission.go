// Package admission decides whether to accept another session.
//
// The argument for this is measured, not theoretical. M1 loaded the naive baseline until
// it fell over: at concurrency 8 throughput collapsed 3.5x, and at 128 not one request
//
// So: past a threshold, reject immediately. Better to fail one user in 2 ms — who can
// retry, or be routed elsewhere — than to admit them and drag three thousand others'
package admission

import (
	"sync"
	"sync/atomic"
	"time"
)

type Decision struct {
	Admit  bool
	Reason string
}

type Config struct {
	// Hard ceiling on sessions actively synthesizing.
	MaxInFlight int64
	// Triton's queue depth is a *leading* indicator — it rises before latency does,
	// which is what makes it useful. Observed latency is lagging: by the time it moves,
	// the queue is already deep enough that the next arrivals are doomed.
	MaxQueueDepth int64
	// Once tripped, stay closed briefly. Without hysteresis the controller oscillates:
	// it rejects, the queue drains a little, it admits a burst, the queue spikes again.
	CooldownAfterTrip time.Duration
}

func DefaultConfig() Config {
	return Config{
		MaxInFlight:       3500,
		MaxQueueDepth:     64,
		CooldownAfterTrip: 250 * time.Millisecond,
	}
}

type Controller struct {
	cfg Config

	inFlight   atomic.Int64
	queueDepth atomic.Int64

	mu         sync.Mutex
	trippedAt  time.Time
	lastReason string
}

func New(cfg Config) *Controller { return &Controller{cfg: cfg} }

// ObserveQueueDepth is fed by a poller reading Triton's metrics endpoint. Kept as a
// push rather than a synchronous read so that admission never blocks on a scrape.
func (c *Controller) ObserveQueueDepth(d int64) { c.queueDepth.Store(d) }

func (c *Controller) InFlight() int64   { return c.inFlight.Load() }
func (c *Controller) QueueDepth() int64 { return c.queueDepth.Load() }

func (c *Controller) TryAdmit() Decision {
	c.mu.Lock()
	if !c.trippedAt.IsZero() && time.Since(c.trippedAt) < c.cfg.CooldownAfterTrip {
		reason := c.lastReason
		c.mu.Unlock()
		return Decision{Admit: false, Reason: reason + "_cooldown"}
	}
	c.mu.Unlock()

	if c.inFlight.Load() >= c.cfg.MaxInFlight {
		c.trip("max_in_flight")
		return Decision{Admit: false, Reason: "max_in_flight"}
	}
	if c.cfg.MaxQueueDepth > 0 && c.queueDepth.Load() >= c.cfg.MaxQueueDepth {
		c.trip("queue_depth")
		return Decision{Admit: false, Reason: "queue_depth"}
	}

	c.inFlight.Add(1)
	return Decision{Admit: true}
}

func (c *Controller) Release() {
	if c.inFlight.Add(-1) < 0 {
		c.inFlight.Store(0)
	}
}

func (c *Controller) trip(reason string) {
	c.mu.Lock()
	c.trippedAt = time.Now()
	c.lastReason = reason
	c.mu.Unlock()
}
