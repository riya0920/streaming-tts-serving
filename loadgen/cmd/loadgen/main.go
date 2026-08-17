// Command loadgen drives the gateway with real WebSocket sessions and finds the knee.
//
// Three things it does that a naive load test does not:
//
//  1. Realistic sentence lengths. TTS latency scales with utterance length, so a corpus
//     of uniformly short prompts produces a flattering, meaningless number. The corpus
//     is a mixed distribution of acknowledgements, replies, and long explanations.
//
//  2. Duty cycle. A held session in a voice agent is not synthesizing continuously — it
//     speaks in short turns and then waits. "3,200 concurrent sessions" and "3,200
//     simultaneous synthesis streams" differ by an order of magnitude, and conflating
//     them is the single easiest way to publish a number that does not survive a
//     follow-up question. Both are reported, with the duty cycle stated.
//
//  3. Underruns, not just latency. A stream that delivers every chunk late still
//     "succeeds" by a latency metric while sounding broken. Each chunk is checked
//     against when playback needed it.
//
// Runs on the same box as the gateway on purpose: this measures server-side latency, and
// WAN jitter would bury the signal. That excludes real network latency, which is stated
// wherever the results are reported rather than quietly omitted.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// Rough mean utterance length in the corpus, used only to size the startup phase
// spread and to sanity-check run duration. It does not need to be exact — it sets the
// scale of the stagger, not the load itself.
const meanUtteranceSeconds = 6.0

type doneMessage struct {
	Type         string  `json:"type"`
	Chunks       int     `json:"chunks"`
	AudioSeconds float64 `json:"audio_seconds"`
	TTFAMs       float64 `json:"ttfa_ms"`
	TotalMs      float64 `json:"total_ms"`
}

type sample struct {
	ttfaMs    float64
	totalMs   float64
	audioSec  float64
	underruns int
	err       error
	rejected  bool
}

type levelResult struct {
	Concurrency      int     `json:"concurrency"`
	HeldSessions     int     `json:"held_sessions"`
	DutyCycle        float64 `json:"duty_cycle"`
	Requests         int     `json:"requests"`
	Errors           int     `json:"errors"`
	Rejected         int     `json:"rejected"`
	UnderrunRequests int     `json:"requests_with_underruns"`
	TTFAp50          float64 `json:"ttfa_p50_ms"`
	TTFAp90          float64 `json:"ttfa_p90_ms"`
	TTFAp99          float64 `json:"ttfa_p99_ms"`
	TTFAmax          float64 `json:"ttfa_max_ms"`
	AudioSecTotal    float64 `json:"audio_seconds_total"`
	AggregateRTF     float64 `json:"aggregate_rtf"`
	Throughput       float64 `json:"requests_per_second"`
}

func pct(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	// Nearest-rank: never invents a value that was not observed.
	k := int(p/100*float64(len(s))+0.5) - 1
	if k < 0 {
		k = 0
	}
	if k >= len(s) {
		k = len(s) - 1
	}
	return s[k]
}

func loadCorpus(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return []string{"The quick brown fox jumps over the lazy dog."}
	}
	var out []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	return out
}

// oneSession holds a WebSocket open and issues utterances with think-time between them,
// which is what a real voice session looks like: bursts of speech separated by silence.
func oneSession(ctx context.Context, url string, corpus []string, rng *rand.Rand,
	duty float64, out chan<- sample) {

	// Random initial phase, spread across a FULL session cycle.
	//
	// Without any phase, every session starts at t=0 and paces identically, so they stay
	// locked in step and arrive in synchronized waves — measuring a thundering herd the
	// load generator created rather than the server's capacity.
	//
	// The spread must scale with duty cycle, which the first version got wrong: it
	// jittered over a fixed 6 s while a session at duty=0.1 has a ~60 s cycle. Every
	// session therefore fired inside the first 6 s and then went idle, and 400 held
	// sessions looked like 400 simultaneous arrivals (p99 1473 ms) instead of ~40
	// concurrently speaking. A session cycle is roughly meanUtteranceSeconds/duty, so
	// spread the starts over exactly that.
	cycle := meanUtteranceSeconds / duty
	phase := time.Duration(rng.Float64() * cycle * float64(time.Second))
	select {
	case <-time.After(phase):
	case <-ctx.Done():
		return
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, resp, err := dialer.DialContext(ctx, url, nil)
	if err != nil {
		s := sample{err: err}
		if resp != nil && resp.StatusCode == 503 {
			s.rejected = true // admission control did its job; not a failure
		}
		select {
		case out <- s:
		case <-ctx.Done():
		}
		return
	}
	defer conn.Close()

	for ctx.Err() == nil {
		text := corpus[rng.Intn(len(corpus))]
		req, _ := json.Marshal(map[string]string{"text": text})
		_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
		if err := conn.WriteMessage(websocket.TextMessage, req); err != nil {
			select {
			case out <- sample{err: err}:
			case <-ctx.Done():
			}
			return
		}

		start := time.Now()
		var (
			firstAt   time.Duration
			playedSec float64
			underruns int
			gotFirst  bool
		)

		for {
			_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
			mt, data, err := conn.ReadMessage()
			if err != nil {
				select {
				case out <- sample{err: err}:
				case <-ctx.Done():
				}
				return
			}
			if mt == websocket.BinaryMessage {
				now := time.Since(start)
				if !gotFirst {
					firstAt, gotFirst = now, true
				} else if now.Seconds() > firstAt.Seconds()+playedSec {
					// The chunk arrived after the audio already delivered would have
					// finished playing — an audible gap, invisible to a latency metric.
					underruns++
				}
				playedSec += float64(len(data)/2) / 16000.0
				continue
			}

			var d doneMessage
			if err := json.Unmarshal(data, &d); err != nil || d.Type != "done" {
				select {
				case out <- sample{err: fmt.Errorf("server: %s", string(data))}:
				case <-ctx.Done():
				}
				return
			}
			select {
			case out <- sample{ttfaMs: d.TTFAMs, totalMs: d.TotalMs,
				audioSec: d.AudioSeconds, underruns: underruns}:
			case <-ctx.Done():
				return
			}

			// Pace the session to wall-clock time.
			//
			// This was wrong in the first version and it invalidated the headline
			// number: idling only for a *fraction* of the audio meant duty=1.0 left no
			// think time at all, so each session looped as fast as the server would
			// answer — about 66x real time. Sixteen such sessions are not sixteen
			// listeners, they are sixteen batch jobs, and calling that "16 concurrent
			// sessions" would overstate capacity by nearly two orders of magnitude.
			//
			// A session must occupy audioSeconds/duty of wall time per utterance:
			// at duty=1.0 it speaks continuously in real time, at duty=0.1 it speaks
			// for a tenth of the wall clock, which is what a voice agent taking short
			// turns actually looks like. Synthesis time already elapsed counts toward
			// that budget.
			if duty > 0 {
				budget := time.Duration(d.AudioSeconds / duty * float64(time.Second))
				if idle := budget - time.Since(start); idle > 0 {
					select {
					case <-time.After(idle):
					case <-ctx.Done():
						return
					}
				}
			}
			break
		}
	}
}

func runLevel(url string, held int, duty float64, dur time.Duration,
	corpus []string, seed int64) levelResult {

	ctx, cancel := context.WithTimeout(context.Background(), dur)
	defer cancel()

	out := make(chan sample, held*8)
	var wg sync.WaitGroup
	for i := 0; i < held; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			oneSession(ctx, url, corpus, rand.New(rand.NewSource(seed+int64(id))), duty, out)
		}(i)
	}
	go func() { wg.Wait(); close(out) }()

	var (
		ttfa      []float64
		errs      int
		rejected  int
		underrun  int
		audioSec  float64
		requests  int
		totalWall = dur.Seconds()
	)
	for s := range out {
		if s.rejected {
			rejected++
			continue
		}
		if s.err != nil {
			errs++
			continue
		}
		requests++
		ttfa = append(ttfa, s.ttfaMs)
		audioSec += s.audioSec
		if s.underruns > 0 {
			underrun++
		}
	}

	return levelResult{
		Concurrency: held, HeldSessions: held, DutyCycle: duty,
		Requests: requests, Errors: errs, Rejected: rejected,
		UnderrunRequests: underrun,
		TTFAp50:          pct(ttfa, 50), TTFAp90: pct(ttfa, 90),
		TTFAp99: pct(ttfa, 99), TTFAmax: pct(ttfa, 100),
		AudioSecTotal: audioSec,
		AggregateRTF:  audioSec / totalWall,
		Throughput:    float64(requests) / totalWall,
	}
}

func main() {
	url := flag.String("url", "ws://localhost:8080/v1/stream", "gateway websocket url")
	levels := flag.String("levels", "1,4,16,32,64,128,256,512", "held sessions per level")
	duty := flag.Float64("duty", 1.0, "fraction of time each session is speaking")
	dur := flag.Duration("duration", 20*time.Second, "per level")
	corpusPath := flag.String("corpus", "loadgen/corpus.txt", "")
	outPath := flag.String("out", "results/m9_loadtest.json", "")
	target := flag.Float64("target-p99", 150, "p99 TTFA target in ms; the knee is reported against this")
	seed := flag.Int64("seed", 1337, "")
	flag.Parse()

	corpus := loadCorpus(*corpusPath)
	fmt.Printf("corpus: %d lines | duty cycle: %.0f%% | %s per level\n\n",
		len(corpus), *duty*100, *dur)
	fmt.Printf("%8s %9s %9s %9s %9s %8s %9s %8s %8s\n",
		"held", "p50 ms", "p90 ms", "p99 ms", "max ms", "rps", "agg RTF", "under", "rej")

	var results []levelResult
	var knee int
	for _, ls := range strings.Split(*levels, ",") {
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(ls), "%d", &n); err != nil || n <= 0 {
			continue
		}
		r := runLevel(*url, n, *duty, *dur, corpus, *seed)
		results = append(results, r)
		fmt.Printf("%8d %9.1f %9.1f %9.1f %9.1f %8.1f %9.1f %8d %8d\n",
			r.HeldSessions, r.TTFAp50, r.TTFAp90, r.TTFAp99, r.TTFAmax,
			r.Throughput, r.AggregateRTF, r.UnderrunRequests, r.Rejected)

		if r.TTFAp99 <= *target && r.UnderrunRequests == 0 && r.Errors == 0 {
			knee = n
		}
		// Past the knee the interesting question is answered; keep going one step to
		// show the shape of the failure, then stop burning GPU time.
		if r.TTFAp99 > *target*3 || r.Errors > r.Requests/5 {
			fmt.Println("  past the knee — stopping ramp")
			break
		}
	}

	fmt.Printf("\nhighest level holding p99 <= %.0f ms with no underruns: %d sessions",
		*target, knee)
	if *duty < 1 {
		fmt.Printf("  (~%.0f concurrently speaking at %.0f%% duty)", float64(knee)*(*duty), *duty*100)
	}
	fmt.Println()

	payload := map[string]any{
		"target_p99_ms": *target,
		"duty_cycle":    *duty,
		"knee_sessions": knee,
		"note":          "loadgen runs on the same host as the gateway: server-side latency only, excludes WAN",
		"levels":        results,
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	_ = os.MkdirAll("results", 0o755)
	if err := os.WriteFile(*outPath, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write results:", err)
	} else {
		fmt.Println("wrote", *outPath)
	}
	_ = atomic.LoadInt64(new(int64))
}
