// Package metrics defines the gateway's Prometheus instrumentation.
//
// Every latency metric is a histogram. Not a gauge, not a summary of averages — a
// histogram. An average latency of 180 ms can sit on top of a p99 of 4 seconds, and the
//
// Bucket boundaries are chosen around the 150 ms target rather than left at the client
// library's defaults. Prometheus' defaults top out at 10 s with coarse spacing below
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// The headline. Dense buckets from 20 ms to 400 ms so the region around the target
	// is legible, then sparse out to 5 s to catch the pathological tail.
	TTFA = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "tts_ttfa_seconds",
		Help: "Time from request accepted to first audio chunk sent to the client.",
		Buckets: []float64{
			0.020, 0.040, 0.060, 0.080, 0.100, 0.120, 0.150, 0.200,
			0.250, 0.300, 0.400, 0.600, 1.0, 2.0, 5.0,
		},
	}, []string{"route"})

	// Real-time factor: audio-seconds produced per wall-clock second of synthesis.
	// Must stay above 1.0 or a live stream underruns mid-sentence. Tracked as a
	// histogram because the question is "did it ever dip", not "was it usually fine".
	RTF = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "tts_realtime_factor",
		Help:    "Audio seconds generated per wall-clock second. Below 1.0 the stream underruns.",
		Buckets: []float64{0.5, 0.8, 1.0, 1.5, 2, 3, 5, 10, 20, 50, 100},
	}, []string{"route"})

	// Gap between a chunk arriving and the moment playback needed it. Negative means
	// the client was already starved. This catches stutter that an aggregate RTF hides
	// completely, because one late chunk in fifty barely moves an average.
	ChunkSlack = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "tts_chunk_slack_seconds",
		Help:    "Playback headroom when a chunk arrived. Negative values are underruns.",
		Buckets: []float64{-1, -0.5, -0.1, 0, 0.1, 0.25, 0.5, 1, 2, 5},
	}, []string{"route"})

	Underruns = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tts_underruns_total",
		Help: "Chunks that arrived after the client had run out of audio to play.",
	}, []string{"route"})

	SessionDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "tts_session_seconds",
		Help:    "Lifetime of a streaming session.",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 12),
	}, []string{"route", "outcome"})

	// Leading saturation indicator: this climbs before latency does, which is exactly
	// what makes it the right signal for admission control rather than a lagging one
	// like observed latency.
	InFlight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "tts_sessions_in_flight",
		Help: "Sessions currently synthesizing.",
	}, []string{"route"})

	Admitted = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tts_sessions_admitted_total",
		Help: "Sessions accepted.",
	}, []string{"route"})

	Rejected = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tts_sessions_rejected_total",
		Help: "Sessions rejected by admission control, by reason.",
	}, []string{"route", "reason"})

	Errors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tts_errors_total",
		Help: "Errors by stage.",
	}, []string{"stage"})

	// Per-stage breakdown, so a TTFA regression can be attributed without a trace.
	StageLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "tts_stage_seconds",
		Help: "Latency of each pipeline stage.",
		Buckets: []float64{
			0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100,
			0.150, 0.200, 0.300, 0.500, 1.0, 2.0,
		},
	}, []string{"stage"})

	AudioSeconds = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "tts_audio_seconds_total",
		Help: "Total audio synthesized. The denominator for cost per audio-minute.",
	}, []string{"route"})
)
