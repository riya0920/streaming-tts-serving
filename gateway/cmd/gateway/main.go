// Command gateway is the control plane in front of Triton.
//
// Triton is deliberately dumb about everything except inference: it does not know what a
// session is, does not do admission, does not decide routing. This process owns all of
// that, terminates client WebSockets, and relays audio chunks as they arrive.
//
// Go because holding thousands of streaming connections means thousands of goroutines
// blocked on I/O, which costs kilobytes each rather than an OS thread each — and because
//
// Two routes over the same models:
//
//	/v1/stream     WebSocket, latency-tuned. Chunks go out the instant they arrive.
//	/v1/synthesize HTTP, throughput-tuned. Whole-file synthesis for offline work.
//
// They are separated so that an offline job — synthesizing an article to a file, where
// nobody is waiting — can never sit in front of a live listener's next chunk.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"

	"github.com/riya0920/streaming-tts-serving/gateway/internal/admission"
	"github.com/riya0920/streaming-tts-serving/gateway/internal/metrics"
	"github.com/riya0920/streaming-tts-serving/gateway/internal/tritonclient"
)

const sampleRate = 16000

type config struct {
	listen        string
	metricsListen string
	tritonGRPC    string  // comma-separated list of endpoints
	tritonMetrics string
	maxInFlight   int64
	maxQueueDepth int64
	otlpEndpoint  string
}

func loadConfig() config {
	env := func(k, def string) string {
		if v := os.Getenv(k); v != "" {
			return v
		}
		return def
	}
	envInt := func(k string, def int64) int64 {
		if v := os.Getenv(k); v != "" {
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				return n
			}
		}
		return def
	}
	return config{
		listen:        env("GATEWAY_LISTEN", ":8080"),
		metricsListen: env("GATEWAY_METRICS_LISTEN", ":9091"),
		// Comma-separated: one entry per GPU. A single RTX 6000 Ada holds ~1,067
		// sessions at 10% duty; beyond that capacity is a horizontal problem.
		tritonGRPC:    env("TRITON_GRPC_ADDR", "localhost:8001"),
		tritonMetrics: env("TRITON_METRICS_URL", "http://localhost:8002/metrics"),
		maxInFlight:   envInt("MAX_INFLIGHT_SESSIONS", 3500),
		maxQueueDepth: envInt("MAX_TRITON_QUEUE_DEPTH", 64),
		otlpEndpoint:  os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
	}
}

type server struct {
	cfg      config
	pool     *tritonclient.Pool
	adm      *admission.Controller
	tracer   trace.Tracer
	log      *slog.Logger
	upgrader websocket.Upgrader
	sessions atomic.Int64
}

type streamRequest struct {
	Text string `json:"text"`
}

type doneMessage struct {
	Type         string  `json:"type"`
	Chunks       int     `json:"chunks"`
	AudioSeconds float64 `json:"audio_seconds"`
	TTFAMs       float64 `json:"ttfa_ms"`
	TotalMs      float64 `json:"total_ms"`
	Normalized   string  `json:"normalized_text"`
	SampleRate   int     `json:"sample_rate"`
}

func main() {
	cfg := loadConfig()
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	shutdownTracing, err := initTracing(cfg, log)
	if err != nil {
		log.Warn("tracing disabled", "err", err)
	}
	defer shutdownTracing()

	pool, err := tritonclient.DialPool(cfg.tritonGRPC)
	if err != nil {
		log.Error("cannot reach triton", "err", err)
		os.Exit(1)
	}
	defer pool.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	for {
		if n, err := pool.Ready(ctx); err == nil {
			log.Info("triton endpoints ready", "ready", n, "total", pool.Size(),
				"addrs", pool.Addrs())
			break
		}
		select {
		case <-ctx.Done():
			log.Error("triton never became ready")
			cancel()
			os.Exit(1)
		case <-time.After(2 * time.Second):
		}
	}
	cancel()

	s := &server{
		cfg:  cfg,
		pool: pool,
		adm: admission.New(admission.Config{
			MaxInFlight:       cfg.maxInFlight,
			MaxQueueDepth:     cfg.maxQueueDepth,
			CooldownAfterTrip: 250 * time.Millisecond,
		}),
		tracer: otel.Tracer("tts-gateway"),
		log:    log,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  4096,
			WriteBufferSize: 32768,
			CheckOrigin:     func(*http.Request) bool { return true },
		},
	}

	go s.pollQueueDepth()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/stream", s.handleStream)
	mux.HandleFunc("/v1/synthesize", s.handleBatch)
	mux.HandleFunc("/healthz", s.handleHealth)

	mmux := http.NewServeMux()
	mmux.Handle("/metrics", promhttp.Handler())

	srv := &http.Server{Addr: cfg.listen, Handler: mux}
	msrv := &http.Server{Addr: cfg.metricsListen, Handler: mmux}

	go func() {
		if err := msrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("metrics server", "err", err)
		}
	}()
	go func() {
		log.Info("gateway listening", "addr", cfg.listen, "triton_endpoints", pool.Addrs(),
			"max_in_flight", cfg.maxInFlight, "max_queue_depth", cfg.maxQueueDepth)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("gateway server", "err", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Info("shutting down")
	sctx, scancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer scancel()
	_ = srv.Shutdown(sctx)
	_ = msrv.Shutdown(sctx)
}

// pollQueueDepth feeds admission control from Triton's own metrics.
//
// Queue depth is a leading indicator: it rises before latency does. Admission control
// driven by observed latency is always too late, because by the time latency moves the
// queue is already deep enough that the next arrivals are doomed.
func (s *server) pollQueueDepth() {
	client := &http.Client{Timeout: 2 * time.Second}
	var lastCount, lastNs float64
	// Triton's counters are cumulative and survive gateway restarts. Without this flag
	// the first poll differences against zero, so the entire lifetime counter reads as
	primed := false
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		resp, err := client.Get(s.cfg.tritonMetrics)
		if err != nil {
			continue
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			continue
		}

		// Triton exposes cumulative queue time and request count per model. The ratio
		// of their deltas is mean queue latency; multiplied by arrival rate it
		// approximates Little's Law queue depth without needing a separate exporter.
		var count, ns float64
		for _, line := range strings.Split(string(body), "\n") {
			if strings.HasPrefix(line, "nv_inference_queue_duration_us") {
				ns += lastField(line)
			} else if strings.HasPrefix(line, "nv_inference_request_success") {
				count += lastField(line)
			}
		}
		dCount, dNs := count-lastCount, ns-lastNs
		lastCount, lastNs = count, ns
		if !primed {
			primed = true
			continue
		}
		if dCount > 0 && dNs > 0 {
			meanQueueSec := (dNs / dCount) / 1e6
			arrivalRate := dCount / 0.5
			s.adm.ObserveQueueDepth(int64(meanQueueSec * arrivalRate))
		}
	}
}

func lastField(line string) float64 {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return 0
	}
	v, _ := strconv.ParseFloat(fields[len(fields)-1], 64)
	return v
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	status := map[string]any{
		"in_flight":   s.adm.InFlight(),
		"queue_depth": s.adm.QueueDepth(),
		"sessions":    s.sessions.Load(),
		"endpoints":   s.pool.Stats(),
	}
	if _, err := s.pool.Ready(ctx); err != nil {
		status["ok"] = false
		w.WriteHeader(http.StatusServiceUnavailable)
	} else {
		status["ok"] = true
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(status)
}

// handleStream is the live path: WebSocket in, PCM chunks out as they are produced.
func (s *server) handleStream(w http.ResponseWriter, r *http.Request) {
	const route = "stream"

	if d := s.adm.TryAdmit(); !d.Admit {
		metrics.Rejected.WithLabelValues(route, d.Reason).Inc()
		// Reject before the upgrade, and fast. A client that is refused in 2 ms can
		// retry or fail over; one admitted into a saturated queue produces audio that
		// underruns mid-sentence, which is worse than no audio at all.
		http.Error(w, `{"error":"overloaded","reason":"`+d.Reason+`"}`,
			http.StatusServiceUnavailable)
		return
	}
	defer s.adm.Release()

	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		metrics.Errors.WithLabelValues("upgrade").Inc()
		return
	}
	defer conn.Close()

	metrics.Admitted.WithLabelValues(route).Inc()
	metrics.InFlight.WithLabelValues(route).Inc()
	defer metrics.InFlight.WithLabelValues(route).Dec()
	s.sessions.Add(1)
	defer s.sessions.Add(-1)

	sessionStart := time.Now()
	outcome := "ok"
	defer func() {
		metrics.SessionDuration.WithLabelValues(route, outcome).
			Observe(time.Since(sessionStart).Seconds())
	}()

	conn.SetReadLimit(64 << 10)
	for {
		_ = conn.SetReadDeadline(time.Now().Add(5 * time.Minute))
		_, data, err := conn.ReadMessage()
		if err != nil {
			return // client closed, or idled out
		}
		var req streamRequest
		if err := json.Unmarshal(data, &req); err != nil || strings.TrimSpace(req.Text) == "" {
			_ = conn.WriteJSON(map[string]string{"type": "error", "error": "expected {\"text\": \"...\"}"})
			continue
		}
		if err := s.synthesizeToSocket(r.Context(), conn, req.Text, route); err != nil {
			outcome = "error"
			metrics.Errors.WithLabelValues("synthesize").Inc()
			s.log.Error("synthesize failed", "err", err)
			_ = conn.WriteJSON(map[string]string{"type": "error", "error": err.Error()})
			return
		}
	}
}

func (s *server) synthesizeToSocket(ctx context.Context, conn *websocket.Conn,
	text, route string) error {

	ctx, span := s.tracer.Start(ctx, "tts.synthesize",
		trace.WithAttributes(attribute.Int("text.len", len(text)),
			attribute.String("route", route)))
	defer span.End()

	start := time.Now()

	// One endpoint for all three hops. Splitting them would ship latents between
	// machines for no benefit, and would let one slow endpoint touch every request
	// rather than a share of them.
	lease, err := s.pool.Acquire()
	if err != nil {
		span.RecordError(err)
		return err
	}
	var leaseErr error
	defer func() { lease.Release(leaseErr) }()
	span.SetAttributes(attribute.String("triton.endpoint", lease.Addr))

	_, fspan := s.tracer.Start(ctx, "tts.frontend")
	fr, err := lease.Client.Frontend(ctx, text)
	fspan.End()
	if err != nil {
		leaseErr = err
		span.RecordError(err)
		return err
	}
	metrics.StageLatency.WithLabelValues("frontend").Observe(time.Since(start).Seconds())

	latStart := time.Now()
	_, lspan := s.tracer.Start(ctx, "tts.latents")
	lat, err := lease.Client.Latents(ctx, fr)
	lspan.End()
	if err != nil {
		leaseErr = err
		span.RecordError(err)
		return err
	}
	metrics.StageLatency.WithLabelValues("latents").Observe(time.Since(latStart).Seconds())
	span.SetAttributes(attribute.Int64("latent.frames", lat.Frames))

	_, sspan := s.tracer.Start(ctx, "tts.stream")
	defer sspan.End()

	var (
		chunks       int
		totalSamples int
		ttfa         time.Duration
		playedSec    float64
	)

	err = lease.Client.StreamAudio(ctx, lat, func(c tritonclient.AudioChunk) error {
		now := time.Since(start)
		if chunks == 0 {
			ttfa = now
			metrics.TTFA.WithLabelValues(route).Observe(now.Seconds())
			span.SetAttributes(attribute.Float64("ttfa_ms", float64(now.Milliseconds())))
		} else {
			// Headroom at the moment this chunk landed. Negative means the client had
			// already run dry and the listener heard a gap — the failure an aggregate
			// real-time factor averages away entirely.
			slack := (ttfa.Seconds() + playedSec) - now.Seconds()
			metrics.ChunkSlack.WithLabelValues(route).Observe(slack)
			if slack < 0 {
				metrics.Underruns.WithLabelValues(route).Inc()
			}
		}

		buf := make([]byte, 2*len(c.PCM))
		for i, v := range c.PCM {
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
		}
		_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
		if err := conn.WriteMessage(websocket.BinaryMessage, buf); err != nil {
			return fmt.Errorf("write chunk: %w", err)
		}

		chunks++
		totalSamples += len(c.PCM)
		playedSec += float64(len(c.PCM)) / sampleRate
		return nil
	})
	if err != nil {
		leaseErr = err
		span.RecordError(err)
		return err
	}

	total := time.Since(start)
	audioSec := float64(totalSamples) / sampleRate
	if total > 0 {
		metrics.RTF.WithLabelValues(route).Observe(audioSec / total.Seconds())
	}
	metrics.AudioSeconds.WithLabelValues(route).Add(audioSec)

	_ = conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
	return conn.WriteJSON(doneMessage{
		Type:         "done",
		Chunks:       chunks,
		AudioSeconds: audioSec,
		TTFAMs:       float64(ttfa.Microseconds()) / 1000,
		TotalMs:      float64(total.Microseconds()) / 1000,
		Normalized:   fr.Normalized,
		SampleRate:   sampleRate,
	})
}

// handleBatch is the offline path: synthesize the whole thing, return a WAV.
//
// It shares the models but not the queue discipline. Nobody is waiting on the far end
// of this, so it must never be allowed to sit in front of a live listener's next chunk.
func (s *server) handleBatch(w http.ResponseWriter, r *http.Request) {
	const route = "batch"
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// Batch work is admitted far more conservatively than live streams: it is throughput
	// work with no listener, so it yields whenever the live path is under any pressure.
	if s.adm.QueueDepth() > s.cfg.maxQueueDepth/4 {
		metrics.Rejected.WithLabelValues(route, "yield_to_live").Inc()
		http.Error(w, `{"error":"busy","reason":"yield_to_live"}`, http.StatusServiceUnavailable)
		return
	}

	var req streamRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
		http.Error(w, `{"error":"expected {\"text\": \"...\"}"}`, http.StatusBadRequest)
		return
	}

	metrics.Admitted.WithLabelValues(route).Inc()
	metrics.InFlight.WithLabelValues(route).Inc()
	defer metrics.InFlight.WithLabelValues(route).Dec()

	ctx, span := s.tracer.Start(r.Context(), "tts.batch")
	defer span.End()
	start := time.Now()

	lease, err := s.pool.Acquire()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	var leaseErr error
	defer func() { lease.Release(leaseErr) }()

	fr, err := lease.Client.Frontend(ctx, req.Text)
	if err != nil {
		leaseErr = err
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	lat, err := lease.Client.Latents(ctx, fr)
	if err != nil {
		leaseErr = err
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var pcm []int16
	err = lease.Client.StreamAudio(ctx, lat, func(c tritonclient.AudioChunk) error {
		pcm = append(pcm, c.PCM...)
		return nil
	})
	if err != nil {
		leaseErr = err
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	audioSec := float64(len(pcm)) / sampleRate
	metrics.AudioSeconds.WithLabelValues(route).Add(audioSec)
	if d := time.Since(start); d > 0 {
		metrics.RTF.WithLabelValues(route).Observe(audioSec / d.Seconds())
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("X-Normalized-Text", fr.Normalized)
	if _, err := w.Write(wavBytes(pcm, sampleRate)); err != nil {
		metrics.Errors.WithLabelValues("batch_write").Inc()
	}
}

func wavBytes(pcm []int16, sr int) []byte {
	dataLen := 2 * len(pcm)
	buf := make([]byte, 44+dataLen)
	copy(buf[0:], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:], uint32(36+dataLen))
	copy(buf[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(buf[16:], 16)
	binary.LittleEndian.PutUint16(buf[20:], 1)
	binary.LittleEndian.PutUint16(buf[22:], 1)
	binary.LittleEndian.PutUint32(buf[24:], uint32(sr))
	binary.LittleEndian.PutUint32(buf[28:], uint32(sr*2))
	binary.LittleEndian.PutUint16(buf[32:], 2)
	binary.LittleEndian.PutUint16(buf[34:], 16)
	copy(buf[36:], "data")
	binary.LittleEndian.PutUint32(buf[40:], uint32(dataLen))
	for i, v := range pcm {
		binary.LittleEndian.PutUint16(buf[44+i*2:], uint16(v))
	}
	return buf
}

func initTracing(cfg config, log *slog.Logger) (func(), error) {
	noop := func() {}
	if cfg.otlpEndpoint == "" {
		return noop, errors.New("OTEL_EXPORTER_OTLP_ENDPOINT unset")
	}
	ctx := context.Background()
	exp, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpointURL(cfg.otlpEndpoint+"/v1/traces"),
		otlptracehttp.WithInsecure())
	if err != nil {
		return noop, err
	}
	res, _ := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL, semconv.ServiceName("tts-gateway")))
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
		// Sample everything at the gateway; the collector does tail sampling and keeps
		// the slow traces. Head-sampling here would discard exactly the requests worth
		// looking at.
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)
	log.Info("tracing enabled", "endpoint", cfg.otlpEndpoint)
	return func() {
		c, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = tp.Shutdown(c)
	}, nil
}
