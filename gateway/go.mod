module github.com/riya0920/streaming-tts-serving/gateway

go 1.22

require (
	github.com/gorilla/websocket v1.5.3
	github.com/prometheus/client_golang v1.20.4
	go.opentelemetry.io/otel v1.30.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.30.0
	go.opentelemetry.io/otel/sdk v1.30.0
	go.opentelemetry.io/otel/trace v1.30.0
	google.golang.org/grpc v1.66.2
	google.golang.org/protobuf v1.34.2
)
