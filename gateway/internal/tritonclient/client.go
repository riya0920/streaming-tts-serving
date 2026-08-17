// Package tritonclient wraps Triton's gRPC inference service for the three-stage TTS
// pipeline.
//
// gRPC rather than HTTP because decoupled streaming — one request producing many
// responses over time — exists only on the gRPC service. That is not a preference; it
// is the mechanism the whole streaming design rests on.
package tritonclient

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	pb "github.com/riya0920/streaming-tts-serving/gateway/internal/tritonpb"
)

const (
	ModelFrontend = "tts_frontend"
	ModelLatents  = "vits_frontend"
	ModelStream   = "tts_stream"
)

type Client struct {
	conn *grpc.ClientConn
	svc  pb.GRPCInferenceServiceClient
}

func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		// Audio chunks are small but frequent; a large window avoids flow-control
		// stalls when many sessions stream at once.
		grpc.WithInitialWindowSize(1<<20),
		grpc.WithInitialConnWindowSize(1<<20),
		// Keepalive must stay inside the SERVER's tolerance, not just be "frequent
		// enough". Triton enforces a minimum ping interval and answers anything faster
		//
		// 5 minutes with pings only while streams are active stays well within the
		// default policy, and streaming sessions keep the connection warm anyway.
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                5 * time.Minute,
			Timeout:             20 * time.Second,
			PermitWithoutStream: false,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("dial triton: %w", err)
	}
	return &Client{conn: conn, svc: pb.NewGRPCInferenceServiceClient(conn)}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

func (c *Client) Ready(ctx context.Context) error {
	r, err := c.svc.ServerReady(ctx, &pb.ServerReadyRequest{})
	if err != nil {
		return err
	}
	if !r.Ready {
		return fmt.Errorf("triton not ready")
	}
	return nil
}

// ---------------------------------------------------------------- tensor helpers

// Triton's BYTES wire format is, per element, a 4-byte little-endian length followed by
// the bytes themselves. Getting this wrong produces a deserialization error deep inside
// the python backend that says nothing about encoding.
func bytesTensorContent(items ...string) []byte {
	out := make([]byte, 0, 8*len(items))
	for _, s := range items {
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], uint32(len(s)))
		out = append(out, n[:]...)
		out = append(out, []byte(s)...)
	}
	return out
}

func int64TensorContent(vals []int64) []byte {
	out := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(out[i*8:], uint64(v))
	}
	return out
}

func float32Content(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func decodeInt64(raw []byte) []int64 {
	out := make([]int64, len(raw)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

func decodeFloat32(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func decodeInt16(raw []byte) []int16 {
	out := make([]int16, len(raw)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return out
}

func outputIndex(resp *pb.ModelInferResponse, name string) int {
	for i, o := range resp.Outputs {
		if o.Name == name {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------- stage 1: text

type FrontendResult struct {
	InputIDs      []int64
	AttentionMask []int64
	SeqLen        int64
	Normalized    string
}

func (c *Client) Frontend(ctx context.Context, text string) (*FrontendResult, error) {
	req := &pb.ModelInferRequest{
		ModelName: ModelFrontend,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{
			{Name: "TEXT", Datatype: "BYTES", Shape: []int64{1}},
		},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{
			{Name: "INPUT_IDS"}, {Name: "ATTENTION_MASK"}, {Name: "NORMALIZED_TEXT"},
		},
		RawInputContents: [][]byte{bytesTensorContent(text)},
	}
	resp, err := c.svc.ModelInfer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("tts_frontend: %w", err)
	}

	out := &FrontendResult{}
	if i := outputIndex(resp, "INPUT_IDS"); i >= 0 {
		out.InputIDs = decodeInt64(resp.RawOutputContents[i])
		out.SeqLen = resp.Outputs[i].Shape[len(resp.Outputs[i].Shape)-1]
	}
	if i := outputIndex(resp, "ATTENTION_MASK"); i >= 0 {
		out.AttentionMask = decodeInt64(resp.RawOutputContents[i])
	}
	if i := outputIndex(resp, "NORMALIZED_TEXT"); i >= 0 {
		raw := resp.RawOutputContents[i]
		if len(raw) >= 4 {
			n := binary.LittleEndian.Uint32(raw[:4])
			if int(n)+4 <= len(raw) {
				out.Normalized = string(raw[4 : 4+n])
			}
		}
	}
	if len(out.InputIDs) == 0 {
		return nil, fmt.Errorf("tts_frontend returned no tokens")
	}
	return out, nil
}

// ---------------------------------------------------------------- stage 2: latents

type Latents struct {
	Data     []float32
	Channels int64
	Frames   int64
}

// tokenBucket rounds a sequence length up to a multiple of this.
//
// Triton's dynamic batcher only groups requests whose non-batch dimensions match, so
// unbucketed token lengths would almost never batch — every utterance is a different
//
// 16 is small enough that padding waste stays under ~15% for typical utterances and
// large enough that the common short replies collapse into one or two buckets.
const tokenBucket = 16

func padTo(vals []int64, n int) []int64 {
	if len(vals) >= n {
		return vals
	}
	out := make([]int64, n)
	copy(out, vals)
	return out
}

func (c *Client) Latents(ctx context.Context, fr *FrontendResult) (*Latents, error) {
	padded := int(fr.SeqLen)
	if r := padded % tokenBucket; r != 0 {
		padded += tokenBucket - r
	}
	// Batch dim stays explicit at 1; the model's max_batch_size lets Triton fuse many
	// of these into one forward pass.
	shape := []int64{1, int64(padded)}
	req := &pb.ModelInferRequest{
		ModelName: ModelLatents,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{
			{Name: "INPUT_IDS", Datatype: "INT64", Shape: shape},
			{Name: "ATTENTION_MASK", Datatype: "INT64", Shape: shape},
		},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{{Name: "LATENTS"}},
		RawInputContents: [][]byte{
			int64TensorContent(padTo(fr.InputIDs, padded)),
			// Padding positions get mask 0, so VITS ignores them rather than
			// synthesizing whatever token id zero happens to mean.
			int64TensorContent(padTo(fr.AttentionMask, padded)),
		},
	}
	resp, err := c.svc.ModelInfer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("vits_frontend: %w", err)
	}
	i := outputIndex(resp, "LATENTS")
	if i < 0 {
		return nil, fmt.Errorf("vits_frontend returned no LATENTS")
	}
	s := resp.Outputs[i].Shape
	if len(s) != 3 {
		return nil, fmt.Errorf("unexpected LATENTS shape %v", s)
	}
	return &Latents{
		Data:     decodeFloat32(resp.RawOutputContents[i]),
		Channels: s[1],
		Frames:   s[2],
	}, nil
}

// ---------------------------------------------------------------- stage 3: stream

type AudioChunk struct {
	PCM     []int16
	Index   int32
	IsFinal bool
}

// StreamAudio opens a decoupled stream and delivers chunks to onChunk as they arrive.
//
// The callback runs on the receive path deliberately: handing chunks to the client
// socket immediately is the entire point, and buffering them into a slice first would
// reintroduce the latency the architecture exists to remove.
func (c *Client) StreamAudio(ctx context.Context, lat *Latents,
	onChunk func(AudioChunk) error) error {

	stream, err := c.svc.ModelStreamInfer(ctx)
	if err != nil {
		return fmt.Errorf("open stream: %w", err)
	}

	req := &pb.ModelInferRequest{
		ModelName: ModelStream,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{
			{Name: "LATENTS", Datatype: "FP32",
				Shape: []int64{1, lat.Channels, lat.Frames}},
		},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{
			{Name: "AUDIO_CHUNK"}, {Name: "CHUNK_INDEX"}, {Name: "IS_FINAL"},
		},
		RawInputContents: [][]byte{float32Content(lat.Data)},
	}
	if err := stream.Send(req); err != nil {
		return fmt.Errorf("send: %w", err)
	}
	if err := stream.CloseSend(); err != nil {
		return fmt.Errorf("close send: %w", err)
	}

	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("recv: %w", err)
		}
		if msg.ErrorMessage != "" {
			return fmt.Errorf("tts_stream: %s", msg.ErrorMessage)
		}
		resp := msg.InferResponse
		if resp == nil {
			continue
		}

		var chunk AudioChunk
		if i := outputIndex(resp, "AUDIO_CHUNK"); i >= 0 {
			chunk.PCM = decodeInt16(resp.RawOutputContents[i])
		}
		if i := outputIndex(resp, "CHUNK_INDEX"); i >= 0 && len(resp.RawOutputContents[i]) >= 4 {
			chunk.Index = int32(binary.LittleEndian.Uint32(resp.RawOutputContents[i]))
		}
		if i := outputIndex(resp, "IS_FINAL"); i >= 0 && len(resp.RawOutputContents[i]) >= 1 {
			chunk.IsFinal = resp.RawOutputContents[i][0] != 0
		}

		if err := onChunk(chunk); err != nil {
			return err
		}
		if chunk.IsFinal {
			return nil
		}
	}
}
