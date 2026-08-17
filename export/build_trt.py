"""M4 — build TensorRT engines from the exported ONNX, then prove they are worth using."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from transformers import VitsModel, VitsTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
LOGGER = trt.Logger(trt.Logger.WARNING)


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


# --------------------------------------------------------------------------- build
def build_engine(onnx_path: Path, out_path: Path, profiles: dict, fp16: bool) -> dict:
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)

    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profiles.items():
        profile.set_shape(name, tuple(mn), tuple(opt), tuple(mx))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError(f"engine build failed for {onnx_path.name}")

    out_path.write_bytes(plan)
    return {"file": out_path.name, "build_seconds": round(build_s, 1),
            "size_mb": round(out_path.stat().st_size / 1e6, 2), "fp16": fp16}


# --------------------------------------------------------------------------- run
class TRTRunner:
    """Minimal executor. Buffers are torch CUDA tensors, so no manual memory management."""

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inputs.append(n)
            else:
                self.outputs.append(n)

    @staticmethod
    def _torch_dtype(t: trt.DataType):
        return {trt.float32: torch.float32, trt.float16: torch.float16,
                trt.int32: torch.int32, trt.int64: torch.int64,
                trt.int8: torch.int8, trt.bool: torch.bool}[t]

    def __call__(self, feeds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        for n, t in feeds.items():
            t = t.contiguous()
            # set_input_shape returns False for a shape outside the engine's optimization
            # profile and does NOT raise. The context keeps its previous shape, the output
            if not self.ctx.set_input_shape(n, tuple(t.shape)):
                lo, hi = None, None
                try:
                    prof = self.engine.get_tensor_profile_shape(n, 0)
                    lo, hi = tuple(prof[0]), tuple(prof[2])
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"shape {tuple(t.shape)} for input '{n}' is outside the engine's "
                    f"optimization profile (min {lo}, max {hi}). Rebuild the engine with "
                    f"a wider profile, or clamp the input upstream."
                )
            want = self._torch_dtype(self.engine.get_tensor_dtype(n))
            if t.dtype != want:
                t = t.to(want).contiguous()
            feeds[n] = t
            self.ctx.set_tensor_address(n, t.data_ptr())

        outs = []
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            buf = torch.empty(shape, dtype=self._torch_dtype(self.engine.get_tensor_dtype(n)),
                              device="cuda")
            self.ctx.set_tensor_address(n, buf.data_ptr())
            outs.append(buf)

        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs


def bench(fn, iters: int = 50) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


# ----------------------------------------------------------------------- quality
def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    err = test[:n].astype(np.float64) - ref[:n].astype(np.float64)
    p_sig = float(np.sum(ref[:n].astype(np.float64) ** 2))
    p_err = float(np.sum(err ** 2))
    return float("inf") if p_err <= 0 else 10.0 * np.log10(p_sig / p_err)


def log_spectral_distance(ref: np.ndarray, test: np.ndarray, n_fft: int = 512) -> float:
    """Mean per-frame RMS difference of log magnitude spectra, in dB.

    Closer to perception than raw SNR: a constant tiny phase error tanks SNR while being
    completely inaudible, and LSD does not punish it.
    """
    n = min(len(ref), len(test))
    a = torch.from_numpy(ref[:n].astype(np.float32))
    b = torch.from_numpy(test[:n].astype(np.float32))
    win = torch.hann_window(n_fft)
    A = torch.stft(a, n_fft, hop_length=n_fft // 4, window=win, return_complex=True).abs()
    B = torch.stft(b, n_fft, hop_length=n_fft // 4, window=win, return_complex=True).abs()
    eps = 1e-8
    d = 20 * (torch.log10(A + eps) - torch.log10(B + eps))
    return float(torch.sqrt((d ** 2).mean(dim=0)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--out", default="results/m4_trt.json")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir or (Path(MODEL_DIR) / "onnx"))
    manifest = json.loads((onnx_dir / "manifest.json").read_text())
    ch = manifest["flow_channels"]
    sr = manifest["sampling_rate"]
    engines_dir = onnx_dir.parent / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    # Frame range from M3's progressive chunking: smallest is a 1-frame remainder with
    # left context only (14); steady state is 50 kept + 26 context (76).
    dec_profile = {"latents": ([1, ch, 14], [8, ch, 76], [32, ch, 76])}
    # Character-level tokenizer: a 60-word sentence is ~300 tokens, so 256 would reject
    # every long utterance in the corpus.
    enc_profile = {"input_ids": ([1, 4], [8, 64], [32, 1024]),
                   "attention_mask": ([1, 4], [8, 64], [32, 1024])}

    results: dict = {"trt_version": trt.__version__, "engines": {}, "decoder": {}, "encoder": {}}

    print(f"TensorRT {trt.__version__}\n")
    for tag, fp16 in (("fp32", False), ("fp16", True)):
        for name, onnx_name, prof in (("decoder", "vits_decoder.onnx", dec_profile),
                                      ("encoder", "vits_encoder.onnx", enc_profile)):
            out = engines_dir / f"vits_{name}_{tag}.plan"
            if out.exists():
                print(f"  {out.name} exists, skipping build")
                info = {"file": out.name, "size_mb": round(out.stat().st_size / 1e6, 2),
                        "fp16": fp16, "build_seconds": None}
            else:
                print(f"  building {out.name} ...", flush=True)
                info = build_engine(onnx_dir / onnx_name, out, prof, fp16)
                print(f"    {info['build_seconds']}s, {info['size_mb']} MB")
            results["engines"][f"{name}_{tag}"] = info

    # ---------------------------------------------------------------- decoder bench
    path = resolve()
    model = VitsModel.from_pretrained(path).cuda().eval()
    tok = VitsTokenizer.from_pretrained(path)

    print("\nDECODER — TensorRT vs PyTorch, at served chunk shapes")
    print("(M2 says this model is launch-bound here, so fusion should dominate FP16)\n")
    print(f"  {'shape':>14}{'torch ms':>10}{'trt32 ms':>10}{'trt16 ms':>10}{'speedup':>9}")

    r32 = TRTRunner(engines_dir / "vits_decoder_fp32.plan")
    r16 = TRTRunner(engines_dir / "vits_decoder_fp16.plan")
    dec_rows = {}
    for B, T in ((1, 38), (1, 76), (8, 76), (32, 76)):
        z = torch.randn(B, ch, T, device="cuda")
        with torch.inference_mode():
            t_torch = bench(lambda: model.decoder(z))
        t32 = bench(lambda: r32({"latents": z}))
        t16 = bench(lambda: r16({"latents": z}))
        dec_rows[f"B{B}_T{T}"] = {"torch_ms": round(t_torch, 3), "trt_fp32_ms": round(t32, 3),
                                  "trt_fp16_ms": round(t16, 3),
                                  "speedup_fp16": round(t_torch / t16, 2)}
        print(f"  {f'B={B} T={T}':>14}{t_torch:>10.3f}{t32:>10.3f}{t16:>10.3f}"
              f"{t_torch / t16:>8.2f}x")
    results["decoder"]["bench"] = dec_rows

    # -------------------------------------------------------------- fp16 quality
    print("\nFP16 QUALITY — TRT FP16 decoder vs PyTorch FP32, on real latents\n")
    from streaming.chunked import ChunkConfig, ChunkedSynthesizer  # noqa: E402
    syn = ChunkedSynthesizer(model, tok, ChunkConfig(), "cuda")
    texts = [
        "Sure, I can help with that.",
        "Your flight leaves at four fifteen from gate B twelve, and boarding starts soon.",
        "The main difference is that the first option charges a flat monthly rate regardless of usage.",
    ]
    torch.manual_seed(7)
    qual = []
    print(f"  {'frames':>8}{'SNR dB':>10}{'LSD dB':>10}")
    for t in texts:
        z = syn.latents_for(t)
        T = z.shape[-1]
        with torch.inference_mode():
            ref = model.decoder(z).squeeze().float().cpu().numpy()
        # Engine profile caps frames at 76, so decode the utterance in valid-size pieces.
        outs = []
        for s in range(0, T, 76):
            piece = z[..., s:min(s + 76, T)]
            if piece.shape[-1] < 14:
                piece = z[..., max(0, T - 14):T]
            outs.append(r16({"latents": piece})[0].float().squeeze().cpu().numpy())
        test = np.concatenate(outs)[: len(ref)]
        s_db, l_db = snr_db(ref, test), log_spectral_distance(ref, test)
        qual.append({"frames": T, "snr_db": round(s_db, 2), "lsd_db": round(l_db, 3)})
        print(f"  {T:>8}{s_db:>10.2f}{l_db:>10.3f}")
    results["decoder"]["fp16_quality"] = qual

    mean_lsd = float(np.mean([q["lsd_db"] for q in qual]))
    # Under ~1 dB LSD is generally accepted as transparent for vocoder output; over ~2 dB
    # tends to be audible as roughness.
    verdict = "transparent" if mean_lsd < 1.0 else ("marginal" if mean_lsd < 2.0 else "AUDIBLE")
    print(f"\n  mean LSD {mean_lsd:.3f} dB -> {verdict}")
    results["decoder"]["fp16_verdict"] = verdict

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
