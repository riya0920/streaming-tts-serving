"""
Triton Python backend: token ids -> decoder latents.

VITS's front half — text encoder, stochastic duration predictor, flow. M9 measured this
stage as **100% of the latency tail** (p50 150 ms, p99 1000 ms under load, while every
other stage stayed under 20 ms), so it, not the decoder, set the concurrency ceiling.

It now runs on TensorRT. Measured on an RTX 6000 Ada:

    duration predictor   11.465 -> 1.906 ms   6.01x
    flow (T=400)          5.742 -> 0.864 ms   6.65x

Two backends, selectable at load time
-------------------------------------
`backend=trt` uses the engines. `backend=torch` keeps the original PyTorch path, which
stays because it is the reference the TRT path is validated against and the fallback if
an engine is missing or was built for a different GPU architecture. Engines are not
portable across cards; a pod that comes up with a different GPU should degrade to a
slower correct answer rather than fail to load.

Batching still applies to either path: M2 showed the model is launch-bound, which is the
regime where batching is nearly free, and M9's ramp confirmed it doubled the knee.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streaming.chunked import capture_latents  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})

        def param(name, default):
            return params.get(name, {}).get("string_value", default)

        model_dir = param("model_dir", os.environ.get(
            "TTS_MODEL_DIR", "/workspace/models") + "/facebook__mms-tts-eng")
        engines_dir = param("engines_dir", os.environ.get(
            "TTS_MODEL_DIR", "/workspace/models") + "/engines")
        want = param("backend", "trt").lower()

        self.device = f"cuda:{args['model_instance_device_id']}" \
            if args["model_instance_kind"] == "GPU" else "cpu"
        self.logger = pb_utils.Logger

        from transformers import VitsModel
        self.model = VitsModel.from_pretrained(model_dir).to(self.device).eval()
        self.model.noise_scale = float(param("noise_scale", "0.667"))
        self.model.noise_scale_duration = float(param("noise_scale_duration", "0.8"))
        self.model.speaking_rate = float(param("speaking_rate", "1.0"))

        self.trt = None
        if want == "trt":
            try:
                from export.build_trt import TRTRunner
                from streaming.trt_frontend import TRTFrontend
                self.trt = TRTFrontend(
                    engines_dir, TRTRunner, self.model.config, self.device,
                    noise_scale=self.model.noise_scale,
                    noise_scale_duration=self.model.noise_scale_duration,
                    speaking_rate=self.model.speaking_rate,
                )
                self.logger.log_info("vits_frontend: TensorRT path active")
            except Exception as exc:  # noqa: BLE001
                # Engines are architecture-specific and may simply not be present. A
                # slower correct answer beats refusing to load.
                self.logger.log_warn(
                    f"vits_frontend: TRT unavailable ({exc}); falling back to PyTorch")
                self.trt = None

        # Warm the shapes the gateway's 16-token bucketing produces. cuDNN and TensorRT
        # both pick an algorithm per shape on first use; M3 measured that first touch at
        # ~66 ms against ~4.9 ms steady state, which would otherwise land on a live
        # request as a latency spike.
        with torch.inference_mode():
            for b in (1, 4, 8):
                for s in (16, 48, 96):
                    ids = torch.ones((b, s), dtype=torch.long, device=self.device)
                    mask = torch.ones_like(ids)
                    try:
                        self._latents(ids, mask)
                    except Exception:  # noqa: BLE001
                        pass
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        self.logger.log_info(
            f"vits_frontend ready on {self.device} "
            f"(backend={'trt' if self.trt else 'torch'}, batching on)")

    def _latents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.trt is None:
            return capture_latents(self.model, input_ids=input_ids,
                                   attention_mask=attention_mask)

        # One item at a time through the TRT path.
        #
        # The batched alignment is wrong for B>1: each item predicts its own duration and
        # therefore its own frame count, so building one alignment matrix across the batch
        # mixes a padded token length with another item's frame count. Under load this
        # produced "size of tensor a (208) must match tensor b (688)" on ~24% of requests,
        # where 208 is a bucketed token length and 688 a frame count. Single requests
        # never hit it, which is why it survived the unit check.
        #
        # Looping is correct by construction. It gives up cross-request batching in this
        # stage, but the stage is now ~6x faster per call, so the trade is worth taking
        # over shipping a batched path that is wrong 24% of the time. Batching the
        # variable-length alignment properly is a separate change.
        outs = []
        for i in range(input_ids.shape[0]):
            outs.append(self.trt(input_ids[i:i + 1], attention_mask[i:i + 1]))
        if len(outs) == 1:
            return outs[0]
        # Pad to the longest so the caller's per-item trim (via NUM_FRAMES) still works.
        tmax = max(o.shape[-1] for o in outs)
        padded = []
        for o in outs:
            if o.shape[-1] < tmax:
                o = torch.nn.functional.pad(o, (0, tmax - o.shape[-1]))
            padded.append(o)
        return torch.cat(padded, dim=0)

    def execute(self, requests):
        try:
            return self._batched(requests)
        except Exception as exc:  # noqa: BLE001
            err = pb_utils.TritonError(f"vits_frontend: {exc}")
            return [pb_utils.InferenceResponse(output_tensors=[], error=err)
                    for _ in requests]

    def _batched(self, requests):
        """Fuse the grouped requests into one forward pass.

        Triton's dynamic batcher delivers requests together but does not concatenate
        them; doing that here is what turns grouping into an actual batched call.
        """
        ids_list, mask_list = [], []
        for request in requests:
            ids = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS").as_numpy()
            mt = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")
            mask = mt.as_numpy() if mt is not None else np.ones_like(ids)
            if ids.ndim == 1:
                ids, mask = ids[None, :], mask[None, :]
            ids_list.append(ids)
            mask_list.append(mask)

        smax = max(a.shape[-1] for a in ids_list)

        def pad(a):
            if a.shape[-1] == smax:
                return a
            out = np.zeros((a.shape[0], smax), dtype=a.dtype)
            out[:, : a.shape[-1]] = a
            return out

        ids = np.concatenate([pad(a) for a in ids_list], axis=0)
        mask = np.concatenate([pad(m) for m in mask_list], axis=0)

        input_ids = torch.from_numpy(ids.astype(np.int64)).to(self.device)
        attention_mask = torch.from_numpy(mask.astype(np.int64)).to(self.device)

        latents = self._latents(input_ids, attention_mask)
        out = latents.float().cpu().numpy()   # [B, C, Tmax]

        # True frame count per item. VITS zeroes padded frames, so trailing silence is
        # detectable — and must be trimmed, or every short utterance batched with a long
        # one would have the decoder spend GPU time on silence and emit it as audio.
        frames = np.zeros((out.shape[0],), dtype=np.int32)
        for i in range(out.shape[0]):
            nz = np.nonzero(np.abs(out[i]).sum(axis=0) > 0)[0]
            frames[i] = int(nz[-1]) + 1 if nz.size else out.shape[-1]

        responses = []
        for i in range(len(requests)):
            n = max(1, int(frames[i]))
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("LATENTS", out[i : i + 1, :, :n].copy()),
                pb_utils.Tensor("NUM_FRAMES", np.array([[n]], dtype=np.int32)),
            ]))
        return responses

    def finalize(self):
        self.trt = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
