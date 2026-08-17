"""
Triton Python backend: token ids -> decoder latents.

VITS's front half — text encoder, stochastic duration predictor, flow. The 36% of GPU
time M4 could not convert to TensorRT (the duration predictor samples `randn` internally,
so its graph is not a pure function of its inputs), and — per M9 — **100% of the latency
tail**: p50 150 ms, p99 1000 ms at 32 concurrent sessions, while every other stage stays
under 20 ms.

This version batches. The first version ran with max_batch_size 0, so eight instances
each did a batch-1 forward and contended for one GPU. M2 measured this model as
launch-bound (96x the work for 1.10x the time), which is precisely the regime where
batching is nearly free — so serving eight requests as one batched forward should cost
about what one cost, and that is an ~8x on the only stage that matters.

Two things batching forces:

  - Inputs are padded to a common length by the caller (the gateway buckets to multiples
    of 16). `attention_mask` already tells VITS which positions are real, so padding is
    correct rather than merely tolerated.
  - Outputs are ragged: each utterance produces a different number of latent frames. They
    are padded to the batch maximum and the true length is returned in NUM_FRAMES, which
    tts_stream uses to ignore the padding. Without that the C++ backend would decode
    silence off the end of every short utterance in a batch.
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
        self.device = f"cuda:{args['model_instance_device_id']}" \
            if args["model_instance_kind"] == "GPU" else "cpu"

        from transformers import VitsModel
        self.model = VitsModel.from_pretrained(model_dir).to(self.device).eval()
        self.model.noise_scale = float(param("noise_scale", "0.667"))
        self.model.noise_scale_duration = float(param("noise_scale_duration", "0.8"))
        self.model.speaking_rate = float(param("speaking_rate", "1.0"))

        self.logger = pb_utils.Logger

        # Warm the shapes the gateway's bucketing can produce. cuDNN picks and caches an
        # algorithm per shape, and M3 measured first-touch at ~66 ms against ~4.9 ms
        # steady state — a spike that would otherwise land on a live request.
        with torch.inference_mode():
            for b in (1, 4, 8):
                for s in (16, 48, 96):
                    ids = torch.ones((b, s), dtype=torch.long, device=self.device)
                    capture_latents(self.model, input_ids=ids,
                                    attention_mask=torch.ones_like(ids))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        self.logger.log_info(f"vits_frontend ready on {self.device} (batching enabled)")

    def execute(self, requests):
        """Triton hands us a list of requests; we fuse them into one forward pass.

        The python backend does not auto-concatenate — dynamic batching only groups the
        requests and delivers them together. Doing the concatenation here is what turns
        that grouping into an actual batched GPU call.
        """
        try:
            return self._batched(requests)
        except Exception as exc:  # noqa: BLE001
            # One bad request must not fail the whole batch silently; return an error
            # per request so callers see something actionable.
            err = pb_utils.TritonError(f"vits_frontend: {exc}")
            return [pb_utils.InferenceResponse(output_tensors=[], error=err)
                    for _ in requests]

    def _batched(self, requests):
        ids_list, mask_list = [], []
        for request in requests:
            ids = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS").as_numpy()
            mt = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")
            mask = mt.as_numpy() if mt is not None else np.ones_like(ids)
            if ids.ndim == 1:
                ids, mask = ids[None, :], mask[None, :]
            ids_list.append(ids)
            mask_list.append(mask)

        # Requests in one batch may still differ in length if the caller bucketed
        # loosely; pad to the batch max so the concatenation is valid.
        smax = max(a.shape[-1] for a in ids_list)
        def pad(a, val):
            if a.shape[-1] == smax:
                return a
            out = np.full((a.shape[0], smax), val, dtype=a.dtype)
            out[:, : a.shape[-1]] = a
            return out

        ids = np.concatenate([pad(a, 0) for a in ids_list], axis=0)
        mask = np.concatenate([pad(m, 0) for m in mask_list], axis=0)

        input_ids = torch.from_numpy(ids.astype(np.int64)).to(self.device)
        attention_mask = torch.from_numpy(mask.astype(np.int64)).to(self.device)

        latents = capture_latents(self.model, input_ids=input_ids,
                                  attention_mask=attention_mask)
        out = latents.float().cpu().numpy()   # [B, C, Tmax]

        # True frame count per item. VITS zeroes padded positions via output_padding_mask,
        # so trailing frames are silence — but decoding them would waste GPU on every
        # short utterance batched with a long one, and emit audible trailing silence.
        frames = np.zeros((out.shape[0],), dtype=np.int32)
        for i in range(out.shape[0]):
            nz = np.nonzero(np.abs(out[i]).sum(axis=0) > 0)[0]
            frames[i] = int(nz[-1]) + 1 if nz.size else out.shape[-1]

        responses = []
        for i in range(len(requests)):
            n = int(frames[i])
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("LATENTS", out[i : i + 1, :, :n].copy()),
                pb_utils.Tensor("NUM_FRAMES", np.array([[n]], dtype=np.int32)),
            ]))
        return responses

    def finalize(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
