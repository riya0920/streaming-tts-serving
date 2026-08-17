"""
Triton Python backend: token ids -> decoder latents.

This is VITS's front half — text encoder, stochastic duration predictor, flow — the
36% of GPU time that M4 could not convert to TensorRT. The duration predictor samples
`randn` internally, so its graph is not a pure function of its inputs, and the flow
layers are the numerically touchy part in half precision. Both stay in FP32 PyTorch.

It is also, per M2, where time-to-first-audio actually goes: 114 ms of a 121 ms TTFA,
against 7 ms for the first audio chunk. Optimizing the decoder further does nothing for
latency until this stage moves.

Output is latents [1, C, T_frames], which `tts_stream` (C++) then decodes in chunks.
Splitting the pipeline here is what lets the expensive, streamable part live in C++ while
the awkward, stateful, PyTorch-only part stays in Python.
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

        # Warm up. The first CUDA call pays cuBLAS/cuDNN autotuning; without this the
        # first real request eats tens of milliseconds that belong to nobody. M3 found
        # the same effect per tensor shape in the decoder.
        with torch.inference_mode():
            warm = torch.ones((1, 16), dtype=torch.long, device=self.device)
            for _ in range(3):
                capture_latents(self.model, input_ids=warm,
                                attention_mask=torch.ones_like(warm))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        self.logger.log_info(f"vits_frontend ready on {self.device} ({model_dir})")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._one(request))
            except Exception as exc:  # noqa: BLE001
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"vits_frontend: {exc}"),
                ))
        return responses

    def _one(self, request):
        ids = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS").as_numpy()
        mask_t = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")
        mask = mask_t.as_numpy() if mask_t is not None else np.ones_like(ids)

        input_ids = torch.from_numpy(ids.astype(np.int64)).to(self.device)
        attention_mask = torch.from_numpy(mask.astype(np.int64)).to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        latents = capture_latents(self.model, input_ids=input_ids,
                                  attention_mask=attention_mask)

        # Handed to a C++ backend next, so send FP32 on the host. Keeping it on-device
        # would be faster but Triton would still have to move it between processes;
        # revisit with CUDA shared memory if this shows up in a trace.
        out = latents.float().cpu().numpy()
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("LATENTS", out),
            pb_utils.Tensor("NUM_FRAMES", np.array([out.shape[-1]], dtype=np.int32)),
        ])

    def finalize(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
