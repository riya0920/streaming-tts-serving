"""Triton Python backend: text -> token ids."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

# The repo root is two levels above model_repo/<model>/<version>/. Adding it lets the
# backend share the exact normalization code the tests exercise, rather than a copy that
# silently drifts.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streaming.text_norm import normalize, split_for_streaming  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})

        def param(name, default):
            return params.get(name, {}).get("string_value", default)

        model_dir = param("tokenizer_dir", os.environ.get(
            "TTS_MODEL_DIR", "/workspace/models") + "/facebook__mms-tts-eng")
        self.max_chars = int(param("max_chars", "2000"))

        from transformers import VitsTokenizer
        self.tok = VitsTokenizer.from_pretrained(model_dir)

        self.logger = pb_utils.Logger
        self.logger.log_info(f"tts_frontend ready (tokenizer={model_dir})")

        # Cache the dtypes Triton expects so execute() does not re-parse the config.
        out = {o["name"]: o for o in cfg["output"]}
        self.id_dtype = pb_utils.triton_string_to_numpy(out["INPUT_IDS"]["data_type"])
        self.mask_dtype = pb_utils.triton_string_to_numpy(out["ATTENTION_MASK"]["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._one(request))
            except Exception as exc:  # noqa: BLE001
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"tts_frontend: {exc}"),
                ))
        return responses

    def _one(self, request):
        raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
        text = raw.reshape(-1)[0]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        text = text[: self.max_chars]

        norm = normalize(text)
        if not norm:
            raise ValueError("text normalized to empty")

        enc = self.tok(norm, return_tensors="np")
        ids = enc["input_ids"].astype(self.id_dtype)
        mask = enc.get("attention_mask")
        mask = (np.ones_like(ids) if mask is None else mask).astype(self.mask_dtype)

        # NORMALIZED_TEXT is returned for observability, not for the pipeline: when a
        # user reports that the audio said something odd, the first question is always
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("INPUT_IDS", ids),
            pb_utils.Tensor("ATTENTION_MASK", mask),
            pb_utils.Tensor("NORMALIZED_TEXT",
                            np.array([norm.encode("utf-8")], dtype=object)),
        ])

    def finalize(self):
        self.tok = None
