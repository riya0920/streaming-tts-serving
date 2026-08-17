"""TensorRT-backed VITS front half: token ids -> decoder latents."""

from __future__ import annotations

import numpy as np
import torch


class TRTFrontend:
    """Runs encoder, duration predictor and flow through TensorRT engines."""

    def __init__(self, engines_dir, runner_cls, config, device: str = "cuda",
                 noise_scale: float = 0.667, noise_scale_duration: float = 0.8,
                 speaking_rate: float = 1.0, max_frames: int = 2000,
                 min_frames: int = 16):
        from pathlib import Path

        d = Path(engines_dir)
        self.enc = runner_cls(d / "vits_encoder_fp16.plan")
        self.dur = runner_cls(d / "vits_duration_fp16.plan")
        self.flow = runner_cls(d / "vits_flow_fp16.plan")

        self.device = device
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.speaking_rate = speaking_rate
        self.channels = int(getattr(config, "flow_size", 192))
        # Must stay inside the flow engine's optimization profile (built at max 2048
        # frames). Exceeding it makes set_input_shape fail, and TensorRT reports that by
        # returning False rather than raising — the output then carries a stale shape.
        self.max_frames = int(max_frames)
        # The flow engine's profile minimum. Short replies fall under it — "Sure." is
        # ~12 frames — so anything shorter is padded and trimmed back.
        self.min_frames = int(min_frames)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[B, S] token ids -> [B, C, T] latents."""
        dev = self.device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        B = input_ids.shape[0]

        # ---- encoder ---------------------------------------------------------
        hidden, m_p, logs_p = self.enc({
            "input_ids": input_ids, "attention_mask": attention_mask,
        })
        # Promote to fp32 immediately. The engines return half precision, and the
        # alignment math that follows takes exp() of a duration — in fp16 that overflows
        hidden = hidden.float().transpose(1, 2).contiguous()
        m_p = m_p.float().transpose(1, 2).contiguous()
        logs_p = logs_p.float().transpose(1, 2).contiguous()

        input_padding_mask = attention_mask.unsqueeze(1).to(hidden.dtype)  # [B, 1, S]

        # ---- duration predictor ---------------------------------------------
        # Fresh noise, every call. This is the randomness that gives VITS its natural
        noise = torch.randn(B, 2, input_ids.shape[1], device=dev,
                            dtype=hidden.dtype) * self.noise_scale_duration
        log_duration = self.dur({
            "hidden_states": hidden,
            "padding_mask": input_padding_mask,
            "noise": noise,
        })[0].float()

        length_scale = 1.0 / self.speaking_rate
        # Clamp before exp. fp16 rounding can produce a spuriously large log-duration,
        # and exp() turns that into an enormous frame count for a single token.
        #
        # exp(5) is ~148 frames — 2.4 s of audio for one token, already far beyond any
        # real phoneme. The earlier exp(9) bound allowed ~8,100 frames from one token,
        log_duration = torch.nan_to_num(log_duration, nan=0.0, posinf=5.0, neginf=-5.0)
        log_duration = torch.clamp(log_duration, max=5.0)
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
        # Hard cap on total frames as well, so no combination of token durations can push
        # the flow engine past its profile. 2,000 frames is ~32 s of audio at this hop.
        predicted_lengths = torch.clamp_max(predicted_lengths, self.max_frames)

        # ---- alignment expansion --------------------------------------------
        # Mirrors VitsModel.forward: turn per-token durations into a hard monotonic
        T = int(predicted_lengths.max().item())
        T = max(1, min(T, self.max_frames))
        idx = torch.arange(T, device=dev, dtype=predicted_lengths.dtype)
        output_padding_mask = (idx.unsqueeze(0) < predicted_lengths.unsqueeze(1))
        output_padding_mask = output_padding_mask.unsqueeze(1).to(hidden.dtype)  # [B,1,T]

        attn_mask = input_padding_mask.unsqueeze(2) * output_padding_mask.unsqueeze(-1)
        S = input_ids.shape[1]
        cum_duration = torch.cumsum(duration, -1).view(B * S, 1)
        idx_t = torch.arange(T, device=dev, dtype=cum_duration.dtype)
        valid = (idx_t.unsqueeze(0) < cum_duration).to(hidden.dtype).view(B, S, T)
        padded = valid - torch.nn.functional.pad(valid, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded.unsqueeze(1).transpose(2, 3) * attn_mask       # [B,1,T,S]

        a = attn.squeeze(1)                                           # [B,T,S]
        m_p = torch.matmul(a, m_p.transpose(1, 2)).transpose(1, 2)    # [B,C,T]
        logs_p = torch.matmul(a, logs_p.transpose(1, 2)).transpose(1, 2)

        # ---- sample the prior ------------------------------------------------
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.noise_scale
        # Zero the tail past each item's own frame count before the flow sees it.
        #
        # Past that point m_p and logs_p are both zero, and exp(0) is 1, not 0 — so the
        # sample above fills the padding with full-scale noise rather than silence. On its
        # own that would be harmless, since the tail is trimmed at the end. But the flow is
        # a stack of dilated convolutions: with a batch built at the longest item, that
        # noise is inside the receptive field of the last real frames of every shorter
        # item, and it leaks into audio that does get played.
        z_p = z_p * output_padding_mask

        # ---- flow ------------------------------------------------------------
        # Pad up to the engine's MINIMUM profile length before calling it.
        #
        # This is the mirror of the max-side guard, and it is the one that actually fired:
        # a short reply like "Sure." is ~12 latent frames, below the flow engine's 16-frame
        #
        # The padding is trimmed straight back off, so it only ever costs a few frames of
        # wasted compute on the shortest utterances — the same trade the C++ decoder
        # backend makes for its own profile minimum.
        T_real = z_p.shape[-1]
        pad_to = max(T_real, self.min_frames)
        if pad_to > T_real:
            z_p = torch.nn.functional.pad(z_p, (0, pad_to - T_real))
            flow_mask = torch.nn.functional.pad(output_padding_mask, (0, pad_to - T_real))
        else:
            flow_mask = output_padding_mask

        latents = self.flow({
            "z_p": z_p.contiguous(),
            "padding_mask": flow_mask.contiguous(),
        })[0]

        latents = latents[..., :T_real]
        return latents * output_padding_mask


def compare_against_torch(trt_frontend, model, tok, texts, seed: int = 11) -> dict:
    """Sanity-check the TRT path against PyTorch.

    Exact agreement is impossible and would in fact be a bad sign: both paths sample
    noise, so identical output would mean the randomness had been lost somewhere. What
    is checked instead is that the two produce latents with the same shape scale and
    similar statistics, and — separately and more importantly — that repeated TRT calls
    differ from each other.
    """
    import torch as _t

    out: dict = {"per_text": []}
    for text in texts:
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        mask = enc.get("attention_mask")
        mask = (_t.ones_like(ids) if mask is None else mask.cuda())

        _t.manual_seed(seed)
        with _t.inference_mode():
            from streaming.chunked import capture_latents
            ref = capture_latents(model, input_ids=ids, attention_mask=mask)

        _t.manual_seed(seed)
        got = trt_frontend(ids, mask)

        a = ref.float().cpu().numpy()
        b = got.float().cpu().numpy()
        out["per_text"].append({
            "text": text[:40],
            "torch_frames": int(a.shape[-1]),
            "trt_frames": int(b.shape[-1]),
            "frame_ratio": round(b.shape[-1] / max(a.shape[-1], 1), 3),
            "torch_std": round(float(np.std(a)), 4),
            "trt_std": round(float(np.std(b)), 4),
        })

    # Two calls with identical input must differ — otherwise the per-request noise is
    # not reaching the duration predictor and every utterance would share one rhythm.
    enc = tok(texts[0], return_tensors="pt")
    ids = enc["input_ids"].cuda()
    mask = _t.ones_like(ids)
    l1 = trt_frontend(ids, mask).float().cpu().numpy()
    l2 = trt_frontend(ids, mask).float().cpu().numpy()
    n = min(l1.shape[-1], l2.shape[-1])
    out["noise_varies_across_calls"] = bool(
        l1.shape[-1] != l2.shape[-1]
        or float(np.abs(l1[..., :n] - l2[..., :n]).mean()) > 1e-6
    )
    return out
