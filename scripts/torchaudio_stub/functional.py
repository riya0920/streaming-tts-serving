"""torchaudio.functional stand-in — see the package docstring for why this exists."""


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchaudio.functional.{name} is stubbed out in this serving image. "
            "This deployment synthesizes audio and never decodes it; "
            "see scripts/torchaudio_stub/."
        )
    return _raise


resample = _stubbed("resample")
spectrogram = _stubbed("spectrogram")
melscale_fbanks = _stubbed("melscale_fbanks")
amplitude_to_DB = _stubbed("amplitude_to_DB")
compute_deltas = _stubbed("compute_deltas")
gain = _stubbed("gain")
highpass_biquad = _stubbed("highpass_biquad")
lowpass_biquad = _stubbed("lowpass_biquad")
