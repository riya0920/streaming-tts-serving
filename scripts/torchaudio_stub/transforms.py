"""torchaudio.transforms stand-in — see the package docstring for why this exists."""


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchaudio.transforms.{name} is stubbed out in this serving image. "
            "This deployment synthesizes audio and never decodes it; "
            "see scripts/torchaudio_stub/."
        )
    return _raise


Resample = _stubbed("Resample")
Spectrogram = _stubbed("Spectrogram")
MelSpectrogram = _stubbed("MelSpectrogram")
MFCC = _stubbed("MFCC")
AmplitudeToDB = _stubbed("AmplitudeToDB")
Fade = _stubbed("Fade")
Vol = _stubbed("Vol")
