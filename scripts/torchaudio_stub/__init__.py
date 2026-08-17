"""Minimal torchaudio stand-in for the TTS serving environment.

Same problem as the torchvision stub next door, different package. `transformers`
imports torchaudio at module scope in `audio_utils.py` whenever it is importable:

    transformers.models.vits.modeling_vits -> modeling_layers -> processing_utils
      -> audio_utils -> `import torchaudio`

The Triton 24.08 image's torchaudio is built against its own torch (2.11+cu128) and
loads a CUDA 13 runtime; the venv this serves from runs torch 2.6+cu124. The extension
load fails with `libcudart.so.13: cannot open shared object file` before any of our code
runs, and no combination of venv-side pins reconciles it — the image is internally
inconsistent for this path.

What transformers actually needs from torchaudio here is only that the *import* succeed.
`torchaudio.load` and `torchaudio.functional.resample` are referenced solely inside
`load_audio()`, which this system never calls: it synthesizes audio from text and never
decodes an audio file. So satisfy the import surface and make every real entry point
fail loudly instead of silently.

Remove this stub if audio *input* is ever added (ASR, voice conversion, anything reading
a waveform); at that point the environment has to be fixed properly.
"""

__version__ = "2.6.0+stub"


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchaudio.{name} is stubbed out in this serving image "
            "(see scripts/torchaudio_stub/__init__.py). This deployment only "
            "synthesizes audio and never decodes it. Fix the image if you need real "
            "torchaudio."
        )
    return _raise


load = _stubbed("load")
save = _stubbed("save")
info = _stubbed("info")
list_audio_backends = lambda: []          # noqa: E731 - trivial, matches upstream shape
set_audio_backend = _stubbed("set_audio_backend")
get_audio_backend = lambda: None          # noqa: E731

from . import functional  # noqa: E402,F401  (must be a real submodule)
from . import transforms  # noqa: E402,F401
