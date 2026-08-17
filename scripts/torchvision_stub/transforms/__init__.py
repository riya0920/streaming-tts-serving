"""torchvision.transforms stand-in — see the package docstring for why this exists.

`transformers.image_utils` imports InterpolationMode at module scope, so it must exist
even for a speech-only deployment that never resizes an image. The member values match
upstream torchvision so anything that merely reads them behaves identically.
"""

from enum import Enum


class InterpolationMode(Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchvision.transforms.{name} is stubbed out in this serving image. "
            "This deployment synthesizes speech; see scripts/torchvision_stub/."
        )
    return _raise


class _CallableStub:
    def __init__(self, name):
        self._name = name

    def __call__(self, *_a, **_k):
        _stubbed(self._name)()

    def __getattr__(self, item):
        return _stubbed(f"{self._name}.{item}")


Compose = _CallableStub("Compose")
Resize = _CallableStub("Resize")
Normalize = _CallableStub("Normalize")
ToTensor = _CallableStub("ToTensor")
CenterCrop = _CallableStub("CenterCrop")


from . import functional  # noqa: E402,F401  (must be a real submodule)
