"""Minimal torchvision stand-in for the TTS serving environment.

Why this exists
---------------
`transformers` imports torchvision eagerly on the way to any model class:

    transformers/__init__ -> loss.loss_utils -> loss_d_fine -> loss_for_object_detection
      -> image_transforms -> image_utils -> `from torchvision.io import decode_image`

The Triton 24.08 image ships a torchvision built against CUDA 13 while its torch is
cu128, and the venv we serve from runs torch 2.6+cu124. Every combination attempted —
matching the venv's torchvision to its torch, removing it so the system one is used,
pinning transformers to 4.x — ends in either `operator torchvision::nms does not exist`
or `libcudart.so.13: cannot open shared object file`. The image is internally
inconsistent for this import path and cannot be reconciled from the venv.

This system synthesizes speech. It never decodes an image, never runs NMS, and never
touches a single torchvision op. The import is pure collateral damage from a dependency
chain meant for object-detection models.

So: satisfy the import surface transformers actually reaches, and nothing else. Placed
first on the backend's path so it shadows the broken real package.

If an image model is ever added to this repository, delete this stub and fix the
environment properly — it would silently do the wrong thing.
"""

__version__ = "0.21.0+stub"


def _unavailable(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchvision.{name} is stubbed out in this serving image "
            "(see scripts/torchvision_stub/__init__.py). This is a speech-only "
            "deployment; if you need real torchvision, fix the image instead."
        )
    return _raise


# transformers touches these attributes during import even when they go unused.
def set_image_backend(*_a, **_k):
    return None


def get_image_backend():
    return "PIL"


class _OpsNamespace:
    def __getattr__(self, name):
        return _unavailable(f"ops.{name}")


ops = _OpsNamespace()
