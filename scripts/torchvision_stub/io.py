"""torchvision.io stand-in — see the package docstring for why this exists.

Only the names `transformers.image_utils` imports are provided. Calling any of them
raises loudly rather than silently returning something wrong, because reaching one means
an image model was added to a speech-only deployment and the stub needs removing.
"""

from enum import Enum


class ImageReadMode(Enum):
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchvision.io.{name} is stubbed out in this serving image. "
            "This deployment synthesizes speech and never decodes images; "
            "see scripts/torchvision_stub/__init__.py."
        )
    return _raise


decode_image = _stubbed("decode_image")
read_image = _stubbed("read_image")
decode_jpeg = _stubbed("decode_jpeg")
encode_jpeg = _stubbed("encode_jpeg")
write_jpeg = _stubbed("write_jpeg")
read_file = _stubbed("read_file")
write_file = _stubbed("write_file")
