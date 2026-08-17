"""torchvision.transforms.functional stand-in — see the package docstring.

`transformers.image_utils` imports `pil_to_tensor` at module scope. It must exist as a
real submodule (transformers does `from torchvision.transforms.functional import ...`),
which is why transforms is a package rather than a single module.
"""


def _stubbed(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torchvision.transforms.functional.{name} is stubbed out in this serving "
            "image. This deployment synthesizes speech; see scripts/torchvision_stub/."
        )
    return _raise


pil_to_tensor = _stubbed("pil_to_tensor")
to_pil_image = _stubbed("to_pil_image")
to_tensor = _stubbed("to_tensor")
normalize = _stubbed("normalize")
resize = _stubbed("resize")
center_crop = _stubbed("center_crop")
rgb_to_grayscale = _stubbed("rgb_to_grayscale")
