import torch
import transformers
import PIL.Image as Image
from PIL.ImageFile import ImageFile
import requests

from tests import check


def get_bus_image() -> ImageFile:
    image_url: str = "https://farm4.staticflickr.com/3793/9071762311_17e92f4cb6_z.jpg"

    return Image.open(requests.get(image_url, stream=True).raw)


# --- SegFormer (b0-sized) model fine-tuned on CityScapes ---


def test_segformer_b0_finetuned_cityscapes_1024_1024_simple(target: str) -> None:
    model: transformers.SegformerForSemanticSegmentation = (
        transformers.SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
    )
    check(
        model.eval(),
        torch.randn(8, 3, 512, 512),
        kwargs={
            "output_hidden_states": True,
            "return_dict": False,
        },
        target=target,
        atol=3e-5,
    )


def test_segformer_b0_finetuned_cityscapes_1024_1024_image(target: str) -> None:
    processor: transformers.SegformerImageProcessor = (
        transformers.SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
    )
    model: transformers.SegformerForSemanticSegmentation = (
        transformers.SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
    )

    image: ImageFile = get_bus_image()
    input_tensor: torch.Tensor = processor(image, return_tensors="pt")["pixel_values"]
    check(
        model.eval(),
        input_tensor,
        kwargs={
            "output_hidden_states": True,
            "return_dict": False,
        },
        target=target,
        atol={
            "none": 5e-5,
            "sequential": 7e-5,
            "openmp": 3e-5,
            "cuda": 3e-5,
            "rocm": 6e-5,
        }[target],
    )
