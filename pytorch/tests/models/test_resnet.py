import torch
import torchvision.models as models
import PIL.Image as Image
from PIL.ImageFile import ImageFile
import requests

from tests import check


def get_dog_image() -> ImageFile:
    image_url: str = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

    return Image.open(requests.get(image_url, stream=True).raw)


# --- resnet18 ---


def test_resnet18_simple(target: str) -> None:
    torch._dynamo.reset_code_caches()
    model: models.resnet.ResNet = models.resnet18(
        weights=models.resnet.ResNet18_Weights.DEFAULT
    )
    check(model.eval(), torch.randn(1, 3, 224, 224), target=target)


def test_resnet18_batched(target: str) -> None:
    torch._dynamo.reset_code_caches()
    model: models.resnet.ResNet = models.resnet18(
        weights=models.resnet.ResNet18_Weights.DEFAULT
    )
    check(model.eval(), torch.randn(4, 3, 224, 224), target=target)


def test_resnet18_image(target: str) -> None:
    torch._dynamo.reset_code_caches()

    weights: models.resnet.ResNet18_Weights = models.resnet.ResNet18_Weights.DEFAULT
    model: models.resnet.ResNet = models.resnet18(weights=weights)

    image: ImageFile = get_dog_image()
    transforms = weights.transforms()
    input_tensor: torch.Tensor = transforms(image).unsqueeze(0)

    check(model.eval(), input_tensor, target=target)


def test_resnet18_image_batched(target: str) -> None:
    torch._dynamo.reset_code_caches()

    weights: models.resnet.ResNet18_Weights = models.resnet.ResNet18_Weights.DEFAULT
    model: models.resnet.ResNet = models.resnet18(weights=weights)

    image: ImageFile = get_dog_image()
    transforms = weights.transforms()
    input_tensor: torch.Tensor = transforms(image).unsqueeze(0)
    input_batch = torch.cat((input_tensor, input_tensor, input_tensor, input_tensor))

    check(model.eval(), input_batch, target=target)
