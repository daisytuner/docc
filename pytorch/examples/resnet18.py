import torch
import torchvision.models as models
import PIL.Image as Image
import requests

import docc.pytorch

image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

weights = models.resnet.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

image = Image.open(requests.get(image_url, stream=True).raw)
transforms = weights.transforms()
input_tensor = transforms(image).unsqueeze(0)

with torch.no_grad():
    program = torch.compile(model, backend="docc", options={"target": "openmp"})
    output_tensor = program(input_tensor)

probs = torch.softmax(output_tensor, dim=1)
print(f"Probability of Samoyed (dog): {(probs[0, 258] * 100):.3f}%")
