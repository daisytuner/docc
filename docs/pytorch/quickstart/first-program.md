---
title: Run Your First Model
sidebar_label: First Program
sidebar_position: 2
description: Compile ResNet-18 with the docc backend and run it on CPU cores and GPUs
---

Throughout this guide, we use one running example: the image classifier **ResNet-18** from torchvision, which we ask whether a picture shows a dog. The full script is available in the [docc repository](https://github.com/daisytuner/docc/blob/main/pytorch/examples/resnet18.py).

```python title="resnet18.py"
import torch
import torchvision.models as models
import PIL.Image as Image
import requests

import docc.pytorch  # registers the "docc" backend

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights).eval()

image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
x = weights.transforms()(image).unsqueeze(0)

with torch.no_grad():
    compiled = torch.compile(model, backend="docc", options={"target": "openmp"})
    y = compiled(x)

probs = torch.softmax(y, dim=1)
print(f"Samoyed: {probs[0, 258] * 100:.1f}%")
```

## Step 1: Run it

```bash
pip install pillow requests
python resnet18.py
```

Compared to plain PyTorch, only two lines are new: `import docc.pytorch` and the `torch.compile` call. The first call of `compiled` takes a while, because docc compiles the whole model to native code that runs on all CPU cores. Later calls with inputs of the same shape reuse the compiled code.

:::note
docc currently supports **inference only**. Run compiled models inside `torch.no_grad()` (or `torch.inference_mode()`). With gradients enabled, `torch.compile` raises an error.
:::

## Step 2: Run it on a GPU

Change the target, and move the model and the input to the GPU as usual. AMD GPUs also use the `cuda` device in PyTorch:

```python
model = model.to("cuda")
x = x.to("cuda")

with torch.no_grad():
    compiled = torch.compile(model, backend="docc", options={"target": "cuda"})  # or "rocm"
    y = compiled(x)
```

The input and output stay on the GPU, so you can chain compiled models and PyTorch operations without copies. If you keep the model on the CPU, docc copies the data to the GPU and back on every call and warns about it.

## What happened

`torch.compile` captured the model as a graph and handed it to docc. docc lowered the graph to PyTorch's core operators, translated them into its intermediate representation (SDFG), fused and parallelized the operations for the target, and compiled the generated code with Clang. The compiled model behaves like the original one: same inputs, same outputs.

## Next step

To build the backend yourself, continue with [Custom Installation](../custom-installation.md). To control how docc compiles ResNet-18, see [Usage](../usage.md).
