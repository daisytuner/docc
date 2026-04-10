import torch
import torchvision.models as models
import time
import os
import sys

# Use Agg backend for headless CI environments
if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import PIL.Image as Image
import requests

import docc.torch

# Load a pre-trained Faster R-CNN model
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
transforms = weights.transforms()

print("\n=== Selective Compilation Strategy ===")
print("Strategy: Compile ONLY the backbone with docc backend")
print("          RPN and ROI heads run eagerly through PyTorch")
print()

# Compile only the backbone with docc
print("--- Compile Times ---")
start = time.perf_counter()
compiled_backbone = torch.compile(model.backbone, backend="docc")
docc_compile_time = time.perf_counter() - start
print(f"Backbone compile (docc): {docc_compile_time:.4f} s")

# Replace the backbone with compiled version
# Now model.backbone is compiled with docc, rest of model runs eagerly
model.backbone = compiled_backbone

# Create reference model for comparison
model_ref = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model_ref.eval()
start = time.perf_counter()
program_ref = torch.compile(model_ref)
torch_compile_time = time.perf_counter() - start
print(f"Full model compile (torch inductor): {torch_compile_time:.4f} s")

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"  # dog (Samoyed)

image = Image.open(requests.get(url, stream=True).raw)


input_tensor = transforms(image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
    _ = plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

# Force dynamo (inference) backend
with torch.no_grad():
    print("\n--- Execution Times ---")
    print("Mixed compilation (docc backbone + eager RPN/ROI):")
    start = time.perf_counter()
    res = model(input_batch)
    mixed_exec_time = time.perf_counter() - start
    print(f"  First execution: {mixed_exec_time:.4f} s")

    print("\nTorch inductor (full model):")
    start = time.perf_counter()
    ref = program_ref(input_batch)
    torch_exec_time = time.perf_counter() - start
    print(f"  First execution: {torch_exec_time:.4f} s")

    print("\n--- Results Comparison ---")
    # Compare outputs - these assertions will fail the test if results don't match
    boxes_match = torch.allclose(res[0]['boxes'], ref[0]['boxes'], rtol=1e-2)
    if boxes_match:
        print("✓ Boxes match between docc and torch")
    else:
        print("✗ Boxes differ with a relative difference of:")
        print((res[0]['boxes'] - ref[0]['boxes']) / ref[0]['boxes'])

    scores_match = torch.allclose(res[0]['scores'], ref[0]['scores'], rtol=1e-2)
    if scores_match:
        print("✓ Scores match between docc and torch")
    else:
        print("✗ Scores differ with a relative difference of:")
        print((res[0]['scores'] - ref[0]['scores']) / ref[0]['scores'])

    print(f"\nDetected {len(res[0]['boxes'])} objects with docc")
    print(f"Detected {len(ref[0]['boxes'])} objects with torch inductor")

    # Assert for CI - fail if results don't match
    assert boxes_match, "Bounding boxes don't match between docc and torch inductor"
    assert scores_match, "Detection scores don't match between docc and torch inductor"
    print("\n✓ All assertions passed!")


# Draw the predicted bounding boxes (only in interactive mode, not CI)
if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
    # COCO labels for object detection
    labels = weights.meta["categories"]

    def draw_boxes(image, boxes, classes, labels, scores, threshold=0.4, title="Predicted Bounding Boxes"):
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        ax = plt.gca()

        box_labels = [labels[min(i, len(labels) - 1)] for i in classes]

        for box, label, score in zip(boxes, box_labels, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 10, f"{label}: {score:.2f}", color="red", fontsize=12)

        plt.axis("off")
        plt.title(title)
        plt.show()

    draw_boxes(image, ref[0]["boxes"].cpu().numpy(), ref[0]["labels"].cpu().numpy(), labels, ref[0]["scores"].cpu().numpy(), threshold=0.5, title="Torch Detections")
    draw_boxes(image, res[0]["boxes"].cpu().numpy(), res[0]["labels"].cpu().numpy(), labels, res[0]["scores"].cpu().numpy(), threshold=0.5, title="Q.ANT Detections")
else:
    print("\nSkipping visualization in CI environment")
