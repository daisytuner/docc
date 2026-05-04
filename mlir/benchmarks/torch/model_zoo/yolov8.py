import torch
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

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("Error: ultralytics not installed. Please install with: pip install ultralytics")
    sys.exit(1)

import docc.torch

docc.torch.set_backend_options(target="openmp", category="server")


def compare_outputs(res, res_ref, rtol=1e-2, atol=1e-3):
    """Recursively compare outputs that may be nested structures of tensors."""
    if isinstance(res, torch.Tensor):
        return torch.allclose(res, res_ref, rtol=rtol, atol=atol)
    elif isinstance(res, dict):
        if res.keys() != res_ref.keys():
            return False
        return all(compare_outputs(res[key], res_ref[key], rtol, atol) for key in res.keys())
    elif isinstance(res, (list, tuple)):
        if len(res) != len(res_ref):
            return False
        return all(compare_outputs(r, r_ref, rtol, atol) for r, r_ref in zip(res, res_ref))
    else:
        # For other types, use standard equality
        return res == res_ref


# Load a pre-trained YOLOv8 model
print("Loading YOLOv8n model...")
yolo = YOLO('yolov8n.pt')
model = yolo.model
model.eval()

print("\n=== Selective Compilation Strategy ===")
print("Strategy: Compile ONLY the backbone with docc backend")
print()

# Compile only the backbone with docc
print("--- Compile Times ---")
start = time.perf_counter()
compiled_backbone = torch.compile(model.model[0], backend="docc")
docc_compile_time = time.perf_counter() - start
print(f"Backbone compile (docc): {docc_compile_time:.4f} s")

# Replace the backbone with compiled version
# Now model.model[0] is compiled with docc, rest of model runs eagerly
model.model[0] = compiled_backbone

# Create reference model for comparison
yolo_ref = YOLO('yolov8n.pt')
model_ref = yolo_ref.model
model_ref.eval()
start = time.perf_counter()
program_ref = torch.compile(model_ref)
torch_compile_time = time.perf_counter() - start
print(f"Full model compile (torch): {torch_compile_time:.4f} s")

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"  # dog (Samoyed)

image = Image.open(requests.get(url, stream=True).raw)

# YOLO expects 640x640 images - convert PIL image to tensor
import numpy as np
image_np = np.array(image)
input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
# Normalize to [0, 1]
input_tensor = input_tensor / 255.0
# Resize to 640x640 (YOLO standard input size)
input_tensor = torch.nn.functional.interpolate(
    input_tensor.unsqueeze(0), 
    size=(640, 640), 
    mode='bilinear', 
    align_corners=False
)

if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
    _ = plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

# Force dynamo (inference) backend
with torch.no_grad():
    print("\n--- Execution Times ---")
    print("DOCC):")
    start = time.perf_counter()
    res = model(input_tensor)
    mixed_exec_time = time.perf_counter() - start
    print(f"  First execution: {mixed_exec_time:.4f} s")

    print("\nTorch:")
    start = time.perf_counter()
    ref = program_ref(input_tensor)
    torch_exec_time = time.perf_counter() - start
    print(f"  First execution: {torch_exec_time:.4f} s")

    print("DOCC:")
    start = time.perf_counter()
    res = model(input_tensor)
    mixed_exec_time = time.perf_counter() - start
    print(f"  Second execution: {mixed_exec_time:.4f} s")

    print("\nTorch:")
    start = time.perf_counter()
    ref = program_ref(input_tensor)
    torch_exec_time = time.perf_counter() - start
    print(f"  Second execution: {torch_exec_time:.4f} s")

    print("\n--- Results Comparison ---")

    # YOLO returns a tuple of outputs at different scales
    if isinstance(res, tuple):
        all_match = True
        for i, (r, r_ref) in enumerate(zip(res, ref)):
            scale_match = compare_outputs(r, r_ref, rtol=1e-2, atol=1e-3)
            if not scale_match:
                print(f"Scale {i} feature map differs")
                if isinstance(r, torch.Tensor):
                    max_diff = torch.max(torch.abs(r - r_ref))
                    print(f"  Max absolute difference: {max_diff:.6f}")
                all_match = False

        assert all_match, "Feature maps don't match between docc and torch"
    else:
        # Single output tensor
        output_match = compare_outputs(res, ref, rtol=1e-2, atol=1e-3)
        if not output_match:
            if isinstance(res, torch.Tensor):
                max_diff = torch.max(torch.abs(res - ref))
                print(f"  Max absolute difference: {max_diff:.6f}")

        assert output_match, "Outputs don't match between docc and torch inductor"

# Visualize predictions using ultralytics API (only in interactive mode, not CI)
if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
    print("\n--- Visualizing Detections ---")

    # Run inference using ultralytics API for visualization
    yolo_viz = YOLO('yolov8n.pt')
    yolo_viz.model.model[0] = compiled_backbone  # Use compiled backbone

    results = yolo_viz.predict(source=image, conf=0.25)
    results_ref = yolo_ref.predict(source=image, conf=0.25)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot docc results
    ax1.imshow(results[0].plot())
    ax1.set_title("DOCC (Mixed compilation)")
    ax1.axis("off")

    # Plot torch inductor results
    ax2.imshow(results_ref[0].plot())
    ax2.set_title("Reference")
    ax2.axis("off")

    plt.tight_layout()

    # Also try to show it
    plt.show(block=True)

    print(f"\nDOCC detected {len(results[0].boxes)} objects")
    print(f"Torch inductor detected {len(results_ref[0].boxes)} objects")
else:
    print("\nSkipping visualization in CI environment")
