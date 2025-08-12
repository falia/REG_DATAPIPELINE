# gpu_check_ort.py
import json, os, numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

print("ORT:", ort.__version__)
print("Build has providers:", ort.get_available_providers())

# Grab the same model unstructured uses
model_path = hf_hub_download(
    repo_id="unstructuredio/detectron2_faster_rcnn_R_50_FPN_3x",
    filename="model.onnx"
)
print("Model:", model_path)

# Prefer CUDA with CPU fallback
so = ort.SessionOptions()
so.enable_profiling = True  # <-- turn on profiling
sess = ort.InferenceSession(
    model_path,
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Session providers (in use order):", sess.get_providers())

# Run a few inferences (use a dummy image)
img = Image.new("RGB", (1200, 1600), "white")
arr = np.array(img.resize((800, 1035), resample=Image.BILINEAR), dtype=np.float32).transpose(2,0,1)
inputs = {sess.get_inputs()[0].name: arr}

for _ in range(5):
    sess.run(None, inputs)

# Close profiling and inspect the JSON
profile_path = sess.end_profiling()
print("Profile file:", profile_path)

with open(profile_path) as f:
    events = json.load(f)

providers_seen = {e.get("args", {}).get("provider") for e in events if isinstance(e, dict)}
print("Providers seen in profile:", providers_seen)
