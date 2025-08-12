import os, inspect, onnxruntime as ort
import unstructured_inference
import unstructured_inference.models.detectron2onnx as d2

print("pkg file:", unstructured_inference.__file__)
print("model file:", d2.__file__)
print("ORT version:", ort.__version__)
print("Wheel available providers:", ort.get_available_providers())   # what's compiled into your wheel
try:
    from onnxruntime.capi import _pybind_state as C
    print("Build providers:", C.get_available_providers())           # what the build can load
except Exception as e:
    print("Build providers: <unavailable>", e)

# Instantiate the model (uses your patched initialize)
from unstructured_inference.models.base import get_model
m = get_model("detectron2_onnx")
print("Session providers (in use order):", m.model.get_providers())
