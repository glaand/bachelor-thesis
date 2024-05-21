import torch
import torch.onnx

from model import Model

# Create random input tensor of size 32x32x1x1
x = torch.randn(1, 1, 32, 32, requires_grad=True)

# Create Kaneda model
model = Model()

# Export the model
torch.onnx.export(model, x, "models/model.onnx", export_params=True, verbose=True)
