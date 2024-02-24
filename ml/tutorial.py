import torch
import torchvision

my_module = torchvision.models.resnet18()
sm = torch.jit.script(my_module)
sm.save("tutorial.pt")