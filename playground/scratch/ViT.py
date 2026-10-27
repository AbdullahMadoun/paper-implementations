import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import OxfordIIITPet

# Optional but highly recommended for ViT implementation (patchification, etc.)
# If you don't have it installed, you can use pip install einops
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    print("Warning: einops is not installed. It is highly recommended for building ViT from scratch.")
to_tensor = [Resize((144,144)), ToTensor()]

class Compose(object): 
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target): 
        for t in self.transforms:
            image = t(image)
        return image, target
def show_images(images, num_samples = 20, cols = 4): 
    plt.figure(figsize=(15,15))
    idx = int(len(dataset) / num_samples)
    print(images)
    for i, img in enumerate(images): 
        if i%idx == 0: 
            plt.subplot(4, 4, i+1) 
            plt.imshow(to_pil_image(img))
dataset = OxfordIIITPet(root = "", download=True, transform=Compose(to_tensor))
show_images(dataset)
