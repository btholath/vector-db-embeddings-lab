import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

def image_embeddings():
    # ResNet18 is a convolutional neural network architecture
    # It's designed to handle the vanishing gradient problem in deep networks
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # pretrained=True: Uses weights pre-trained on ImageNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last fully connected layer
    model.eval()
    
    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open("1.jpg")  
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(img_tensor).squeeze()
    
    print("Image Embedding shape:", embedding.shape)
    print("First few dimensions of image embedding:", embedding[:5])

# Run function
image_embeddings()