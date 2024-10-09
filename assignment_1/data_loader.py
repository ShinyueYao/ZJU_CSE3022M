import torch
import torchvision.transforms as transforms
from torchvision import datasets
import random
import numpy as np

# Load MNIST dataset
def load_mnist_data():
    
    # Define a transform to normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    # Load the MNIST dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    print(f"Loaded {len(train_data)} MNIST images.")
    
    return train_data

# Sample at least 200 images per class
def sample_images(train_data, samples_per_class=200):
    
    points = []
    
    # loop for digit 0 to 9
    for digit in range(10):
        digit_images = [train_data[i][0].numpy().flatten() for i in range(len(train_data)) if train_data.targets[i] == digit]
        
        # randomly choose at least 200 images in certain class
        sampled_images = random.sample(digit_images, samples_per_class)
        
        # add images to the list
        points.extend(sampled_images)
    
    print(f"Sampled {len(points)} images, {samples_per_class} from each class.")
    
    return np.array(points) # Return sampled images as NumPy array
