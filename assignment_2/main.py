import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Import tqdm for progress bar

# Define the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Input to Hidden layer
        self.layer2 = nn.Linear(128, 64)   # Hidden to Hidden layer
        self.layer3 = nn.Linear(64, 10)    # Hidden to Output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for layer 1
        x = torch.relu(self.layer2(x))  # ReLU activation for layer 2
        x = self.layer3(x)              # Output layer (no activation here)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the neural network
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)  # Optimizer

# Initialize loss tracking for dynamic plotting
losses = []

# Create directories for saving model and images
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('image'):
    os.makedirs('image')

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    # Wrap the trainloader with tqdm to show progress bar
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
        # Flatten the inputs to a 1D tensor for the fully connected layers
        inputs = inputs.view(-1, 28*28)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(trainloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    # Update the plot in real-time
    plt.clf()  # Clear the current figure
    plt.plot(range(1, epoch+2), losses, label="Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot

    # Save the loss curve image after each epoch
    loss_image_path = f'image/loss_epoch_{epoch+1}.png'
    plt.savefig(loss_image_path)

# Show final plot after training
plt.ioff()  # Turn off interactive mode
plt.show()
plt.close()

# Save the trained model
model_save_path = 'model/mnist_model.pth'
torch.save(model.state_dict(), model_save_path)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.view(-1, 28*28)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
