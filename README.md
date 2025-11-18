# Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train = datasets.MNIST(root='.', train=True, transform=t, download=True)
test = datasets.MNIST(root='.', train=False, transform=t, download=True)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
loss_list = []
acc_list = []

for e in range(epochs):
    for x, y in train_loader:
        p = model(x)
        l = loss_fn(p, y)
        opt.zero_grad()
        l.backward()
        opt.step()
    loss_list.append(l.item())

    model.eval()
    c = 0
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x)
            _, pr = torch.max(p, 1)
            c += (pr == y).sum().item()
            n += y.size(0)
    acc_list.append(c/n)
    model.train()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_list)
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(acc_list)
plt.title("Accuracy")

plt.show()

model.eval()
wimg, wpr, wtr = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        p = model(x)
        _, pr = torch.max(p, 1)
        m = pr != y
        if m.sum() > 0:
            for i in range(len(y)):
                if m[i]:
                    wimg.append(x[i])
                    wpr.append(pr[i].item())
                    wtr.append(y[i].item())
                if len(wimg) == 9:
                    break
        if len(wimg) == 9:
            break

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(wimg[i].squeeze(), cmap='gray')
    plt.title(f"P:{wpr[i]} T:{wtr[i]}")
    plt.axis('off')

plt.show()
```
## Output:
<img width="411" height="421" alt="image" src="https://github.com/user-attachments/assets/db40caf0-4934-4dc3-a6dd-3495c66e123d" />



