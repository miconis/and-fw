import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from multiprocessing import freeze_support
import numpy as np


# DOWNLOAD AND LOAD IN A DATASET USING DATALOADER UTILITY
BATCH_SIZE = 32  # used when creating the loader, so that when iterating over it, collects BATCH_SIZE images and labels at a time

# transformations
transform = transforms.Compose([transforms.ToTensor()])

# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# visualize
print("Length of train set:", len(trainset))
print(trainset[10])

# define the model: conv layer followed by two fully-connected layers
class MyModel(nn.Module):
    def __init__(self): # definition of the architecture
        super(MyModel, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x): # what is done to the input data (use layers defined in the init)
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)
        #x = x.view(32, -1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out


if __name__ == '__main__':
    freeze_support()

    a = np.array([[1, 2], [3, 4]])
    b = np.ones((2, 2))

    ta = torch.tensor(a, dtype=float).to("cuda:0" if torch.cuda.is_available() else "cpu")
    tb = torch.ones(2, 2, dtype=float).to("cuda:0" if torch.cuda.is_available() else "cpu")

    print(ta)
    print(ta @ tb)


    # MODEL TRAINING
    # instantiate the model, the loss function criterion and the optimizer, which will adjust the parameters of our model in order to minimize the loss output by criterion

    learning_rate = 0.001
    num_epochs = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # training loop
    for epoch in range(num_epochs):  # for each epoch repeat the same thing
        train_running_loss = 0.0
        train_acc = 0.0

        # training step
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # forward + backprop + loss
            logits = model(images)  # run the data through the model (forward pass)
            loss = criterion(logits, labels)  # compute the loss
            optimizer.zero_grad()  # zero out the gradient from the previous round of training
            loss.backward()  # backpropagate the new round of gradients

            # update model params
            optimizer.step()  # adjust the model parameters

            train_running_loss += loss.detach().item()
            train_acc += (torch.argmax(logits, 1).flatten() == labels).type(torch.float).mean().item()
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' % (epoch, train_running_loss / i, train_acc / i))


    # MODEL TESTING
    # run the forward pass in order to run it on the test set
    test_acc = 0.0
    for i, (images, labels) in enumerate(testloader, 0):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item()
        preds = torch.argmax(outputs, 1).flatten().cpu().numpy()

    print('Test Accuracy: %.2f' % (test_acc / i))


