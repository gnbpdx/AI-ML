import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys

class LeNet(torch.nn.Module):
    def __init__(self, activation, dropout=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(in_features=400, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = activation
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.activation(self.conv2(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 400)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.activation(self.fc3(x))
        return x

class LeNet3x3(torch.nn.Module):
    def __init__(self, activation, dropout=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(in_features=576, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = activation
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.activation(self.conv2(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 576)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        #Cross Entropy Loss uses Log-softmax
        x = self.fc3(x)
        return x

class LeNet5_3x3(torch.nn.Module):
    def __init__(self, activation, dropout=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=(1,1))
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1, padding=(1,1))
        self.conv3 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=(1,1))
        self.conv4 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=(1,1))
        self.conv5 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=(1,1))
        self.fc1 = torch.nn.Linear(in_features=36450, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = activation
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        x = self.activation(self.conv2(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        x = self.activation(self.conv3(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        x = self.activation(self.conv4(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        x = self.activation(self.conv5(x))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=1)
        x = x.view(-1, 36450)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.activation(self.fc3(x))
        return x





def train(model, loader, optimizer, device, criterion):
    total_loss = 0
    model.train()
    for _, batch in enumerate(loader, 0):

        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs) 
        loss = criterion(output, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)

def test(model, loader, device, num_classes):
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            output = output.to('cpu')
            labels = labels.to('cpu')
            for i in range(len(inputs)):
                prediction = int(torch.argmax(output[i]))
                true_label = int(labels[i])
                if prediction == true_label:
                    correct += 1
                total += 1
    return correct / total

def experiment_1():
    #Preprocessing transform
    transform_train = torchvision.transforms.Compose([
      torchvision.transforms.RandomAffine(10, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
      torchvision.transforms.ToTensor()]  
    )
    transform_test = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])
    #Download datasets
    trainset = torchvision.datasets.CIFAR10('./data',
        download=True,
        train=True,
        transform=transform_train)
    testset = torchvision.datasets.CIFAR10('./data',
        download=True,
        train=False,
        transform=transform_test)

    #Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
        shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
        shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate = float(sys.argv[1])
    momentum = float(sys.argv[2])
    regularization = float(sys.argv[3])
    dropout = float(sys.argv[4])
    activation_function = None
    loss_function = None
    if sys.argv[5] == 'sigmoid':
        activation_function = torch.sigmoid
    if sys.argv[5] == 'tanh':
        activation_function = torch.tanh
    if sys.argv[5] == 'relu':
        activation_function = torch.relu
    if sys.argv[6] == 'CE':
        loss_function = torch.nn.CrossEntropyLoss
    if sys.argv[6] == 'MSE':
        loss_function = torch.nn.MSELoss
    
    
    filename = "Lenet 5" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6] + '.csv'
    model = LeNet5_3x3(activation_function, dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=regularization)
    accuracy_1_test = [test(model, testloader, device, 10)]
    accuracy_1_train = [test(model, trainloader, device, 10)]
    for epoch in range(1,5001):
        print(f'Epoch {epoch} lr {learning_rate}')
        print(str(activation_function) + str(loss_function))
        train(model, trainloader, optimizer, device, loss_function())
        train_error = test(model, trainloader, device, 10)
        accuracy_1_train.append(train_error)
        print("Train error: ", train_error)
        test_error = test(model, testloader, device, 10)
        accuracy_1_test.append(test_error)
        print("Test error ", test_error)
        print_to_csv(accuracy_1_train, accuracy_1_test, filename)

def print_to_csv(train_error, test_error, file):
    with open(file, "w") as f:
        for i in range(len(train_error)):
            f.write(str(i) + ', ' + str(train_error[i]) + ", " + str(test_error[i]) + '\n')
            
        
            
if __name__ == '__main__':
    experiment_1()