import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
class Polluted_Data(torchvision.datasets.FashionMNIST):
    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        if label % 600 == 0:
            return data, (label + 1) % 10
        return data, label

class FC(torch.nn.Module):
    def __init__(self, activation, num_hidden_nodes, outputs):
        super().__init__()
        num_hidden_nodes.insert(0, (28*28))
        num_hidden_nodes.append(outputs)
        self.layers = torch.nn.ModuleList()
        for i in range(len(num_hidden_nodes) - 1):
            self.layers.append(torch.nn.Linear(num_hidden_nodes[i], num_hidden_nodes[i+1]))
        self.activation = activation
    def forward(self, x):
        x= x.view(-1, 28*28)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def train(model, loader, optimizer, device, criterion):
    total_loss = 0
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
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0))])
    #Download datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)

    #Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
        shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
        shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FC(torch.relu, [1024], 10)
    model = model.to(device)
    model_2 = FC(torch.relu, [1024, 1024], 10)
    model_2 = model_2.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.001, momentum=0)
    x = np.linspace(0, 500, 501)
    accuracy_1_test = [test(model, testloader, device, 10)]
    accuracy_1_train = [test(model, trainloader, device, 10)]
    accuracy_2_test = [test(model_2, testloader, device, 10)]
    accuracy_2_train = [test(model_2, trainloader, device, 10)]
    for epoch in range(1, 501):
        print(f'Epoch {epoch}')
        train(model, trainloader, optimizer, device, torch.nn.CrossEntropyLoss())
        train(model_2, trainloader, optimizer_2, device, torch.nn.CrossEntropyLoss())
        accuracy_1_train.append(test(model, trainloader, device, 10))
        accuracy_2_train.append(test(model_2, trainloader, device, 10))
        accuracy_1_test.append(test(model, testloader, device, 10))
        accuracy_2_test.append(test(model_2, testloader, device, 10))

    plt.figure()
    plt.axis([0,500,0,1])
    plt.plot(x, accuracy_1_test, 'r', label='1 hidden layer testing')
    plt.plot(x, accuracy_1_train, 'r--', label='1 hidden layer training')
    plt.plot(x, accuracy_2_test, 'b', label='2 hidden layers testing')
    plt.plot(x, accuracy_2_train, 'b--', label='2 hidden layers training')
    plt.legend()
    plt.show()
    torch.save(model, 'FC-1')
    torch.save(model_2, 'FC-2')

def experiment_2():
    #Preprocessing transform
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])
    #Download datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)
    iteration = 1
    activations = [torch.relu, torch.sigmoid]
    learning_rates = [1, 0.1, 0.01, 0.001]
    batch_size = [1, 10, 1000]
    accuracys = []
    for activation in activations:
        for rate in learning_rates:
            for size in batch_size:
                model = FC(activation, [1024, 1024], 10)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=size,shuffle=True)
                testloader = torch.utils.data.DataLoader(testset, batch_size=size,shuffle=False)
                for epoch in range(1, 51):
                    train(model, trainloader, optimizer, device, torch.nn.CrossEntropyLoss())
                    print(f"Iteration: {iteration}, Epoch: {epoch}")
                accuracy = test(model, testloader, device, 10)
                string = "Accuracy for activation: " + str(activation) + " learning rate: " + str(rate) + " batch size: " + str(size) + " accuracy: " + str(accuracy)
                accuracys.append(string)
                iteration += 1
                print(string)
    for accuracy in accuracys:
        print(accuracy)

def experiment_3():
    #Preprocessing transform
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])
    #Download datasets
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)

    polluted_data = Polluted_Data('./data',
        download=True,
        train=True,
        transform=transform)
    #Dataloaders
    trainloader = torch.utils.data.DataLoader(polluted_data, batch_size=1,
        shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
        shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FC(torch.relu, [1024, 1024], 10)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    x = np.linspace(0, 100, 101)
    accuracy_1_test = [test(model, testloader, device, 10)]
    accuracy_1_train = [test(model, trainloader, device, 10)]
    for epoch in range(1, 101):
        print(f'Epoch {epoch}')
        train(model, trainloader, optimizer, device, torch.nn.CrossEntropyLoss())
        accuracy_1_train.append(test(model, trainloader, device, 10))
        accuracy_1_test.append(test(model, testloader, device, 10))

    plt.figure()
    plt.axis([0,100,0,1])
    plt.plot(x, accuracy_1_test, 'r', label='testing')
    plt.plot(x, accuracy_1_train, 'r--', label='training')
    plt.legend()
    plt.show()
    torch.save(model, '2-1')


if __name__ == '__main__':
    experiment_1()