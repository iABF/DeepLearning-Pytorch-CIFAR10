import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2)
testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 256)
        self.fc3 = nn.Linear(256, 144)
        self.fc4 = nn.Linear(144, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def train():
    file = open('fcn_round.txt', 'r+')
    r = int(file.read())
    net = Net()
    net.load_state_dict(torch.load('./cifar_net_fcn.pth'))  # load trained result
    criterion = nn.CrossEntropyLoss()  # Loss Function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Optimizer
    lossFile = open('fcn_loss.txt', 'a')
    for epoch in range(1):
        r += 1
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                lossFile.write('[%d, %5d] loss: %.3f' % (epoch + r, i + 1, running_loss / 100))
                # print('[%d, %5d] loss: %.3f' % (epoch + r, i + 1, running_loss / 100))
                lossFile.write('\n')
                running_loss = 0.0
    lossFile.close()
    print('[%d round]: Finished Training' % r)
    file.seek(0)
    file.write(str(r))
    file.close()
    torch.save(net.state_dict(), './cifar_net_fcn.pth')  # save trained result


def test():
    file = open('fcn_round.txt', 'r')
    r = int(file.read())
    file.close()
    net = Net()
    net.load_state_dict(torch.load('./cifar_net_fcn.pth'))  # load trained result
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    file = open('fcn/accuracy.txt', 'a')
    file.write('%d %%' % (100 * correct / total))
    file.write('\n')
    file.close()
    # print('[%d round]: Accuracy of the network on the 10000 test images: %d %%' % (r, 100 * correct / total))
    for i in range(10):
        fileName = 'fcn/accuracy_' + str(i) + '.txt'
        file = open(fileName, 'a')
        file.write('%d %%' % (100 * class_correct[i] / class_total[i]))
        file.write('\n')
        file.close()
        # print('[%d round]: Accuracy of %5s : %2d %%' % (r, classes[i], 100 * class_correct[i] / class_total[i]))
    print('[%d round]: Finished Testing' % r)


if __name__ == '__main__':
    for _ in range(20):
        train()
        test()
