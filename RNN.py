import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2)
testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(32 * 3, 128, num_layers=3, batch_first=True)
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        r_out, (h_n, c_n) = self.LSTM(x)
        return self.linear(r_out[:, -1, :])


def train(testRound):
    file = open('rnn_round.txt', 'r+')
    r = int(file.read())
    net = Net()
    net.load_state_dict(torch.load('./cifar_net_lstm.pth'))  # load trained result
    criterion = nn.CrossEntropyLoss()  # Loss Function
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # Optimizer
    for epoch in range(testRound):
        r += 1
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs = inputs.view(-1, 32, 32 * 3)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + r, i + 1, running_loss / 100))
                running_loss = 0.0
    print('[%d round]: Finished Training' % r)
    file.seek(0)
    file.write(str(r))
    file.close()
    torch.save(net.state_dict(), './cifar_net_lstm.pth')  # save trained result


def test():
    file = open('rnn_round.txt', 'r')
    r = int(file.read())
    file.close()
    net = Net()
    net.load_state_dict(torch.load('./cifar_net_lstm.pth'))  # load trained result
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images = Variable(images)
            images = images.view(-1, 32, 32 * 3)
            outputs = net(images)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    for _ in range(10):
        train(1)
        test()
