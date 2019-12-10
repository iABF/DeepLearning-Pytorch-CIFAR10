import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=60000, shuffle=True, num_workers=2)
testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class KNN(object):
    def __init__(self, im, la, k):
        self.trainImages = im
        self.labels = la
        self.k = k

    def decide(self, testImages):
        dists = self.compute_distance(testImages)
        testNum = testImages.shape[0]
        label = np.zeros(testNum)
        for i in range(testNum):
            minLabel = self.labels[dists[i, :].argsort()[: self.k]]
            label[i] = np.argmax(np.bincount(minLabel))  # i-th testImage's label
        return label

    def compute_distance(self, testImages):
        dists = np.multiply(np.dot(testImages, self.trainImages.T), -2)
        add1 = np.sum(np.square(testImages), keepdims=True)
        dists = np.add(dists, add1)
        add2 = np.sum(np.square(self.trainImages), axis=1)
        dists = np.sqrt(np.add(dists, add2))
        return dists

    def set_k(self, k):
        self.k = k


def test(machine):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            predicted = machine.decide(images.numpy().reshape(images.shape[0], 32 * 32 * 3))
            predicted = torch.from_numpy(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('Finished Testing')


if __name__ == '__main__':
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    Machine = None
    for trainData in trainLoader:
        trainImages, trainLabels = trainData
        Machine = KNN(
            trainImages.numpy().reshape(trainImages.shape[0], 32 * 32 * 3),
            trainLabels.numpy(),
            10)
        break
    for K in range(1, 21):
        print('Round %d start' % K)
        localtime = time.asctime(time.localtime(time.time()))
        print(localtime)
        Machine.set_k(K)
        test(Machine)
    print('Finished')
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
