import matplotlib.pyplot as plt


def rnn_loss():
    file = open('rnn_loss.txt', 'r')
    lines = file.read().split('\n')
    file.close()
    data = []
    for line in lines:
        data.append(float(line[-5:]))
    plt.plot(data)
    plt.xlabel('Train Number (400 pictures each)')
    plt.ylabel('Loss Value')
    plt.title('RNN Loss')
    plt.show()


def cnn_loss():
    file = open('cnn_loss.txt', 'r')
    lines = file.read().split('\n')
    file.close()
    data = []
    for line in lines:
        data.append(float(line[-5:]))
    plt.plot(data)
    plt.xlabel('Train Number (400 pictures each)')
    plt.ylabel('Loss Value')
    plt.title('CNN Loss')
    plt.show()


def fcn_loss():
    file = open('fcn_loss.txt', 'r')
    lines = file.read().split('\n')
    file.close()
    data = []
    for line in lines:
        data.append(float(line[-5:]))
    plt.plot(data)
    plt.xlabel('Train Number (400 pictures each)')
    plt.ylabel('Loss Value')
    plt.title('FCN Loss')
    plt.show()


if __name__ == '__main__':
    fcn_loss()
