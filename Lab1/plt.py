
import numpy as np
import matplotlib.pyplot as plt

############## plot ##############

def plot(path):

        with open(path + 'train_loss.txt','r') as file:
            train_loss = file.read()
        train_loss = train_loss.split(",")[:-1]
        train_loss = np.array([float(i) for i in train_loss])

        with open(path + 'val_loss.txt','r') as file:
            val_loss = file.read()
        val_loss = val_loss.split(",")[:-1]
        val_loss = np.array([float(i) for i in val_loss])


        ax1 = plt.subplot()
        l1, = ax1.plot(np.array([i for i in range(len(train_loss))]), train_loss, color='m')
        l2, = ax1.plot(np.array([i for i in range(len(val_loss))]), val_loss, color='royalblue')

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        plt.legend([l1, l2],['train','val'])
        plt.savefig(path + 'loss.png', dpi=500, bbox_inches="tight")
        # plt.show()
        plt.clf()


        with open(path + 'train_acc.txt','r') as file:
            train_acc = file.read()
        train_acc = train_acc.split(",")[:-1]
        train_acc = np.array([float(i) for i in train_acc])

        with open(path + 'val_acc.txt','r') as file:
            val_acc = file.read()
        val_acc = val_acc.split(",")[:-1]
        val_acc = np.array([float(i) for i in val_acc])


        ax1 = plt.subplot()
        l1, = ax1.plot(np.array([i for i in range(len(train_acc))]), train_acc, color='m')
        l2, = ax1.plot(np.array([i for i in range(len(val_acc))]), val_acc, color='royalblue')

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('acc')
        plt.legend([l1, l2],['train','val'])
        plt.savefig(path + 'acc.png', dpi=500, bbox_inches="tight")
        # plt.show()



  

if __name__ == '__main__':
    plot('./log5/')