import os
import torch
from test import test, loader, vggnet
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



def train(model, device, training, optimizer, criterion, hparams, val):
    model = model.to(device)
    train_acc = []
    test_acc = []
    loss_list = []
    best = 50
    for epoch in range(hparams['epochs']):
        model.train()
        correct_train = 0
        total_loss = 0
        with torch.set_grad_enabled(True):
            for idx, (data, target) in enumerate(training):
                torch.cuda.empty_cache()

                data, target = data.to(device, dtype = torch.float), target.to(device, dtype = torch.long)
                pred = model(data)

                loss = criterion(pred, target)
                total_loss += loss

                pred = pred.argmax(dim=1,keepdim=True)
                pred = pred.squeeze(1)

                correct_train += len(pred[pred == target])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        acc = (100*(correct_train/len(training.dataset)))
        total_loss = (total_loss/len(training))

        train_acc.append(acc)
        loss_list.append(total_loss)

        val_acc, val_loss = test(model, val, device, criterion)
        test_acc_ = (100*val_acc/len(val.dataset))
        test_acc.append(test_acc_)
        print(f"[{epoch:2d}/{hparams['epochs']}]  Training accuracy: {acc : .3f}%  loss: {total_loss: .5f}  Val accuracy: {test_acc_ : .3f}%  Val Loss: {val_loss : .3f}")

        if( test_acc_ >= best ):
            best = test_acc_
            torch.save(model.state_dict(),"log/best_vgg.pt")
            print('Model saved.')

        with open('log/train_loss.txt', 'a') as file:
            file.write(str(total_loss.item())+',')

        with open('log/train_acc.txt', 'a') as file:
            file.write(str(acc)+',')

        with open('log/val_loss.txt', 'a') as file:
            file.write(str(val_loss.item())+',')

        with open('log/val_acc.txt', 'a') as file:
            file.write(str(test_acc_)+',')

    torch.save(model.state_dict(), "log/vgg.pt")



if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device)

    hparams = {
        'lr'        : 0.0005,
        'epochs'    : 1,
        'batch_size': 128
    }

    vgg = vggnet()
    vgg = vgg.to(device)

    num = sum(p.numel() for p in vgg.parameters() if p.requires_grad)
    print('Number of Model parameters:',num)

    optimizer = torch.optim.Adam(vgg.parameters(), lr=hparams['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    train_set = loader(transform = True)
    training  = torch.utils.data.DataLoader(train_set, batch_size = hparams['batch_size'], shuffle = True)
    val_set = loader(mode = 'val')
    val  = torch.utils.data.DataLoader(val_set, batch_size = hparams['batch_size'])

    os.makedirs('log/', exist_ok=True)

    if os.path.exists('log/train_loss.txt'):
        os.remove('log/train_loss.txt')

    if os.path.exists('log/val_loss.txt'):
        os.remove('log/val_loss.txt')

    if os.path.exists('log/train_acc.txt'):
        os.remove('log/train_acc.txt')

    if os.path.exists('log/val_acc.txt'):
        os.remove('log/val_acc.txt')

    train(vgg, device, training, optimizer, criterion, hparams, val)

    plot('log/')