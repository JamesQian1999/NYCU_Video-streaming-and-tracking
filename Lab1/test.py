import torch
from nn import vggnet
import pandas as pd
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas
from glob import glob
import re
import torch.nn as nn

class vggnet(nn.Module):
    def __init__(self):
        super(vggnet, self).__init__()

        self.c1 = nn.Sequential(
                nn.Conv2d(3, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(),
                )

        self.c2 = nn.Sequential(
                nn.Conv2d(4, 8, 3, 1, 1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                )

        self.c3 = nn.Sequential(
                nn.Conv2d(8, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                )

        self.c4 = nn.Sequential(
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                )

        self.fc = nn.Sequential(
                nn.Conv2d(1,3,(1,896),896),
                nn.LeakyReLU(),
                nn.Conv2d(3,10,(1,7)),
                nn.ReLU(),
                nn.Softmax(dim = 1)
        )

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(self.mp(h1))
        h3 = self.c3(self.mp(h2))
        h4 = self.c4(self.mp(h3))
        out = self.fc(self.mp(h4).reshape((-1,1,1,6272)))
        
        return out.view(-1,10)


class loader(data.Dataset):
    def __init__(self, mode = 'train', transform = None):
        
        self.mode = mode
        print(f'Mode "{self.mode}"')
        if mode == 'train':
            self.data = pandas.read_csv('train.csv')
        elif mode == 'val':
            self.data = pandas.read_csv('val.csv')
        else:
            self.data = glob('test/*')

        self.transform = transform


    def __len__(self):
        if self.mode == 'test':
            return len(self.data)
        else:
            return self.data.shape[0]

    def __getitem__(self, index):

        preprocess = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0.05)])

        try:
            img = Image.open( self.mode  + '/'+ self.data['names'][index])
            label = self.data['label'][index]
        except:
            img = Image.open( self.data[index])
            label = self.data[index]
            label = re.sub('.*/', '', label)

        if self.transform != None:
            img = preprocess(img)

        img = transforms.ToTensor()(img)

        return img, label



def test(model, data, device = torch.device("cpu"), criterion = torch.nn.CrossEntropyLoss(), t = False):
    model.eval()
    correct_test = 0
    test_output = []
    if t:
        for idx, (data, name) in enumerate(data):
            torch.cuda.empty_cache()
            
            data = data.to(device, dtype = torch.float)
            pred = model(data)

            pred = pred.argmax(dim=1,keepdim=True)
            pred = pred.squeeze(1)

            test_output.extend(zip(name, pred.tolist()))
            
        df = pd.DataFrame(test_output, columns = ['names','label'])
        df['file'] = df['names'].str.replace('.jpg','').astype(int)
        df = df.sort_values(by=['file'])
        df.to_csv('HW1_311551096.csv', index = False, columns = ['names','label'])
        print(df)

    else:
        total_loss = 0
        for idx, (data, target) in enumerate(data):
            torch.cuda.empty_cache()
            
            data, target = data.to(device, dtype = torch.float), target.to(device, dtype = torch.long)
            pred = model(data)

            loss = criterion(pred, target)
            total_loss += loss

            pred = pred.argmax(dim=1,keepdim=True)
            pred = pred.squeeze(1)

            correct_test += len(pred[pred == target])

        return correct_test, total_loss/len(data)


if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device)

    path = 'HW1_311551096.pt'
    print('Loaded',path)

    vgg = vggnet()
    vgg.load_state_dict(torch.load(path))
    vgg = vgg.to(device)

    num = sum(p.numel() for p in vgg.parameters() if p.requires_grad)
    print('Number of Model parameters:',num)

    data = loader(mode = 'test')
    dataset  = torch.utils.data.DataLoader(data, batch_size = 64)

    test(vgg, dataset, device, t = True)