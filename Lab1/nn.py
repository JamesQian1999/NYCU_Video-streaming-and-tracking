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