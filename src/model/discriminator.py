import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, 
                 number_channel=3, 
                 image_size=64,
                 ngpu = False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(number_channel, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size) x 32 x 32
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*2) x 16 x 16
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*4) x 8 x 8
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*8) x 4 x 4
            nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output