class Generator(nn.Module):
    def __init__(self,number_channel = 3, image_size = 64, ngpu = False):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( number_channel, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
            # state size. (image_size*8) x 4 x 4
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            # state size. (image_size*4) x 8 x 8
            nn.ConvTranspose2d( image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            # state size. (image_size*2) x 16 x 16
            nn.ConvTranspose2d( image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            # state size. (image_size) x 32 x 32
            nn.ConvTranspose2d( image_size, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1,1).squeeze(1)
