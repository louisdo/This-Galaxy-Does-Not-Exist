import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, 
                 number_channel=3, 
                 image_size=64,
                 num_classes = 10,
                 ngpu = False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(number_channel, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.real_fake_classifier = nn.Sequential(
            nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.label_classifier = nn.Sequential(
            nn.Conv2d(image_size * 8, num_classes, 4, 1, 0, bias=False)
        )

    def forward(self, inp, classify = False):
        if inp.is_cuda and self.ngpu > 1:
            feature = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            feature = self.main(inp)

        if classify:
            return self.real_fake_classifier(feature), self.label_classifier(feature)
        return self.real_fake_classifier(feature)