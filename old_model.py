import torch
import torch.nn as nn
import torch.nn.functional as F

class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()
        self.enc1_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.enc1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.enc2_1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.enc2_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.enc3_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.enc3_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.enc3_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.enc4_1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.enc4_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.enc4_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())


        self.enc5_1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.enc5_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.enc5_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.dec5_3= nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.dec5_2 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.dec5_1 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.dec4_3 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.dec4_2 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.dec4_1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.dec3_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.dec3_2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.dec3_1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.dec2_2 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.dec2_1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.dec1_2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.dec1_1 = nn.Sequential(nn.ConvTranspose2d(64, 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU())

    def forward(self, image):

        size0=image.size()

        x=self.enc1_1(image)
        x=self.enc1_2(x)

        x,i0=F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        size1=x.size()

        x=self.enc2_1(x)
        x=self.enc2_2(x)

        x,i1=F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        size2=x.size()

        x = self.enc3_1(x)
        x = self.enc3_2(x)
        x = self.enc3_3(x)

        x, i2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        size3 = x.size()


        x = self.enc4_1(x)
        x = self.enc4_2(x)
        x = self.enc4_3(x)

        x, i3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        size4 = x.size()


        x = self.enc5_1(x)
        x = self.enc5_2(x)
        x = self.enc5_3(x)

        x, i4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x=F.max_unpool2d(x,i4,kernel_size=2, stride=2,output_size=size4)

        x = self.dec5_3(x)
        x = self.dec5_2(x)
        x = self.dec5_1(x)

        x = F.max_unpool2d(x, i3, kernel_size=2, stride=2, output_size=size3)

        x = self.dec4_3(x)
        x = self.dec4_2(x)
        x = self.dec4_1(x)

        x = F.max_unpool2d(x, i2, kernel_size=2, stride=2, output_size=size2)

        x = self.dec3_3(x)
        x = self.dec3_2(x)
        x = self.dec3_1(x)

        x = F.max_unpool2d(x, i1, kernel_size=2, stride=2, output_size=size1)

        x = self.dec2_2(x)
        x = self.dec2_1(x)

        x = F.max_unpool2d(x, i0, kernel_size=2, stride=2, output_size=size0)

        x = self.dec1_2(x)
        x = self.dec1_1(x)

        softmax=F.softmax(x,dim=1)

        return x,softmax