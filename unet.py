
import torch
from torch import nn
import torch.nn.functional as F

# https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv_channels = [16, 32, 64, 128, 256, 512]

        kernel_size = 5
        same_padding = (kernel_size-1)//2

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.l1loss = nn.L1Loss()

        # encoder
        self.conv1 = nn.Conv2d(2, self.conv_channels[0], kernel_size=(5,5), padding=same_padding, stride=(2,2))
        self.bn1 = nn.BatchNorm2d(self.conv_channels[0])
        self.conv2 = nn.Conv2d(self.conv_channels[0], self.conv_channels[1], kernel_size=(5,5), stride=(2,2), padding=same_padding)
        self.bn2 = nn.BatchNorm2d(self.conv_channels[1])
        self.conv3 = nn.Conv2d(self.conv_channels[1], self.conv_channels[2], kernel_size=(5,5), stride=(2,2), padding=same_padding)
        self.bn3 = nn.BatchNorm2d(self.conv_channels[2])
        self.conv4 = nn.Conv2d(self.conv_channels[2], self.conv_channels[3], kernel_size=(5,5), stride=(2,2), padding=same_padding)
        self.bn4 = nn.BatchNorm2d(self.conv_channels[3])
        self.conv5 = nn.Conv2d(self.conv_channels[3], self.conv_channels[4], kernel_size=(5,5), stride=(2,2), padding=same_padding)
        self.bn5 = nn.BatchNorm2d(self.conv_channels[4])
        self.conv6 = nn.Conv2d(self.conv_channels[4], self.conv_channels[5], kernel_size=(5,5), stride=(2,2), padding=same_padding)

        # decoder
        self.convt1 = nn.ConvTranspose2d(self.conv_channels[5], self.conv_channels[4], kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn6 = nn.BatchNorm2d(self.conv_channels[4])
        self.drop1 = nn.Dropout2d(0.5)
        self.convt2 = nn.ConvTranspose2d(2*self.conv_channels[4], self.conv_channels[3], kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn7 = nn.BatchNorm2d(self.conv_channels[3])
        self.drop2 = nn.Dropout2d(0.5)
        self.convt3 = nn.ConvTranspose2d(2*self.conv_channels[3], self.conv_channels[2], kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn8 = nn.BatchNorm2d(self.conv_channels[2])
        self.drop3 = nn.Dropout2d(0.5)
        self.convt4 = nn.ConvTranspose2d(2*self.conv_channels[2], self.conv_channels[1], kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn9 = nn.BatchNorm2d(self.conv_channels[1])
        self.convt5 = nn.ConvTranspose2d(2*self.conv_channels[1], self.conv_channels[0], kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn10 = nn.BatchNorm2d(self.conv_channels[0])
        self.convt6 = nn.ConvTranspose2d(2*self.conv_channels[0], 2, kernel_size=(5,5), stride=(2,2), padding=same_padding, output_padding=1)
        self.bn11 = nn.BatchNorm2d(2)

    def forward(self, x):
        # encoder
        c1 = self.conv1(x)
        l1 = self.leaky_relu(self.bn1(c1))
        c2 = self.conv2(l1)
        l2 = self.leaky_relu(self.bn2(c2))
        c3 = self.conv3(l2)
        l3 = self.leaky_relu(self.bn3(c3))
        c4 = self.conv4(l3)
        l4 = self.leaky_relu(self.bn4(c4))
        c5 = self.conv5(l4)
        l5 = self.leaky_relu(self.bn5(c5))
        c6 = self.conv6(l5)

        # decoder
        ct1 = self.convt1(c6)
        lt1 = self.drop1(self.bn6(self.relu(ct1)))
        cat1 = torch.cat((lt1,c5), dim=1)
        ct2 = self.convt2(cat1)
        lt2 = self.drop2(self.bn7(self.relu(ct2)))
        cat2 = torch.cat((lt2,c4), dim=1)
        ct3 = self.convt3(cat2)
        lt3 = self.drop3(self.bn8(self.relu(ct3)))
        cat3 = torch.cat((lt3,c3), dim=1)
        ct4 = self.convt4(cat3)
        lt4 = self.bn9(self.relu(ct4))
        cat4 = torch.cat((lt4,c2), dim=1)
        ct5 = self.convt5(cat4)
        lt5 = self.bn10(self.relu(ct5))
        cat5 = torch.cat((lt5,c1), dim=1)
        ct6 = self.convt6(cat5)
        mask = torch.sigmoid(ct6)

        return torch.mul(mask, x)

    def l1loss(y_pred, y):
        return F.l1_loss(y_pred, y)