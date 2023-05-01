
import os
import torch.nn as nn
import torch
from .base_model import BaseModel
from . import networks
import kornia



class Dense_Block(nn.Module):
    def __init__(self, channels):
        super(Dense_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        fea_1 = self.relu(self.conv1(x))
        res_1 = fea_1 + x
        fea_2 = self.conv2(res_1)
        res_2 = fea_2 + res_1
        return res_2

class UPD_model(nn.Module):
    def __init__(self, args):
        super(UPD_model, self).__init__()
        self.args = args
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.pan_channel, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=args.pan_channel, out_channels=args.pan_channel, kernel_size=3, stride=2, padding=1),
        )
        self.conv_3 = nn.Sequential(
            self.make_layer(Dense_Block, 8, 64),
        )
        self.conv_4 =  nn.Sequential(
            nn.Conv2d(in_channels=args.mul_channel, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
              )
        self.conv_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
        )

        self.conv_7 = nn.Sequential(
            self.make_layer(Dense_Block, 8, 64),
            )
        self.conv_10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
              )

        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv_12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=args.mul_channel, kernel_size=3, padding=1),
        )
        self.conv_13 = nn.Sequential(
            self.make_layer(Dense_Block, 8, 64),
        )

        self.bicubic = networks.bicubic()

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)


    def forward(self, x, y):
        x_up = self.bicubic(x, scale=self.args.scale)#torch.nn.functional.interpolate(x, scale_factor=cfg.scale, mode='bicubic', align_corners=True)#
        y1_lp = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(y)
        y1_hp = y - y1_lp
        pan_2_down = self.bicubic(y, scale=1 / 2)
        y2_lp = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan_2_down)
        y2_hp = pan_2_down - y2_lp
        pan_3_down = self.bicubic(pan_2_down, scale=1 / 2)
        y3_lp = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan_3_down)
        y3_hp = pan_3_down - y3_lp


        pan_1 = self.conv_1(y)
        y_hp_C = self.conv_1(y1_hp)
        pan_2 = self.conv_2(y)
        pan_2_d = self.conv_1(pan_2)

        pan_3 = self.conv_2(pan_2)
        pan_3_d = self.conv_1(pan_3)

        y2_hp_C = self.conv_1(y2_hp)
        y3_hp_C = self.conv_1(y3_hp)

        ms_1 = self.conv_4(x)
        ms_2 = self.conv_5(ms_1)
        ms_3 = self.conv_5(ms_2)


        fus_1_a = ms_3 + y_hp_C
        fus_2_a= ms_2 + y2_hp_C
        fus_3_a = ms_1 + y3_hp_C
        fus_1 = torch.cat([pan_1,fus_1_a], -3)#64
        fus_2 = torch.cat([pan_2_d, fus_2_a], -3)#64
        fus_3 = torch.cat([pan_3_d, fus_3_a], -3)#64


        fus_3_3 = self.conv_7(fus_3)
        fus_3_3 = fus_3_3 + fus_3
        fus_2_2 = self.conv_7(fus_2)
        fus_2_2 = fus_2_2 + fus_2
        fus_1_1 = self.conv_7(fus_1)
        fus_1_1 = fus_1_1 + fus_1

        l_fus_1_down = self.conv_down(fus_1_1)
        l_fus_2 = l_fus_1_down + fus_2_2
        l_fus_2 = self.conv_3(l_fus_2)
        l_fus_2_down = self.conv_down(l_fus_2)
        l_fus_3 = l_fus_2_down + fus_3_3

        S_fus_3 = self.conv_3(l_fus_3)
        S_fus_3_up = self.conv_10(S_fus_3)
        S_fus_2 = S_fus_3_up + fus_2_2
        S_fus_2 = self.conv_3(S_fus_2)
        S_fus_2_up = self.conv_10(S_fus_2)
        S_fus_1 = S_fus_2_up + fus_1_1
        out_2 = self.conv_13(S_fus_1)

        out_3 = self.conv_12(out_2)

        out = out_3 + x_up
        return out
        
class UPD7Model(BaseModel):

    def initialize(self, args):
        self.args = args
        BaseModel.initialize(self, args)
        self.save_dir = os.path.join(args.checkpoints_dir, args.model) # 定义checkpoints路径
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Create file path: ", self.save_dir)
        
        if args.isUnlabel:
            self.save_dir = os.path.join(self.save_dir, 'unsupervised')
        else:
            self.save_dir = os.path.join(self.save_dir, 'supervised')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Create file path: ", self.save_dir)
        
        self.loss_names = ['G']

        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        self.netG = networks.init_net(UPD_model(args)).cuda()
        
        if self.isTrain:

            self.criterionL1 = networks.taLoss(device=self.device, args=args)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input_dict):
            self.real_A_1 = input_dict['A_1'].to(self.device)  # mul
            self.real_A_2 = input_dict['A_2'].to(self.device)  # pan
            self.real_B = input_dict['B'].to(self.device) # fus

    def forward(self):
        self.fake_B = self.netG(self.real_A_1, self.real_A_2)

    def backward_G(self):
        self.loss_G = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

