from collections import OrderedDict
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes[0],
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(
            planes[0], planes[1],
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.use_range = True
        self.use_xyz = True
        self.use_remission = True
        self.drop_prob = 0.01
        self.bn_d = 0.01
        self.OS = 32
        self.layers = 53
        print("Using DarknetNet" + str(self.layers) + " Backbone")

        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        print("Depth of backbone input = ", self.input_depth)

        self.strides = [2, 2, 2, 2, 2]
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        self.conv1 = nn.Conv2d(
            self.input_depth, 32,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(
            BasicBlock, [32, 64],
            self.blocks[0], stride=self.strides[0], bn_d=self.bn_d,
        )
        self.enc2 = self._make_enc_layer(
            BasicBlock, [64, 128],
            self.blocks[1], stride=self.strides[1], bn_d=self.bn_d,
        )
        self.enc3 = self._make_enc_layer(
            BasicBlock, [128, 256],
            self.blocks[2], stride=self.strides[2], bn_d=self.bn_d,
        )
        self.enc4 = self._make_enc_layer(
            BasicBlock, [256, 512],
            self.blocks[3], stride=self.strides[3], bn_d=self.bn_d,
        )
        self.enc5 = self._make_enc_layer(
            BasicBlock, [512, 1024],
            self.blocks[4], stride=self.strides[4], bn_d=self.bn_d,
        )
        self.dropout = nn.Dropout2d(self.drop_prob)
        self.last_channels = 1024

    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
        layers = []
        layers.append(("conv", nn.Conv2d(
            planes[0], planes[1],
            kernel_size=3, stride=[1, stride], dilation=1, padding=1, bias=False,
        )))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), block(inplanes, planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # first layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.relu1, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth


class Decoder(nn.Module):
    def __init__(self, stub_skips, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = 0.01
        self.bn_d = 0.01

        self.strides = [2, 2, 2, 2, 2]
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))

        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        self.dec5 = self._make_dec_layer(
			BasicBlock, [self.backbone_feature_depth, 512],
            bn_d=self.bn_d, stride=self.strides[0],
		)
        self.dec4 = self._make_dec_layer(
			BasicBlock, [512, 256],
			bn_d=self.bn_d, stride=self.strides[1],
		)
        self.dec3 = self._make_dec_layer(
			BasicBlock, [256, 128],
			bn_d=self.bn_d, stride=self.strides[2],
		)
        self.dec2 = self._make_dec_layer(
			BasicBlock, [128, 64],
			bn_d=self.bn_d, stride=self.strides[3],
		)
        self.dec1 = self._make_dec_layer(
			BasicBlock, [64, 32],
			bn_d=self.bn_d, stride=self.strides[4],
		)

        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]
        self.dropout = nn.Dropout2d(self.drop_prob)
        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
        layers = []

        if stride == 2:
            layers.append(("upconv", nn.ConvTranspose2d(
				planes[0], planes[1],
				kernel_size=[1, 4], stride=[1, 2], padding=[0, 1],
			)))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size=3, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.backbone_OS

        # run layers
        x, skips, os = self.run_layer(x, self.dec5, skips, os)
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        x, skips, os = self.run_layer(x, self.dec1, skips, os)

        x = self.dropout(x)

        return x

    def get_last_depth(self):
        return self.last_channels
