import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

def concat(encoder, decoder):
    return torch.cat((decoder, encoder), dim=1)


class EFFUnet_encoder(nn.Module):
    def __init__(self, reqd_blocks):
        super().__init__()
        self.required_blocks = reqd_blocks

    def forward(self, x):
        block_outputs = []
        for blocks in self.required_blocks:
            for block in blocks:
                x = block(x)
                # print("Shape", x.shape)
            block_outputs.append(x)

        return x, block_outputs


class Upconvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)

        # out = F.interpolate(out, (height, width), mode='bilinear', align_corners=True)

class EFFUnet_decoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.upconvolution_1 = Upconvolution(640, 512)
        self.upconvolution_2 = Upconvolution(512, 256)
        self.upconvolution_3 = Upconvolution(256, 128)
        self.upconvolution_4 = Upconvolution(128, 64)
        self.upconvolution_5 = Upconvolution(64, 16)

        # self.encoder_outputs = encoder_outputs
        self.conv_1 = nn.Conv2d(896, 512, kernel_size=(3,3), padding=(1,1), padding_mode='zeros')
        self.conv_2 = nn.Conv2d(416, 256, kernel_size=(3,3), padding=(1,1), padding_mode='zeros')
        self.conv_3 = nn.Conv2d(208, 128, kernel_size=(3,3), padding=(1,1), padding_mode='zeros')
        self.conv_4 = nn.Conv2d(112, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros')

        self.conv_end = nn.Conv2d(16, n_classes, kernel_size=(1,1), stride=(1,1))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, encoder_outputs):
        x = self.upconvolution_1(x)
        # print(x.size())
        encoder_output = F.interpolate(encoder_outputs[7], scale_factor=2, mode='bilinear', align_corners=True)
        # print(encoder_output.size())
        x = concat(x, encoder_output)
        # print(x.size())
        x = self.conv_1(x)
        x = self.relu(x)

        x = self.upconvolution_2(x)
        # print(x.size())
        encoder_output = F.interpolate(encoder_outputs[5], scale_factor=2, mode='bilinear', align_corners=True)
        # print(encoder_output.size())
        x = concat(x, encoder_output)
        # print(x.size())
        x = self.conv_2(x)
        x = self.relu(x)

        x = self.upconvolution_3(x)
        # print(x.size())
        encoder_output = F.interpolate(encoder_outputs[4], scale_factor=2, mode='bilinear', align_corners=True)
        # print(encoder_output.size())
        x = concat(x, encoder_output)
        # print(x.size())
        x = self.conv_3(x)
        x = self.relu(x)

        x = self.upconvolution_4(x)
        # print(x.size())
        encoder_output = F.interpolate(encoder_outputs[3], scale_factor=2, mode='bilinear', align_corners=True)
        # print(encoder_output.size())
        x = concat(x, encoder_output)
        # print(x.size())
        x = self.conv_4(x)
        x = self.relu(x)

        x = self.upconvolution_5(x)
        # print(x.size())
        x = self.conv_end(x)
        # print(x.size())
        x = self.softmax(x)

        return x

class EFFUnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model.to(device=device)

        blocks = []
        blocks.append([list(model.children())[0]])
        blocks.append([list(model.children())[1]])

        MBBlocks_in_block = [4, 7, 7, 10, 10, 13, 4]
        k = 0
        for i in MBBlocks_in_block:
            block = []
            for j in range(0, i, 1):
                block.append(list(model.children())[2][k])
                k += 1
            blocks.append(block)


        self.effunet_encoder = EFFUnet_encoder(blocks)
        for param in self.effunet_encoder.parameters():
            param.requires_grad = False
        self.effunet_decoder = EFFUnet_decoder(n_classes=n_classes)

    def forward(self, x):
        x, block_outputs = self.effunet_encoder(x)
        out = self.effunet_decoder(x, block_outputs)

        return out





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EFFUnet(n_classes = 2)
model.to(device=device)
x = torch.randn(1,3,320,224).to(device=device)
out = model(x)
out.size()