import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class _EfficientUNetEncoder(nn.Module):
    def __init__(self, skip_connections, model_name='efficientnet-b0'):
        super(_EfficientUNetEncoder, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        self.skip_connections = skip_connections

        self.model._blocks[1].register_forward_hook(self._hook_fn('block1'))
        self.model._blocks[3].register_forward_hook(self._hook_fn('block2'))
        self.model._blocks[5].register_forward_hook(self._hook_fn('block3'))

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.skip_connections.append(output)
        return hook

    def forward(self, x):
        return self.model.extract_features(x)

class _EfficientUNetDecoder(nn.Module):
    def __init__(self, skip_connections):
        super(_EfficientUNetDecoder, self).__init__()
        self.skip_connections = skip_connections
        
        self.upconv1 = nn.ConvTranspose2d(1280, 640, kernel_size=2, stride=2)
        self.conv1 = self._conv_block(640 + 80, 640)

        self.upconv2 = nn.ConvTranspose2d(640, 320, kernel_size=2, stride=2)
        self.conv2 = self._conv_block(320 + 40, 320)

        self.upconv3 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)
        self.conv3 = self._conv_block(160 + 24, 160)

        self.upconv4 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.conv4 = self._conv_block(80, 80)

        self.conv5 = nn.Conv2d(80, 1, kernel_size=1)
        self.upconv5 = nn.ConvTranspose2d(80, 80, kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.upconv1(x)
        if self.skip_connections:
            x = torch.cat((x, self.skip_connections.pop()), dim=1)
        x = self.conv1(x)

        x = self.upconv2(x)
        if self.skip_connections:
            x = torch.cat((x, self.skip_connections.pop()), dim=1)
        x = self.conv2(x)

        x = self.upconv3(x)
        if self.skip_connections:
            x = torch.cat((x, self.skip_connections.pop()), dim=1)
        x = self.conv3(x)

        x = self.upconv4(x)
        x = self.conv4(x)

        x = self.upconv5(x)

        x = self.conv5(x)
        return x

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.skip_connections = []
        self.encoder = _EfficientUNetEncoder(self.skip_connections)
        self.decoder = _EfficientUNetDecoder(self.skip_connections)
        self.classifier = nn.Conv2d(1, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.sigmoid(y)  