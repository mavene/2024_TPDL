import torch
import torch.nn as nn

import math

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, pool_size: int = 2):
        super(ConvolutionBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(pool_size, pool_size)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.pooling(out)
        return out
    
class FCBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: int=0.2):
        super(FCBlock, self).__init__()
        
        self.fc = nn.Linear(in_features, out_features, dtype=torch.float32)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        out = self.dropout(out)
        return out
    
class ClassifierBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: tuple[int], out_features: int, dropout_rate: int = 0.2):
        super(ClassifierBlock, self).__init__()

        self.inputs = FCBlock(in_features, hidden_features[0], dropout_rate)
        self.layers = nn.Sequential(*[FCBlock(hidden_features[i], hidden_features[i+1], dropout_rate) for i in range(len(hidden_features)-1)])
        self.linear = nn.Linear(hidden_features[-1], out_features, dtype=torch.float32)
    
    def forward(self, x):
        out = self.inputs(x)
        out = self.layers(out)
        out = self.linear(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, channels: tuple[int], img_size: tuple[int], hidden_size: tuple[int], output_size: int, dropout_rate: int = 0.2):
        super(ConvNet, self).__init__()
        
        # Hyperparameters for Convolutional Layers
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.pool_size = 2

        # Input size of Image
        self.input_height, self.input_width = img_size

        # Calculate Input Size of Classifier
        for i in range(len(channels) - 1):
            self.input_height = math.floor((self.input_height + 2 * self.padding - self.kernel_size)/ self.stride + 1) // self.pool_size
            self.input_width =  math.floor((self.input_width  + 2 * self.padding - self.kernel_size)/ self.stride + 1) // self.pool_size
        
        self.classifier_size = channels[-1] * self.input_height * self.input_width
        
        # Model Layers
        self.conv_layers = nn.Sequential(*[ConvolutionBlock(channels[i], channels[i+1], self.kernel_size, self.stride, self.padding, self.pool_size) for i in range(len(channels)-1)])
        self.classifier1 = ClassifierBlock(self.classifier_size, hidden_size, output_size, dropout_rate)
        self.classifier2 = ClassifierBlock(self.classifier_size, hidden_size, output_size, dropout_rate)


    def forward(self,x):
        # Convolutional Layers
        out1 = self.conv_layers(x)

        # keep batch size and flatten the rest
        out2 = out1.view(out1.size(0), -1)

        # Classifier Layers
        out3 = self.classifier1(out2)
        out4 = self.classifier2(out2)
        return out3, out4
    
class SkipBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, pool_size: int = 2):
        super(SkipBlock, self).__init__()

        # Conv Layers + Skip Connections
        self.conv1 = ConvolutionBlock(in_channels, hidden_channels, kernel_size, stride, padding, pool_size)
        self.conv2 = ConvolutionBlock(hidden_channels, out_channels, kernel_size, stride, padding, pool_size)

        # TODO: Try out 2 different skip connection implementation
        # self.skip_connection = nn.Sequential(nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=2*self.stride, padding=self.padding),
        #                                       nn.AvgPool2d(self.pool_size, self.pool_size))
        
        self.skip_connection = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
                                              nn.AvgPool2d(2*pool_size, 2*pool_size)
                                              )
        
    def forward(self, x):
        x1 = self.conv1(x)
        out = self.conv2(x1)

        skip = self.skip_connection(x)
        out += skip
        return out

class SkipNet(nn.Module):
    def __init__(self, channels: tuple[int], img_size: tuple[int], hidden_size: tuple[int], output_size: int, dropout_rate: int = 0.2):
        super(SkipNet, self).__init__()
        
        if len(channels) % 2 != 1:
            raise Exception("len(channels) must be an odd number!")
        elif channels[0] != 3:
            raise Exception("first element in channels must match the input channel size!")

        # Hyperparameters for Convolutional Layers
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.pool_size = 2

        # Input size of Image
        self.input_height, self.input_width = img_size

        # Calculate Input Size of Classifier
        for i in range(len(channels) - 1):
            self.input_height = math.floor((self.input_height + 2 * self.padding - self.kernel_size)/ self.stride + 1) // self.pool_size
            self.input_width =  math.floor((self.input_width  + 2 * self.padding - self.kernel_size)/ self.stride + 1) // self.pool_size
            
        self.classifier_size = channels[-1] * self.input_height * self.input_width

        # Conv Layers + Skip Connections
        self.skip_layers = nn.Sequential(*[SkipBlock(channels[i], channels[i+1], channels[i+2], self.kernel_size, self.stride, self.padding, self.pool_size) for i in range(0, len(channels)-2, 2)])

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)

        # Classifier Layers
        self.classifier1 = ClassifierBlock(self.classifier_size, hidden_size, output_size, dropout_rate)
        self.classifier2 = ClassifierBlock(self.classifier_size, hidden_size, output_size, dropout_rate)
        

    def forward(self,x):
        # SkipBlock layers
        x1 = self.skip_layers(x)

        # Flatten output of convolutions
        x2 = x1.view(x1.size(0), -1)
        # print(x.shape)
        x3 = self.dropout(x2)

        # Classifier layer
        x4 = self.classifier1(x3)
        x5 = self.classifier2(x3)
        
        return x4, x5

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, res_layers: tuple[int], classifier_layers1: tuple[int], classifier_layers2: tuple[int], num_classes: int=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.layer1 = self.make_layer(64, res_layers[0])
        self.layer2 = self.make_layer(128, res_layers[1], stride=2)
        self.layer3 = self.make_layer(256, res_layers[2], stride=2)
        self.layer4 = self.make_layer(512, res_layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classifier_layers1[0])

        self.classifier1 = ClassifierBlock(classifier_layers1[0], classifier_layers1[1:], num_classes, 0.2)
        self.classifier2 = ClassifierBlock(classifier_layers2[0], classifier_layers2[1:], num_classes, 0.2)
    
    def make_layer(self, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
            
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2