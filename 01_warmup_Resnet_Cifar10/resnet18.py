import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Layer):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2D(in_dim, out_dim, 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_dim)
        self.conv2 = nn.Conv2D(out_dim, out_dim, 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_dim)
        self.relu = nn.ReLU()
        if stride != 1 or in_dim != out_dim:
            self.dawnsample = nn.Sequential(*[nn.Conv2D(in_dim, out_dim, 1, stride=stride),
                                            nn.BatchNorm2D(out_dim)
                                            ])
        else:
            self.dawnsample = Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.dawnsample(h)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet18(nn.Layer):
    def __init__(self, in_dim=64, num_classes=1000):
        super().__init__()
        self.in_dim = in_dim
        # stem 部分
        self.conv1 = nn.Conv2D(in_channels=3, 
                                out_channels=in_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias_attr=False
                                )
        self.bn1 = nn.BatchNorm2D(in_dim)
        self.relu = nn.ReLU()
        #blocks 残差块
        self.layer1 = self._make_layer(dim = 64, n_blocks=2, stride = 1)
        self.layer2 = self._make_layer(dim = 128, n_blocks=2, stride = 2)
        self.layer3 = self._make_layer(dim = 256, n_blocks=2, stride = 2)
        self.layer4 = self._make_layer(dim = 512, n_blocks=2, stride = 2)
        
        #head layer 头部
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, dim, n_blocks, stride):
        '''
        根据参数生成特定的残差残差块层
        '''
        layer_list = []
        layer_list.append(Block(self.in_dim, dim, stride))
        self.in_dim = dim
        for i in range(1, n_blocks):
            layer_list.append(Block(self.in_dim, dim, stride=1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
         

def main():
    #paddle.set_device('cpu')
    model = ResNet18()
    print(model)
    x = paddle.randn([2, 3, 32, 32])
    out = model(x)
    print(out.shape)
    #paddle.summary(model, (4,3,32,32))

if __name__ == "__main__":
    main()
