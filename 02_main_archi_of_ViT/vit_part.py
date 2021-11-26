# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image

paddle.set_device('cpu')

class Identity(nn.Layer):
    '''
    空标识符？
    可能是为了避免在定义神经网络模型中出现if语句吧，那样无法backward
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    '''
    多层感知机模块，输入是所有patch的embed得分，再降回来。
    输入[n, flatten索引, embedding_score]，输出还是它
    '''
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Layer):
    '''
    patch的embedding器
    输入[n, c, h, w]格式的图片
    输出 [n, flatten索引, embedding_score]
    '''
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        self.patch_embedding = nn.Conv2D(in_channels, 
                                            embed_dim, 
                                            kernel_size=patch_size,
                                            stride=patch_size,
                                            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                            bias_attr=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)     # [n, c, h, w] -> [n, embedding_score, 行索引，列索引]
        x = x.flatten(2)                # [n, embedding_score, 行索引，列索引] -> [n, embedding_score, flatten索引]
        x = x.transpose([0, 2, 1])      # [n, embedding_score, flatten索引] -> [n, flatten索引, embedding_score]
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    '''
    attention机制
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EncoderLayer(nn.Layer):
    '''
    编码器结构
    '''
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Attention()
        self.attn_norm = nn.LayerNorm(embed_dim)    #post-norm，效果差于pre-norm，还得warmup，但容易收敛
        self.mlp = Mlp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)     #post-norm，效果差于pre-norm但容易收敛

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)                    #224尺寸图片，patch边长取7，三通道，最终编码尺寸为16
        self.encoders = nn.LayerList([EncoderLayer(16) for i in range(5)])  #视觉词embedding size取16，编码次数5次
        self.norm = nn.LayerNorm(16)
        self.avgpool = nn.AdaptiveAvgPool1D(1)                              #自适应平均池化后尺寸取1。此时自适应地计算所有patch的embedding得分的平均值
        self.head = nn.Linear(16, 10)                                       #分类器头，此时对每个patch得到的是10维的分类向量
        
    def forward(self, x):
        x = self.patch_embed(x)                 # [n h w c]->[n, flatten索引, embedding_score]
        for encoder in self.encoders:           # [n, flatten索引, embedding_score] -> [n, flatten索引, embedding_score]
            x = encoder(x)
        # avg
        x = self.norm(x)                        
        x = x.transpose([0, 2, 1])              #[n, flatten索引, embedding_score] -> [n, embedding_score, flatten索引]
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


def main():
    t = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(t)
    print(out.shape)


if __name__ == "__main__":
    main()
