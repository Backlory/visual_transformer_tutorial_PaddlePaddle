# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https:///github.com/BR-IDL/PaddleViT)
# 2021.11

import copy
import paddle
import paddle.nn as nn

paddle.set_device('cpu')

class Identity(nn.Layer):
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
    '''[n, c, h, w] -> [n, n_patches+1, embed_dim]'''
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size)** 2
        # word embedding
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        # class token
        self.cls_token = paddle.create_parameter(shape=[1, 1, embed_dim],
                                    dtype='float32',
                                    default_initializer=paddle.nn.initializer.Constant(0.))
        # position embedding，整体叠在【n_patches(16) + classtoken(1)】上
        self.position_embed = paddle.create_parameter(shape=[1, n_patches+1, embed_dim],
                                    dtype='float32',
                                    default_initializer=nn.initializer.TruncatedNormal(std=.02))
        # 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)     # [n, c, h, w] -> [n, embed_dim, 行索引，列索引]
        x = x.flatten(2)                # [n图, embed_dim, 行索引，列索引] -> [n图, embed_dim, n_patches]
        x = x.transpose([0, 2, 1])      # [n图, embed_dim, n_patches] -> [n图, n_patches, embed_dim]
        #cls_token和x并在一起
        cls_token = self.cls_token                          # [1, 1 ,embed_dim]
        cls_token = cls_token.expand([x.shape[0], -1, -1])  #扩展到n个图片上，即[n, 1 ,embed_dim]
        x = paddle.concat([cls_token, x], axis = 1)         #[n, 1+n_patches, embed_dim]
        #pos_embed 叠加在【cls_token和x】底下
        x = x + self.position_embed # broadcasting，把[1, n_patches+1, embed_dim]广播叠加到
        #
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    '''对整张图片执行多头注意力，实现模型内部的集成学习，不同头学习到的模式可能是不一样的\n
    #输入[B个图, N个patch, embed_dim]，输出[B, N, embed_dim]
    '''
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(self.embed_dim / self.num_heads)    # 头的维度=编码维度/头的数目
        self.all_head_dim = self.head_dim * num_heads           # 用到的embed维度，不一定等于embed_dim，因为可能不整除
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,             # 要放大三倍做QKV所以乘以3
                             bias_attr=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, self.embed_dim)    #最后把all_head_dim投影到embed_dim，共同决策。这俩差不多大
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
    def transpose_multi_head(self, x):
        '''输入Q或K或V矩阵。输出整理后的Q或K或V矩阵。
        B张图，N个切片，all_head_dim个embed值；
        分成num_heads，head_dim份；
        然后再转置，把num_heads放到N前边。
        
        等价于:\n
        [B, N, all_head_dim] -> [B, num_heads, N, head_dim]
        '''
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]  #取B张图片，N个切片，num_heads=4个头，head_dim=24维度/每个头
        x = x.reshape(new_shape)                # x: [B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])           # x: [B, num_heads, N, head_dim]
        return x

    def forward(self, x):
        B, N, _ = x.shape                                   # B=8张图，N=每张图16个tokens。每个tokens的embedding编码维度为96
        qkv = self.qkv(x).chunk(3, -1)                      # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)       # q, k, v: [B, num_heads, N, head_dim]  =[8, 4, 16, 24]
        #
        attn = paddle.matmul(q, k, transpose_y=True)        # q * k^T: [B, num_heads, N, N]         =[8, 4, 16, 16]
        attn = self.scale * attn                            # 缩放
        attn = self.softmax(attn)                           # softmax归一化, [B, num_heads, N, N]。第B个图片，每个头在每个word上的注意力值
                                                            # word的数量为N×N
        attn_weights = attn
        attn = self.attention_dropout(attn)
        #
        out = paddle.matmul(attn, v)                        # [B, num_heads, N, head_dim]           =[8, 4, 16, 24]
        out = out.transpose([0, 2, 1, 3])                   # out: [B, N, num_heads, head_dim]      =[8, 16, 4, 24]
        out = out.reshape([B, N, -1])                       # out: [B, N, num_heads*head_dim]       =[8, 16, 96]

        out = self.proj(out)                                #[B, N, embed_dim]
        out = self.dropout(out) 
        return out, attn_weights        




class EncoderLayer(nn.Layer): 
    '''编码器结构，输入[B, N, embed_dim]，输出[B, N, embed_dim]'''
    def __init__(self, embed_dim=768, num_heads=8, qkv_bias=True, mlp_ratio=4.0, dropout=0., attention_dropout=0.):
        super().__init__()  #768=96*8
        self.attn = Attention(embed_dim, num_heads, qkv_bias, dropout=dropout, attention_dropout=attention_dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)    
        self.mlp = Mlp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)  

    def forward(self, x):
        h = x
        x = self.attn_norm(x)                   #pre-norm，效果好
        x, attn = self.attn(x)
        x = x + h
        #x = self.attn_norm(x)                   #post-norm，效果差于pre-norm，还需要warmup，但容易收敛

        h = x
        x = self.mlp_norm(x)                    #pre-norm
        x = self.mlp(x)
        x = x + h
        #x = self.mlp_norm(x)                    #post-norm
        return x, attn


class Encoder(nn.Layer):
    '''编码器结构，输入[B, N, embed_dim]，输出[B, N, embed_dim]和attention矩阵'''
    def __init__(self, embed_dim, depth,  num_heads=8, qkv_bias=True, mlp_ratio=4.0, dropout=0., attention_dropout=0.):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim, num_heads, qkv_bias, mlp_ratio, dropout, attention_dropout)
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        self_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            self_attn.append(attn)
        out = self.encoder_norm(x)
        return out, self_attn
        


class VisualTransformer(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 ):
        super().__init__()  
        self.patch_embedding = PatchEmbedding(image_size, 
                                patch_size, 
                                in_channels, 
                                embed_dim)  #[n, c, h, w] -> [n, 1+n_patches, embed_dim]
        self.encoder = Encoder(embed_dim, 
                                depth,
                                num_heads,
                                qkv_bias,
                                mlp_ratio, 
                                dropout, 
                                attention_dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)     # [N, C, H, W] -> [n, 1+n_patches, embed_dim]
        x, _ = self.encoder(x)          # [n, 1+n_patches, embed_dim] -> [n, 1+n_patches, embed_dim]
        x = self.classifier(x[:, 0])    # [n, 1 + n_patches, embed_dim]
        return x

def main():
    vit = VisualTransformer(image_size=224,
                            patch_size=16,
                            in_channels=3,
                            num_classes=1000,
                            embed_dim=768,
                            depth=3,
                            num_heads=8,
                            mlp_ratio=4,
                            qkv_bias=True,
                            dropout=0.,
                            attention_dropout=0.)
    print(vit)
    paddle.summary(vit, (100, 3, 224, 224)) # must be tuple

if __name__ == "__main__":
    main()

