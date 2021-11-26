# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle as paddle
import paddle.nn as nn

paddle.set_device('cpu')

class Attention(nn.Layer):
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
        self.proj = nn.Linear(self.all_head_dim, self.embed_dim)    #最后把all_head_dim投影到embed_dim，这俩差不多大
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
        attn = self.softmax(attn)                           # softmax归一化, [B, num_heads, N, N]
        attn = self.attention_dropout(attn)
        #
        out = paddle.matmul(attn, v)                        # [B, num_heads, N, head_dim]           =[8, 4, 16, 24]
        out = out.transpose([0, 2, 1, 3])                   # out: [B, N, num_heads, head_dim]      =[8, 16, 4, 24]
        out = out.reshape([B, N, -1])                       # out: [B, N, num_heads*head_dim]       =[8, 16, 96]

        out = self.proj(out)
        out = self.dropout(out)
        return out

def main():
    t = paddle.randn([8, 16, 96])   # image tokens，8张图，每张图16个tokens，每个tokens的embedding编码维度为96
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    print(model)
    out, attn_weight = model(t)
    print(out.shape)
    print(attn_weight.shape)


if __name__ == "__main__":
    main()