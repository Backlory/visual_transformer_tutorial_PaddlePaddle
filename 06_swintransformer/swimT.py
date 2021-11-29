# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https:///github.com/BR-IDL/PaddleViT)
# 2021.11

import paddle
import paddle.nn as nn

paddle.set_device('cpu')


class Mlp(nn.Layer):
    '''
    多层感知机模块。
    输入[所有图片的总窗口数, 窗口的总patch数, embed_dim]，输出还是它。
    或者输入[总图片数, 图片上的总patch数, embed_dim]，输出也还是它。
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
    '''将一组NCHW的像素图片按尺寸patch_size转换成patch，
    再转换成n*word*embed_dim的视觉词向量，再layerNorm。
    [n, c, h, w] -> [n, n_patches, embed_dim]'''
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_size = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_size(x)      # [n, c, h, w]->[n, embed_dim, h', w']
        x = x.flatten(2)            # -> [n, embed_dim, n_patches]
        x = x.transpose([0, 2, 1])  # -> [n, n_patches, embed_dim]
        #
        x = self.norm(x)
        return x

class PatchMerging(nn.Layer):
    '''邻域patch之间的融合。先融合到embed_dim维度，再对embed_dim维度做降维
    [n个图, n_patches, embed_dim] ->  [n个图, n_patches, embed_dim]
    '''
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution  #原图分成patch后，横纵方向的patch个数
        self.dim = dim                      # embed_dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution                  #原图有h × w个patches。
        b, _, c = x.shape                       # [n个图, n_patches, embed_dim]

        x = x.reshape([b, h, w, c])             # ->[n个图, H方向的patch个数, W方向的patch个数, embed_dim]

        x0 = x[:, 0::2, 0::2, :]                # 取四个位置的patch
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)    # [n个图, h/2, w/2, 4*c]， 对patch进行融合
        x = x.reshape([b, -1, 4 * c])                   # -> [n个图, h*w/4, 4*c]
        x = self.norm(x)    
        x = self.reduction(x)                           # 在最后一维度做降维

        return x


def windows_partition(x, ws):
    '''视觉词视图转窗口视图。x是n个图的embed结果，ws是windowsize，代表每个window中包括多少个patch
    [n, H纵向视觉词个数, W横向视觉词个数, embed_dim] -> [所有图片的总窗口数B*num_patches, 窗口纵尺寸, 窗口横尺寸, embed_dim]
    '''
    B, H, W, C = x.shape                # [n, H纵向视觉词个数, W横向视觉词个数, embed_dim]
    x = x.reshape([B, H//ws, ws, W//ws, ws, C]) # -> [n, 纵窗口数, 窗口纵尺寸, 横窗口数, 窗口横尺寸, embed_dim]
    x = x.transpose([0, 1, 3, 2, 4, 5]) # -> [n, 纵窗口数, 横窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim] 4张图,高含8个,宽含8个,高7patch,宽7patch,96的embed
    x = x.reshape([-1, ws, ws, C])      # -> [所有图片的总窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim] 256个窗口,每个高7patch,宽7patch,并含96的embed
    return x

def i_windows_partition(windows, ws, H, W):
    '''
    从窗口视图转换回视觉词视图。
    输入：windows,    [所有图片的总窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim]
    ws(ws, ws)个patch
    H,原始图片中切成patches之后，在纵向包含的patch数，如56
    W,原始图片中切成patches之后，在纵向包含的patch数，如56
    输出:
    [图片数B, 原图纵向patch数, 原图纵向patch数, embed_dim]
    '''
    B = int(windows.shape[0] // (H / ws * W / ws))      #计算图片个数，总window数除以每个图片的window数
    x = windows.reshape([B, H//ws, W//ws, ws, ws, -1])  #->[B, 纵window数, 横window数, window内纵patch数, 横, embed_dim]
    x = x.transpose([0, 1, 3, 2, 4, 5])                 #->[B, 纵window数, window内纵patch数, 横window数, 横, embed_dim]
    x = x.reshape([B, H, W, -1])                        #->[B, 纵window数*window内纵patch数, 横window数*横, embed_dim]。它是按顺序依次匹配的
    return x

class WindowAttention(nn.Layer):
    '''窗口注意力机制。对窗口执行注意力。
    和普通attention没有任何区别，窗口注意力只是应用的对象变成了一个个的小窗口，显著降低了计算量。
    输入->[所有图片的总窗口数, 窗口的总patch数, embed_dim]
    输出->[所有图片的总窗口数, 窗口的总patch数, embed_dim]
    '''
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim                      # 视觉词向量的维度
        self.dim_head = dim // num_heads    # 每个头负责多少个维度
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5  # attention 缩放尺度
        self.softmax = nn.Softmax(axis=-1)
        self.qkv = nn.Linear(dim, 3 * dim)  #qkv矩阵
        self.proj = nn.Linear(dim, dim)     #这里假设embed维度能被head整除，否则该层attention之前会出现

    def transpose_multi_head(self, x):
        '''输入Q或K或V矩阵。输出整理后的Q或K或V矩阵。从patch视图转为head视图
        B张图，N个切片，all_head_dim个embed值；根据头数分成num_heads份，每份维度为head_dim；
        然后再把num_heads放到N前边。
        
        等价于:\n
        [B, N, all_head_dim] -> [B, num_heads, num_patches, head_dim]'''
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])   # [B, num_heads, num_patches, dim_head]
        return x

    def forward(self, x):
        B, N, C = x.shape                                   # B=256个window，N=每个window包含49个tokens。每个token的embedding编码维度为96
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)       # q, k, v: [B, num_heads, num_patches, head_dim]

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)        # -> [B, num_heads, num_patches, N] 
        attn = self.softmax(attn)

        out = paddle.matmul(attn, v)                        # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3])                   # [B, num_patches, num_heads, dim_head] 
        out = out.reshape([B, N, C])                        # [B, num_patches, num_heads, dim_head] 
        out = self.proj(out)                                # [B, N, embed_dim]
        return out

class SwinBlock(nn.Layer):
    '''swim操作。
    [n, n_patches, embed_dim]->[n, n_patches, embed_dim]
    '''
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size  #窗口的尺寸
        #
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        #
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        H, W = self.resolution      # 获取视觉词的个数，代表一张图由H×W个patch构成。
        B, N, C = x.shape           # [n, n_patches, embed_dim]
        #
        # 残差连接 1， window SA后再接SA
        h = x
        x = self.attn_norm(x)
        # 窗口attention，WSA
        x = x.reshape([B, H, W, C])                         # ->[n, H纵向视觉词个数, W横向视觉词个数, embed_dim]
        x_windows = windows_partition(x, self.window_size)  # [所有图片的总窗口数B*num_patches, 窗口纵尺寸, 窗口横尺寸, embed_dim] [256,7,7,96]
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C]) #-> [所有图片的总窗口数, 窗口的总patch数, embed_dim] [256,49,96]
        attn_windows = self.attn(x_windows)                 #[所有图片的总窗口数, 窗口的总patch数, embed_dim]
        # 全图attention，SA
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])#-> [所有图片的总窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim]
        x = i_windows_partition(attn_windows, self.window_size, H, W)   #逆处理，-> [B, 纵视觉词H, 横视觉词W, C=embed_dim]
        x = x.reshape([B, H*W, C])                          #-> [B张图, patch数, embed_dim]
        x = self.attn(x)                                    #全局attention
        x = h + x

        # 残差连接 2，mlp
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x

def main():
    t = paddle.randn([4, 3, 224, 224])
    print('image shape = ', t.shape)
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    out = patch_embedding(t)                            # [4, 56*56, 96]。4张图，56*56个patch，96维的embed
    print('patch_embedding out shape = ', out.shape)
    out = swin_block(out)                               # [4, 56*56, 96]。4张图，56*56个patch，96维的embed
    print('swin_block out shape = ', out.shape)
    out = patch_merging(out)
    print('patch_merging out shape = ', out.shape)      # [4, (56/2)*(56/2), 96]。4张图，56*56个patch，96维的embed

if __name__ == '__main__':
    main()