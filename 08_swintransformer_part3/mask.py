# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https:///github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
from PIL import Image
paddle.set_device('cpu')

def window_partition(x, ws):
    '''视觉词视图转窗口视图。x是n个图的embed结果，ws是windowsize，代表每个window中包括多少个patch
    [n, H纵向视觉词个数, W横向视觉词个数, embed_dim] -> [所有图片的总窗口数B*num_patches, 窗口纵尺寸, 窗口横尺寸, embed_dim]
    '''
    B, H, W, C = x.shape                # [n, H纵向视觉词个数, W横向视觉词个数, embed_dim]
    x = x.reshape([B, H//ws, ws, W//ws, ws, C]) # -> [n, 纵窗口数, 窗口纵尺寸, 横窗口数, 窗口横尺寸, embed_dim]
    x = x.transpose([0, 1, 3, 2, 4, 5]) # -> [n, 纵窗口数, 横窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim] 4张图,高含8个,宽含8个,高7patch,宽7patch,96的embed
    x = x.reshape([-1, ws, ws, C])      # -> [所有图片的总窗口数, 窗口纵尺寸, 窗口横尺寸, embed_dim] 256个窗口,每个高7patch,宽7patch,并含96的embed
    return x

def generate_mask(ws=4, shift_size=2, input_resolution=(8, 8)):
    '''
    根据窗口尺寸生成遮罩。\n
    ws       窗口的尺寸，代表一个window中包含几乘几的patches\n
    shift_size        平移尺度，一般取 0 或者 ws//2\n
    input_resolution  原图分成patch后，横纵方向的patch个数\n
    输出：[一张图片中的总窗口数, ws*ws, ws*ws]\n
    '''
    H, W = input_resolution                             # 取出原图patch规模
    img_mask = paddle.zeros([1, H, W, 1])               # 构建分辨率与原图patch规模相同的mask
    h_slices = [slice(0, -ws),                 #下侧保留一个windowsize
                slice(-ws, -shift_size),       #下侧windowsize到shift_size
                slice(-shift_size, None)]               #下侧shift_size
    w_slices = [slice(0, -ws),
                slice(-ws, -shift_size),
                slice(-shift_size, None)]
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt                  # 给目标区域赋值为cnt
            cnt += 1                                    # cnt自增
    #print(img_mask)
    windows_mask = window_partition(img_mask, ws=ws)   #[1, h_num_patch, w_num_patch, 1] -> [总窗口数n, h_win, w_win, 1]
    windows_mask = windows_mask.reshape([-1, ws * ws])#[总窗口数, h_win, w_win, 1] -> [总窗口数n，num_patch]

    attn_mask = windows_mask.unsqueeze(1) - windows_mask.unsqueeze(2)   # 广播机制，[n, 1, ws*ws] - [n, ws*ws, 1]=[n, ws*ws, ws*ws]
    attn_mask = paddle.where(attn_mask!=0,                              # 为了画图，非0处填上255
                             paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
    return attn_mask

def main():
    #mask = generate_mask()      #ws=7, shift_size=3, input_resolution=(56, 56)
    #mask = generate_mask(ws=7, shift_size=3, input_resolution=(56, 56))      #
    mask = generate_mask(ws=7, shift_size=3, input_resolution=(7, 7))      #
    print(mask.shape)
    mask = mask.cpu().numpy().astype('uint8')
    I, J, K = mask.shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                print(mask[i, j, k], end='\t')
            print()

        im = Image.fromarray(mask[i, :, :])
        im.save(f'07_swintransformer_part2/{i}.png')
        print()
        print()
    print()

if __name__ == '__main__':
    main()