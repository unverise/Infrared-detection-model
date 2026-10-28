"""
DNANet: Dense Nested Attention Network
论文: Dense Nested Attention Network for Infrared Small Target Detection (TIP 2023)

这个文件只定义网络结构，不包含训练循环。训练入口在 train.py，
实例化方式（默认参数）是:

    nb_filter, num_blocks = load_param('three', 'resnet_18')
    # nb_filter   = [16, 32, 64, 128, 256]   五个尺度的通道数，记作 C0..C4
    # num_blocks  = [2, 2, 2, 2]             由浅到深，每个 stage 堆几个残差块
    DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block,
           num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision='True')

红外图在 utils.py 里被 convert('RGB') 复制成 3 通道，所以 input_channels=3，
并不是网络真的在用彩色信息。num_classes=1，输出一张单通道的目标概率图（logits）。

---------------------------------------------------------------------------
网络在干什么
---------------------------------------------------------------------------
红外小目标往往只有几个像素。普通 U-Net 一路下采样到 1/16 之后，目标会被池化掉，
深层特征里几乎看不见它。DNANet 用两件事把浅层的小目标信息留住:

1. 密集嵌套跳连（思路来自 UNet++）
   不只有一条“编码器 → 解码器”的跳连，而是在每两个相邻尺度之间反复做
   上采样 / 下采样，并把同一行上已经算过的所有节点都拼进来。
   浅层高分辨率特征会被后面的节点一次又一次重新看到。

2. 每个卷积节点内部加 CBAM 注意力
   通道注意力挑“哪些特征图有用”，空间注意力挑“图上哪些位置有用”，
   用来压背景杂波、突出稀疏的小目标。

---------------------------------------------------------------------------
节点命名（读 forward 之前先记住这个）
---------------------------------------------------------------------------
变量名 x{行}_{列}，层名 conv{行}_{列}，两个下标的含义固定:

    行 i = 分辨率等级，越大图越小
        i=0  原图大小 H×W          通道 nb_filter[0] = C0 = 16
        i=1  H/2 × W/2             通道 C1 = 32
        i=2  H/4 × W/4             通道 C2 = 64
        i=3  H/8 × W/8             通道 C3 = 128
        i=4  H/16 × W/16           通道 C4 = 256
        （下面的数字都是默认 channel_size='three'、输入 256×256 时的具体值）

    列 j = 这一行上第几个节点
        j=0  编码器主干，只做“下采样 + 卷积”，不融合别的尺度
        j>=1 嵌套节点，输入 = 左边所有同尺度节点 + 下面上来的特征 +（若有）上面下来的特征

把所有节点摆成一张表，forward() 是按“对角线”从左上往右下填的:

        列 j=0        j=1        j=2        j=3        j=4
    i=0  x0_0  -----> x0_1 ----> x0_2 ----> x0_3 ----> x0_4 ──┐
    i=1  x1_0  -----> x1_1 ----> x1_2 ----> x1_3             │ 五个尺度
    i=2  x2_0  -----> x2_1 ----> x2_2                        │ 各自最右边
    i=3  x3_0  -----> x3_1                                   │ 的节点拉回
    i=4  x4_0                                                │ 原图大小再融合
                                                             ▼
                                                        Final_x0_4 → 1×1 → 预测图

    竖着往下（只发生在 j=0 这一列）: MaxPool 2×2，尺寸减半。
    斜着往上: bilinear 上采样 ×2（self.up）。
    斜着往下: bilinear 下采样 ×0.5（self.down），不是 MaxPool。
    横着往右: 同一行已经算完的节点，通道维直接拼接，不改空间尺寸。

一个嵌套节点 x{i}_{j}（j>=1）的输入通道可以套这个公式核对 __init__ 里的数字:

    j 个同尺度旧节点，每个 nb_filter[i] 通道
    + 从下一行上来的 up(x{i+1}_{j-1})，nb_filter[i+1] 通道
    + 若 i>0，再加从上一行下来的 down(x{i-1}_{j})，nb_filter[i-1] 通道

例如 x1_2: 2*C1 + C2 + C0 = 2*32 + 64 + 16 = 144，对应 conv1_2 的 in_channels。

---------------------------------------------------------------------------
默认配置下各层堆几块（backbone='resnet_18'）
---------------------------------------------------------------------------
num_blocks = [2, 2, 2, 2] 只作用在第 1~4 行。
第 0 行（原图分辨率）的 conv0_* 都没传 num_blocks，_make_layer 默认只堆 1 块。
也就是说: 越深的 stage 卷积越多，最浅的那一行最轻，避免在 H×W 上算太贵。

train.py 里写死了 block=Res_CBAM_block。
--backbone vgg_10 只会把 num_blocks 改成 [1,1,1,1]，不会自动换成 VGG_CBAM_Block。
VGG_CBAM_Block 是另一套可选的节点实现，要自己把 block= 参数换掉才会用到。

---------------------------------------------------------------------------
输出
---------------------------------------------------------------------------
网络内部不做 sigmoid。返回的是 logits，SoftIoULoss（model/loss.py）里才会 sigmoid。

deep_supervision=True 时返回 4 张原图大小的预测，对应嵌套越来越深的四个结果:
    output1 ← x0_1          只融合过 1 次的浅层结果
    output2 ← x0_2
    output3 ← x0_3
    output4 ← Final_x0_4    多尺度融合之后的最终结果
训练时四张图都和同一个 mask 算 SoftIoU，再取平均。
测试时 train.py 取 preds[-1]，也就是 output4。

阅读顺序见文件末尾不方便翻，聊天回复里有一份从哪一行开始看的清单。
建议先读本说明和 DNANet.forward，再回头看 __init__ 里的通道数。
"""

import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    """
    不带残差的卷积节点，对应论文里的 DNANet-VGG-10。

    结构（空间尺寸不变，因为两层都是 3×3、padding=1）:
        3×3 卷积 → BN → ReLU → 3×3 卷积 → BN → 通道注意力 → 空间注意力 → ReLU

    和 Res_CBAM_block 的差别只有一条: 这里没有 shortcut，输入不会被加回来。
    当前 train.py / test.py 没有使用这个类，默认节点是 Res_CBAM_block。

    通道: in_channels → out_channels（第一层卷积负责变通道），之后保持 out_channels。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一层 3×3: 改变通道数，padding=1 保证 H、W 不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二层 3×3: 通道数不变，继续在局部 3×3 邻域里混合特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下面这行把上面的 self.relu 覆盖掉了。两次构造的都是同样的 inplace ReLU，
        # 行为没有变化，只是第一次赋值是多余的。读代码时不要以为这里有两个不同的激活。
        self.relu = nn.ReLU(inplace=True)
        # CBAM 的两支注意力，都作用在 out_channels 上，不改变张量形状
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # x: (B, in_channels, H, W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # ca(out) 形状是 (B, C, 1, 1)，按通道广播相乘，给每个通道一个 0~1 的权重
        out = self.ca(out) * out
        # sa(out) 形状是 (B, 1, H, W)，按空间位置广播相乘，给每个像素一个 0~1 的权重
        out = self.sa(out) * out
        out = self.relu(out)
        # 返回 (B, out_channels, H, W)
        return out

class ChannelAttention(nn.Module):
    """
    CBAM 的通道注意力（Woo et al., ECCV 2018）。

    回答的问题: 这 C 张特征图里，哪些通道该被放大、哪些该被压掉。
    对红外小目标，目标响应往往集中在少数通道，背景杂波占满其余通道，
    所以先在通道上做一次筛选。

    做法:
        1. 全局平均池化、全局最大池化，各自把 (B,C,H,W) 收成 (B,C,1,1)
           平均池化看这个通道的整体响应，最大池化看这个通道里最强的那个点
           （小目标像素少，最大池化不容易被大片背景平均掉）
        2. 两支共用同一个两层 1×1 卷积（相当于共享 MLP）:
              C → C/16 → ReLU → C
           先压缩再还原，是为了让通道之间发生交互，同时少用参数。
        3. 两支相加，sigmoid 压到 (0, 1)，得到每个通道一个门控系数。

    返回 (B, C, 1, 1)，调用处用 out * ca(out) 做广播乘法。

    参数 ratio 写在签名里但函数体没有用它，缩减倍数被写死成 16
    （in_planes // 16）。传别的 ratio 不会生效。
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 输出空间尺寸固定为 1×1，和输入 H、W 无关
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 用 1×1 卷积实现 MLP。作用在 1×1 特征图上时，和 nn.Linear 等价，
        # 而且不用把张量展平成 (B, C)。bias=False 是 CBAM 原文的写法。
        # 缩减比写死为 16，没有读取上面的 ratio。
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        # 两支共用 fc1/relu1/fc2，所以“平均”和“最大”走的是同一套通道权重
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # (B, C, 1, 1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # (B, C, 1, 1)
        out = avg_out + max_out
        # 每个通道一个 (0, 1) 系数。越接近 1 越保留，越接近 0 越抑制。
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    CBAM 的空间注意力。

    回答的问题: 这张 H×W 的图上，哪些位置更像目标。
    通道注意力已经挑过“看哪些特征图”，这里再挑“看图上的哪里”。
    小目标检测里这一步相当于在空间上压低大面积背景。

    做法:
        1. 沿通道维做平均、做最大，得到两张 H×W 的图，拼成 (B, 2, H, W)
        2. 一个 7×7 卷积把 2 通道收成 1 通道（padding=3，H、W 不变）
        3. sigmoid 得到每个像素一个门控系数

    返回 (B, 1, H, W)，调用处用 out * sa(out) 做广播乘法。
    kernel_size 只允许 3 或 7，CBAM 原文默认 7，感受野更大，适合看局部上下文。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 7×7 要 padding=3，3×3 要 padding=1，才能保持空间尺寸
        padding = 3 if kernel_size == 7 else 1
        # 输入固定是 2 通道（通道平均图 + 通道最大图），输出 1 通道的空间权重图
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)。dim=1 是通道维。
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B, 1, H, W)，下划线丢掉最大值的下标
        x = torch.cat([avg_out, max_out], dim=1)           # (B, 2, H, W)
        x = self.conv1(x)                                   # (B, 1, H, W)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    """
    带残差的卷积节点，是 train.py 实际使用的 block。

    结构（默认 stride=1，空间尺寸不变）:
        主分支: 3×3 → BN → ReLU → 3×3 → BN → 通道注意力 → 空间注意力
        捷径:   通道数不变时就是输入本身；通道数要变时用 1×1 卷积把输入投影到同样通道
        输出:   主分支 + 捷径，再 ReLU

    残差的作用: 让这个节点至少能把输入原样传下去（加上一个可学习的残差）。
    层数堆深时梯度不容易断，浅层的小目标响应也不容易被后面的卷积完全改写掉。

    注意注意力加在“相加之前”的主分支上，捷径不过 CBAM。
    也就是说注意力只调节新算出来的残差，恒等映射那一路保持原样。

    stride 参数保留了，但 DNANet 调用时从不传 stride（一直是默认 1）。
    下采样全部交给外面的 MaxPool，不靠残差块自己做 stride=2。
    因此 shortcut 分支会不会出现，只取决于 in_channels 和 out_channels 是否相同:
        一个 stage 的第一块通常要变通道，走 1×1 shortcut；
        后面几块通道不变，shortcut 是 None，直接用输入做残差。
    """

    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        # 第一层卷积同时负责: 变通道，以及（如果传入 stride>1）缩小空间尺寸。
        # 本网络里 stride 始终为 1，所以这里只变通道。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        # 第二层卷积通道数、空间尺寸都不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差相加要求主分支和捷径的形状完全一致。
        # 通道变了，或者 stride 把尺寸变了，输入就不能直接加，要用 1×1 卷积对齐。
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # x: (B, in_channels, H, W)
        # 先把捷径准备好。shortcut 为 None 时，残差就是输入本身（恒等映射）。
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)          # (B, out_channels, H', W')

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 先通道门控，再空间门控。顺序和 CBAM 原文一致: 通道注意力在前。
        out = self.ca(out) * out
        out = self.sa(out) * out
        # 残差相加。inplace ReLU 作用在相加之后的结果上，不改 residual 那个张量。
        out += residual
        out = self.relu(out)
        # 返回 (B, out_channels, H', W')。本网络中 H'、W' 等于输入的 H、W。
        return out

class DNANet(nn.Module):
    """
    密集嵌套注意力网络本体。

    可以把它看成一张 5 行的特征表（见文件顶部的图）:
        第 0 列是编码器，MaxPool 逐级把 256×256 收到 16×16；
        右边每一列都是 UNet++ 式的嵌套融合节点；
        最后把每一行最右边的节点上采样回原图大小，拼在一起再卷积一次，得到分割 logits。

    参数
    ----
    num_classes:
        输出通道数。红外小目标分割是二分类前景/背景，这里传 1。
    input_channels:
        输入图像通道数。数据加载时灰度图被复制成 RGB，所以是 3。
    block:
        每个节点用哪种卷积块。train.py 传的是 Res_CBAM_block。
        换成 VGG_CBAM_Block 就是论文里的 VGG 版本。
    num_blocks:
        长度 4 的列表，依次是第 1、2、3、4 行每个节点里堆多少个 block。
        resnet_18 → [2,2,2,2]；resnet_34 → [3,4,6,3]；resnet_10 / vgg_10 → [1,1,1,1]。
        第 0 行不看这个列表，固定 1 个 block。
    nb_filter:
        长度 5 的列表，五个尺度的通道数 [C0,C1,C2,C3,C4]。
        channel_size='three'（默认）时是 [16, 32, 64, 128, 256]。
    deep_supervision:
        为真时，第 0 行的中间节点 x0_1、x0_2、x0_3 和最终融合结果各接一个 1×1，
        返回 4 张预测图，训练时每张都算损失（深监督）。
        为假时只返回最终那一张。

        注意: train.py 把 argparse 的字符串 'True' / 'False' 原样传进来。
        非空字符串在 Python 里都是真值，所以字符串 'False' 也会走进深监督分支。
        真正按字符串判断的是 train.py 里的 `args.deep_supervision == 'True'`。
        默认配置两边都是开启，读默认训练流程时不用担心这个差异。
    """

    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        # 编码器下采样: 2×2 最大池化，尺寸减半，通道不变。只用于第 0 列。
        self.pool  = nn.MaxPool2d(2, 2)
        # 嵌套节点之间的尺度对齐。mode='bilinear' 是线性插值，align_corners=True
        # 让角点像素对齐，上采样后再拼回原尺度时边界更稳。
        # up:   放大 2 倍，例如 H/2 → H，H/4 → H/2
        # down: 缩小到一半。用双线性而不是 MaxPool，是为了把浅层细节“平滑地”送回深层，
        #       避免再做一次最大池化把只有几个像素的目标抹掉。
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        # 最终融合时，深层特征要一次拉回原图大小，所以另外准备 ×4、×8、×16。
        # self.up 已经覆盖 ×2，这里不再重复定义。
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        # ------------------------------------------------------------------
        # 第 0 列: 编码器。只吃上一行池化后的特征，不和其他列融合。
        # _make_layer 的第 4 个参数是堆几块；省略时默认 1。
        # 第 0 行固定 1 块，第 1~4 行用 num_blocks[0]~[3]。
        # ------------------------------------------------------------------
        # conv0_0: (B, 3, H, W)         → (B, C0, H, W)       例如 3 → 16，256×256
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        # conv1_0: (B, C0, H/2, W/2)    → (B, C1, H/2, W/2)   例如 16 → 32，128×128
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        # conv2_0: C1 → C2，H/4
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        # conv3_0: C2 → C3，H/8
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        # conv4_0: C3 → C4，H/16。这是最深的节点，整张表里第 4 行只有它一个。
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        # ------------------------------------------------------------------
        # 第 1 列: 每个节点第一次和相邻尺度融合。
        # 输入通道 = 左边同尺度节点 + 下面上来的节点 +（第 0 行没有）上面下来的节点。
        # 输出通道收回这一行自己的宽度 nb_filter[i]，方便右边的列继续拼接。
        # ------------------------------------------------------------------
        # conv0_1 吃 [x0_0, up(x1_0)]           通道 C0+C1          → C0
        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        # conv1_1 吃 [x1_0, up(x2_0), down(x0_1)] 通道 C1+C2+C0      → C1
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        # conv2_1 吃 [x2_0, up(x3_0), down(x1_1)] 通道 C2+C3+C1      → C2
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        # conv3_1 吃 [x3_0, up(x4_0), down(x2_1)] 通道 C3+C4+C2      → C3
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        # ------------------------------------------------------------------
        # 第 2 列: 同尺度节点变成 2 个（所以 nb_filter[i]*2），再加上下相邻尺度各 1 个。
        # ------------------------------------------------------------------
        # conv0_2 吃 [x0_0, x0_1, up(x1_1)]                 通道 C0*2+C1     → C0
        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        # conv1_2 吃 [x1_0, x1_1, up(x2_1), down(x0_2)]     通道 C1*2+C2+C0  → C1
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        # conv2_2 吃 [x2_0, x2_1, up(x3_1), down(x1_2)]     通道 C2*2+C3+C1  → C2
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        # ------------------------------------------------------------------
        # 第 3 列: 同尺度旧节点 3 个。第 2 行到这里已经结束（没有 conv2_3）。
        # ------------------------------------------------------------------
        # conv0_3 吃 [x0_0, x0_1, x0_2, up(x1_2)]                    通道 C0*3+C1    → C0
        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        # conv1_3 吃 [x1_0, x1_1, x1_2, up(x2_2), down(x0_3)]        通道 C1*3+C2+C0 → C1
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        # ------------------------------------------------------------------
        # 第 4 列: 只剩原图尺度这一行。4 个旧的 C0 节点 + 从第 1 行上来的 C1。
        # ------------------------------------------------------------------
        # conv0_4 吃 [x0_0, x0_1, x0_2, x0_3, up(x1_3)]      通道 C0*4+C1 → C0
        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # 多尺度融合之后再过一个节点。5 路特征都先被 1×1 收成 C0，所以输入是 C0*5。
        # 输出仍是 C0，交给最后的 1×1 分类卷积。
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        # 融合前的通道对齐。每一行最深的那个节点通道数不同（C4/C3/C2/C1），
        # 1×1 卷积只改通道、不改空间尺寸，把它们都变成 C0，才能和 x0_4 拼接。
        # x0_4 自己已经是 C0，所以没有 conv0_0_1x1。
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)  # x4_0: C4 → C0
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)  # x3_1: C3 → C0
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)  # x2_2: C2 → C0
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)  # x1_3: C1 → C0

        # 分类头: C0 → num_classes（默认 1），1×1 卷积，空间尺寸不变。
        # 没有 sigmoid，输出是 logits。
        # 深监督的 4 个头权重不共享，各自学习“在这个融合深度上怎么变成一张预测图”。
        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)  # 接在 x0_1 上
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)  # 接在 x0_2 上
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)  # 接在 x0_3 上
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)  # 接在最终融合上
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        """
        把 num_blocks 个卷积块串成一个 nn.Sequential，作为一个节点。

        第一块负责把 input_channels 变成 output_channels（通道对不齐时，
        Res_CBAM_block 内部会建 1×1 shortcut）。
        后面的 num_blocks-1 块输入输出都是 output_channels，走恒等残差。

        这里不传 stride，所以每一块的 stride 都是 Res_CBAM_block 的默认值 1，
        节点内部不会缩小特征图。缩小只发生在 forward 里的 self.pool。

        第 0 行的 conv0_* 都不传 num_blocks，于是每个原图尺度节点只有 1 块。
        """
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        """
        input: (B, input_channels, H, W)，默认 (B, 3, 256, 256)。

        计算顺序不是“先把编码器走完再解码”，而是编码器每加深一层，
        就立刻把右边能算的嵌套节点补上。这样浅层节点可以尽快把信息
        送回深层，深层节点也可以马上把语义送回浅层。

        下面每个赋值后面的形状，都按默认 nb_filter=[16,32,64,128,256]、H=W=256 来写。
        换 channel_size 时通道数跟着变，空间尺寸的比例不变。
        torch.cat(..., 1) 的 1 是通道维（dim=0 是 batch）。
        """

        # ===== 对角线 1: 原图 + 半分辨率，得到第一个浅层融合节点 x0_1 =====
        # 编码器第 0 行。还没有下采样，保留最细的空间位置，小目标在这里仍是几个像素。
        x0_0 = self.conv0_0(input)                                      # (B, 16, 256, 256)
        # 池化减半后再卷积，进入第 1 行。
        x1_0 = self.conv1_0(self.pool(x0_0))                            # (B, 32, 128, 128)
        # x0_1 的输入: 原图特征 x0_0，拼上放大回 256 的 x1_0。
        # 这一步就是最浅的一次 U 形跳连: 细节 + 稍大一点的感受野。
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))        # cat (B, 48, 256, 256) → (B, 16, 256, 256)

        # ===== 对角线 2: 编码器再到 H/4，并补上 x1_1、x0_2 =====
        x2_0 = self.conv2_0(self.pool(x1_0))                            # (B, 64, 64, 64)
        # x1_1 三路拼接:
        #   x1_0            同尺度、左边已有的节点          32 通道, 128×128
        #   up(x2_0)        更深层上来的语义                64 通道, 放大到 128×128
        #   down(x0_1)      更浅层下来的细节（刚算出来的）  16 通道, 缩小到 128×128
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        #                                                                 cat (B, 112, 128, 128) → (B, 32, 128, 128)
        # x0_2 把第 0 行已经有的两个节点都留下（密集连接），再拼上从 x1_1 上来的特征。
        # 注意拼的是 x0_0 和 x0_1 本身，不是只拼最近的一个。这就是 "Dense"。
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))  # cat (B, 64, 256, 256) → (B, 16, 256, 256)

        # ===== 对角线 3: 编码器到 H/8，从下往上补 x2_1、x1_2、x0_3 =====
        x3_0 = self.conv3_0(self.pool(x2_0))                            # (B, 128, 32, 32)
        # 同样的三路模式，整体下移一行: 左邻 + 下面上来 + 上面下来。
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        #                                                                 cat (B, 224, 64, 64) → (B, 64, 64, 64)
        # x1_2 的同尺度旧节点已经有 x1_0、x1_1 两个，所以是四路:
        #   两个旧节点 32+32，up(x2_1) 的 64，down(x0_2) 的 16。
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        #                                                                 cat (B, 144, 128, 128) → (B, 32, 128, 128)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        #                                                                 cat (B, 80, 256, 256) → (B, 16, 256, 256)

        # ===== 对角线 4: 编码器到最深的 H/16，再一路填回原图，得到 x0_4 =====
        x4_0 = self.conv4_0(self.pool(x3_0))                            # (B, 256, 16, 16)
        # 第 4 行没有更下面的节点了，所以 x3_1 是第 3 行的最后一个节点。
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        #                                                                 cat (B, 448, 32, 32) → (B, 128, 32, 32)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        #                                                                 cat (B, 288, 64, 64) → (B, 64, 64, 64)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        #                                                                 cat (B, 176, 128, 128) → (B, 32, 128, 128)
        # x0_4: 第 0 行四个历史节点全部保留，再加上下一行最新的 x1_3。
        # 到这里，原图尺度的特征已经反复融合过 4 次深层语义，同时每次都还拿着最初的 x0_0。
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #                                                                 cat (B, 96, 256, 256) → (B, 16, 256, 256)

        # ===== 多尺度最终融合 =====
        # 每一行取“最右边、也就是融合次数最多”的那个节点，拉回 256×256 再拼接:
        #   x4_0  16×16  → 1×1 把 256 通道收到 16 → 上采样 ×16
        #   x3_1  32×32  → 1×1 把 128 通道收到 16 → 上采样 ×8
        #   x2_2  64×64  → 1×1 把 64  通道收到 16 → 上采样 ×4
        #   x1_3  128×128 → 1×1 把 32 通道收到 16 → 上采样 ×2
        #   x0_4  256×256 已经是 16 通道，直接用
        # 五路都是 (B, 16, 256, 256)，拼起来 (B, 80, 256, 256)。
        # 深层节点提供“这是不是目标”的语义，浅层节点提供“目标在哪个像素”的位置。
        # 小目标容易在 H/16 上消失，所以不能只靠 x4_0，必须把浅层一起留下。
        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)), self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)), self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))
        # Final_x0_4: (B, 16, 256, 256)

        if self.deep_supervision:
            # 四张图都是 (B, num_classes, H, W) 的 logits，空间大小和输入相同。
            # 越往后的输出，看到的嵌套融合越多。
            # train.py 测试时用 preds[-1]，即 output4；训练时四张损失取平均。
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output
