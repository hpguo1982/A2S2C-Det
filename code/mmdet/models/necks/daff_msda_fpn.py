import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class DAFF_MSDA_FPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        # --- 构建卷积层 ---
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cross_convs = nn.ModuleList()

        # --- 为每个层级准备补偿模块 ---
        self.compensation_modules = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=None if self.no_norm_on_lateral else norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            cross_conv = DAFF(out_channels)

            # 为每个层级分配对应的补偿模块
            if i == 0:  # P2 - 低层语义补偿
                comp_module = CompensationSemantic(out_channels)
            elif i == 1:  # P3 - 低层语义补偿
                comp_module = CompensationSemantic(out_channels)
            else:  # P4, P5 - 高层空间补偿
                comp_module = CompensationSpatial(out_channels)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.cross_convs.append(cross_conv)
            self.compensation_modules.append(comp_module)

        # --- 额外层 ---
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_ch = self.in_channels[
                    self.backbone_end_level - 1] if i == 0 and self.add_extra_convs == 'on_input' else out_channels
                extra_fpn_conv = ConvModule(
                    in_ch, out_channels, 3, stride=2, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 1. lateral连接
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        # 2. 自上而下融合，每一层融合后都进行补偿
        used_backbone_levels = len(laterals)

        # 从最高层开始
        current_level = used_backbone_levels - 1
        compensated_features = [None] * used_backbone_levels

        # P5层先补偿（因为没有更高层与之融合）
        compensated_p5 = self.compensation_modules[current_level](laterals[current_level])
        compensated_features[current_level] = compensated_p5
        current_level -= 1

        # 自上而下融合：高层补偿后的特征与低层特征融合，融合结果再补偿
        while current_level >= 0:
            # 获取上一层已经补偿后的特征
            prev_compensated_high = compensated_features[current_level + 1]

            # 上采样高层补偿特征
            if 'scale_factor' in self.upsample_cfg:
                upsampled_high = F.interpolate(prev_compensated_high, **self.upsample_cfg)
            else:
                prev_shape = laterals[current_level].shape[2:]
                upsampled_high = F.interpolate(prev_compensated_high, size=prev_shape, **self.upsample_cfg)

            # DAFF融合：当前层特征 + 上采样的高层补偿特征
            fused_feature = self.cross_convs[current_level + 1](
                laterals[current_level],
                upsampled_high
            )

            # 对融合结果进行补偿
            compensated_feature = self.compensation_modules[current_level](fused_feature)
            compensated_features[current_level] = compensated_feature

            current_level -= 1

        # 3. FPN卷积输出（对补偿后的特征进行最终卷积）
        outs = [
            self.fpn_convs[i](compensated_features[i])
            for i in range(used_backbone_levels)
        ]

        # 4. 额外层处理
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                extra_out = self.fpn_convs[len(outs)](extra_source)
                outs.append(extra_out)

                for i in range(len(outs), self.num_outs):
                    if self.relu_before_extra_convs:
                        out = self.fpn_convs[i](F.relu(outs[-1]))
                    else:
                        out = self.fpn_convs[i](outs[-1])
                    outs.append(out)

        return tuple(outs)



class DAFF(BaseModule):
    def __init__(self, channel, init_cfg=None):
        super().__init__(init_cfg)
        self.pre_low_conv = ConvModule(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=None,
            act_cfg=None,
            inplace=False
        )
        self.pre_high_conv = ConvModule(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=None,
            act_cfg=None,
            inplace=False
        )

        self.SE_add = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.Sigmoid()
        )
        self.SE_diff = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, low_feature, high_feature):
        # [B,C,H,Wl] + [B,C,H,W] -> [B,C,H,W]
        pre_low_feature = self.pre_low_conv(low_feature)  # 先经过3*3卷积
        pre_high_feature = self.pre_high_conv(high_feature)
        add = pre_low_feature + pre_high_feature  # 计算侧面连接和边缘
        diff = pre_low_feature - pre_high_feature

        intermedia_add = add  # 中间变换也可能不需要
        intermedia_diff = diff
        # 这里可能要进行一定修改，把交叉的SE模块给去掉，后面也不能加SE模块
        SE_add = intermedia_add * self.SE_add(
            torch.mean(intermedia_add, dim=[2, 3], keepdims=True)) + intermedia_add


        SE_diff = intermedia_diff * self.SE_diff(intermedia_diff) + intermedia_diff

        result_low_feature = self.alpha * SE_add + self.beta * SE_diff
        return result_low_feature


class CompensationSpatial(nn.Module):
    """空间补偿模块 (高层特征 → 通道注意力 C×1×1)"""

    def __init__(self, channels):
        super().__init__()
        # DW-Conv 提取空间纹理
        self.deform_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 通道级参数 (1×C×1×1)
        self.contrast_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.contrast_offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        deformed = self.deform_conv(x)

        # 使用全局均值计算能量
        local_energy = torch.mean(torch.abs(deformed), dim=[2, 3], keepdim=True)  # [B,C,1,1]
        global_energy = torch.mean(torch.abs(x), dim=[2, 3], keepdim=True)  # [B,C,1,1]

        energy_ratio = local_energy / (global_energy + 1e-6)
        attn = torch.sigmoid(self.contrast_scale * energy_ratio + self.contrast_offset)  # [B,C,1,1]

        return deformed * attn + x * (1 - attn)


class CompensationSemantic(nn.Module):
    """语义补偿模块 (低层特征 → 空间注意力 1×H×W)"""

    def __init__(self, channels):
        super().__init__()
        # 1×1卷积进行语义变换
        self.semantic_conv = nn.Conv2d(channels, channels, 1)

        # 通道级参数 (1×1×1×1)
        self.contrast_scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.contrast_offset = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        semantic = self.semantic_conv(x)

        # 使用通道均值计算空间能量
        local_energy = torch.mean(torch.abs(semantic), dim=1, keepdim=True)  # [B,1,H,W]
        global_energy = torch.mean(torch.abs(x), dim=1, keepdim=True)  # [B,1,H,W]

        energy_ratio = local_energy / (global_energy + 1e-6)
        attn = torch.sigmoid(self.contrast_scale * energy_ratio + self.contrast_offset)  # [B,1,H,W]

        return semantic * attn + x * (1 - attn)