import torch
import torch.nn as nn
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmengine.structures import InstanceData
from typing import List, Tuple, Optional


class CascadeRoIHead(StandardRoIHead):
    def __init__(self,
                 num_stages: int = 3,
                 stage_loss_weights: Tuple[float] = (1.0, 0.5, 0.3),
                 **kwargs):
        super().__init__(**kwargs)
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        # 保存原始bbox_head的配置用于构建共享分类头
        bbox_head_cfg = self.bbox_head

        # 移除父类的标准bbox_head，我们将重新构建
        del self.bbox_head

        # 构建共享分类头（使用与原始相同的结构）
        self.shared_cls_head = self._build_shared_cls_head(bbox_head_cfg)

        # 为每个阶段创建独立的回归头
        self.stage_reg_heads = nn.ModuleList()
        for i in range(num_stages):
            reg_head = self._build_reg_head(bbox_head_cfg)
            self.stage_reg_heads.append(reg_head)

    def _build_shared_cls_head(self, bbox_head_cfg):
        """构建共享分类头"""
        # 这里假设原始bbox_head有一个fc_cls属性
        # 实际实现可能需要根据你的具体bbox_head结构调整
        if hasattr(bbox_head_cfg, 'fc_cls'):
            # 如果是FC层结构的头
            cls_head = nn.Sequential(
                nn.Linear(bbox_head_cfg.in_channels, bbox_head_cfg.fc_cls.in_features),
                nn.ReLU(inplace=True),
                nn.Linear(bbox_head_cfg.fc_cls.in_features, bbox_head_cfg.fc_cls.out_features)
            )
        else:
            # 通用实现：创建一个与原始分类头结构相同的网络
            cls_head = nn.Sequential(
                nn.Linear(bbox_head_cfg.in_channels, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, self.bbox_head.num_classes + 1)  # +1 for background
            )
        return cls_head

    def _build_reg_head(self, bbox_head_cfg):
        """为每个阶段构建独立的回归头"""
        if hasattr(bbox_head_cfg, 'fc_reg'):
            # 如果是FC层结构的头
            reg_head = nn.Sequential(
                nn.Linear(bbox_head_cfg.in_channels, bbox_head_cfg.fc_reg.in_features),
                nn.ReLU(inplace=True),
                nn.Linear(bbox_head_cfg.fc_reg.in_features, 4)  # 4 coordinates
            )
        else:
            # 通用实现
            reg_head = nn.Sequential(
                nn.Linear(bbox_head_cfg.in_channels, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 4)  # 4 coordinates
            )

        # 初始化权重
        for layer in reg_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.001)
                nn.init.constant_(layer.bias, 0)

        return reg_head

    def loss(self, x: Tuple[torch.Tensor], rpn_results_list: List[InstanceData],
             batch_data_samples: List[dict]) -> dict:
        """计算多阶段损失"""
        losses = dict()

        # 获取第一阶段的proposals（来自RPN）
        proposals_list = [r.bboxes for r in rpn_results_list]
        batch_gt_instances = [s.gt_instances for s in batch_data_samples]

        # 对每个阶段进行迭代
        for stage in range(self.num_stages):
            # 1. 分配GT和采样
            assigned_results = self.bbox_assigner.assign(
                proposals_list, batch_gt_instances)
            sampling_results = self.bbox_sampler.sample(
                assigned_results, proposals_list, batch_gt_instances)

            # 2. 提取ROI特征
            rois = bbox2roi([res.priors for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            # 3. 展平特征
            bbox_feats = bbox_feats.flatten(1)

            # 4. 通过共享分类头
            cls_scores = self.shared_cls_head(bbox_feats)

            # 5. 通过当前阶段的独立回归头
            bbox_preds = self.stage_reg_heads[stage](bbox_feats)

            # 6. 计算当前阶段的损失
            stage_loss_weight = self.stage_loss_weights[stage]

            # 获取targets
            cls_targets, bbox_targets = self._get_targets(sampling_results, batch_gt_instances)

            # 计算分类损失
            loss_cls = self.bbox_head.loss_cls(
                cls_scores, cls_targets, avg_factor=cls_targets.numel())

            # 计算回归损失
            loss_bbox = self.bbox_head.loss_bbox(
                bbox_preds, bbox_targets, avg_factor=bbox_targets.size(0))

            losses[f's{stage}.loss_cls'] = loss_cls * stage_loss_weight
            losses[f's{stage}.loss_bbox'] = loss_bbox * stage_loss_weight

            # 7. 如果不是最后阶段，用当前回归结果更新proposals，用于下一阶段
            if stage < self.num_stages - 1:
                proposals_list = self._refine_bboxes(
                    sampling_results, bbox_preds, batch_data_samples[0]['img_shape'])

        return losses

    def _get_targets(self, sampling_results: List[SamplingResult],
                     batch_gt_instances: List[InstanceData]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取分类和回归目标"""
        cls_targets = []
        bbox_targets = []

        for res in sampling_results:
            # 分类目标
            cls_target = res.pos_gt_labels
            if cls_target is None:
                cls_target = torch.zeros(0, dtype=torch.long, device=res.priors.device)
            cls_targets.append(cls_target)

            # 回归目标
            if res.pos_gt_bboxes is not None and len(res.pos_gt_bboxes) > 0:
                bbox_target = self.bbox_coder.encode(res.priors, res.pos_gt_bboxes)
            else:
                bbox_target = torch.zeros(0, 4, device=res.priors.device)
            bbox_targets.append(bbox_target)

        cls_targets = torch.cat(cls_targets, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)

        return cls_targets, bbox_targets

    def _refine_bboxes(self, sampling_results: List[SamplingResult],
                       bbox_preds: torch.Tensor,
                       img_shape: Tuple[int, int]) -> List[torch.Tensor]:
        """用当前阶段的回归结果细化bbox，用于下一阶段"""
        refined_proposals = []
        start_idx = 0

        for res in sampling_results:
            num_samples = len(res.priors)
            if num_samples == 0:
                refined_proposals.append(res.priors)
                continue

            # 获取当前样本的预测
            sample_preds = bbox_preds[start_idx:start_idx + num_samples]
            start_idx += num_samples

            # 将偏移量应用到proposals上，得到更精细的bbox
            refined_bboxes = self.bbox_coder.decode(res.priors, sample_preds)

            # 裁剪bbox到图像范围内
            h, w = img_shape
            refined_bboxes[:, 0::2].clamp_(min=0, max=w)
            refined_bboxes[:, 1::2].clamp_(min=0, max=h)

            refined_proposals.append(refined_bboxes)

        return refined_proposals

    def predict(self, x: Tuple[torch.Tensor], rpn_results_list: List[InstanceData],
                batch_data_samples: List[dict]) -> List[InstanceData]:
        """多阶段预测"""
        proposals_list = [r.bboxes for r in rpn_results_list]
        img_shapes = [s.metainfo['img_shape'] for s in batch_data_samples]

        # 迭代所有阶段
        for stage in range(self.num_stages):
            # 提取ROI特征
            rois = bbox2roi(proposals_list)
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            bbox_feats = bbox_feats.flatten(1)

            # 分类和回归
            cls_scores = self.shared_cls_head(bbox_feats)
            bbox_preds = self.stage_reg_heads[stage](bbox_feats)

            # 如果不是最后阶段，更新proposals
            if stage < self.num_stages - 1:
                # 需要将预测的bboxes按样本拆分
                split_sizes = [len(p) for p in proposals_list]
                bbox_preds_split = torch.split(bbox_preds, split_sizes, dim=0)

                new_proposals_list = []
                for i, (proposals, preds) in enumerate(zip(proposals_list, bbox_preds_split)):
                    if len(proposals) == 0:
                        new_proposals_list.append(proposals)
                        continue

                    refined_bboxes = self.bbox_coder.decode(proposals, preds)
                    h, w = img_shapes[i]
                    refined_bboxes[:, 0::2].clamp_(min=0, max=w)
                    refined_bboxes[:, 1::2].clamp_(min=0, max=h)
                    new_proposals_list.append(refined_bboxes)

                proposals_list = new_proposals_list

        # 最后阶段的输出作为最终结果
        return self.bbox_head.predict_by_feat(
            rois, cls_scores, bbox_preds, batch_data_samples)

    def _forward(self, x: Tuple[torch.Tensor], rois: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播（用于简单测试）"""
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        bbox_feats = bbox_feats.flatten(1)

        # 使用最后阶段的回归头进行预测
        cls_scores = self.shared_cls_head(bbox_feats)
        bbox_preds = self.stage_reg_heads[-1](bbox_feats)

        return cls_scores, bbox_preds