# A^2^S^2^C-DET:Dual-Path Adaptive Aggregation with Spatial-Semantic Compensation for Strip Steel Surface Defect Detection

## abstract

Identification of surface defects on steel strip, crucial for ensuring manufacturing quality and operational reliability, has advanced significantly with the rise of deep learning; however, accurate recognition remains challenging due to complex morphological variations, diverse defect scales, and irregular spatial distributions. To address these challenges, we propose A$^2$S$^2$C-Det, a novel detector that employs dual-path adaptive aggregation with spatial–semantic compensation, strengthening feature representation to achieve robust defect detection. First, we propose a plug-and-play semantic refinement bottleneck (SRB) that enhances backbone representations through multiscale perception and a feature-screening bottleneck, thereby capturing subtle and diverse defect shapes while accelerating model convergence. 
We further design a dual-path adaptive aggregation (DPAA) module that integrates complementary perspectives of cross-level semantic consistency and fine-grained structural cues through its coordinated dual paths, ensuring robust recognition across different defect sizes. Finally, we develop a spatial–semantic gated compensation (SSGC) module that adaptively supplements semantic information for low-level features and spatial details for high-level features, enabling precise localization of irregular defects. 

### NOTE

A^2^S^2^C-DET is implemented based on mmdetection3.0.

**All our implementation can be find in projects directory**

## Prepare the dataset

data
├── coco
│   ├── annotations
│   │      ├──instances_train2017.json
│   │      ├──instances_val2017.json
│   ├── train2017
│   ├── val2017

## train and test

The model A²S²C-DET is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection).

- **train (Take the NEU-DET dataset as an example.):**
  
  ```bash
  python train.py configs/MAFDet/mafdet_r50_mffe_daff_fpn_NEU-DET.py
  ```

- **test(Take the NEU-DET dataset as an example.):**
  
  ```bash
  python test.py configs/MAFDet/mafdet_r50_mffe_daff_fpn_NEU-DET.py path_to_checkpoint 
  ```

