# hw2_voc_mask_sparse_rcnn

## 概述
本项目使用 **Mask R-CNN** 和 **Sparse R-CNN** 两种模型在 PASCAL VOC 数据集上进行目标检测和实例分割。两种模型均使用 MMDetection 框架实现，并对结果进行了可视化与对比分析。

实验包含训练这两种模型、在一组图片上进行测试，并可视化它们的表现，重点分析 proposal 框和实例分割结果。


## 项目结构

以下是项目的目录结构：

- **`config.py`**: 基础配置文件，定义数据集路径和部分模型参数。
- **`mask-rcnn_r50_fpn_1x_voc.py`**: Mask R-CNN 的配置文件。
- **`sparse-rcnn_r50_fpn_1x_coco.py`**: Sparse R-CNN 的配置文件。
- **`test.py`**: 测试脚本，用于加载模型并可视化检测结果。
- **`extra_photos/`**: 测试图像目录，包含 `boat.jpg`、`bottle.jpg`、`cat.jpg` 和 `cow.jpg`。
- **`mask_rcnn_voc/`**: Mask R-CNN 相关文件和日志目录。
  - `best_coco_bbox_mAP_epoch_43.pth`: Mask R-CNN 最佳训练权重。
- **`sparse_rcnn_voc/`**: Sparse R-CNN 相关文件和日志目录。
  - `20250520_083511/best_coco_bbox_mAP_epoch_98.pth`: Sparse R-CNN 最佳训练权重。
- **`output/`**: 测试结果的输出目录。
- **`checkpoints/`**: 预训练模型权重保存目录。
- **`tensorboard_logs/`**: TensorBoard 日志文件。
- **`vis_data/`**: 可视化数据目录。

## 环境配置

### 1. 安装依赖

确保已安装 Python 3.7+，然后安装以下依赖：

```bash
# 安装 MMDetection 及其依赖
pip install -U openmim
mim install mmcv
mim install mmdet

# 安装其他依赖
pip install torch torchvision matplotlib numpy opencv-python
```

### 2. 硬件要求

- 推荐使用 GPU（支持 CUDA）以加速训练和测试。
- 若无 GPU，可使用 CPU 模式（`test.py` 中默认 `device='cpu'`）。

## 数据准备

1. **下载数据集**  
   下载 PASCAL VOC 2012 数据集，并将其放置在 `/home/mmdetection/data/coco/` 目录下。你也可以根据需要调整路径并在配置文件中修改 `data_root`。

2. **数据集结构**  
   确保数据目录结构如下：

   ```
   /home/mmdetection/data/coco/
   ├── annotations/
   │   ├── voc0712_train.json  # 训练集标注文件
   │   ├── voc0712_val.json    # 验证集标注文件
   ├── images/
   │   ├── train/              # 训练集图像
   │   ├── val/                # 验证集图像
   ```

## 训练

### Mask R-CNN 训练

1. **启动训练**  
   使用以下命令开始训练 Mask R-CNN：

   ```bash
   python tools/train.py mask-rcnn_r50_fpn_1x_voc.py
   ```

2. **训练参数**  
   - 配置文件：`mask-rcnn_r50_fpn_1x_voc.py`
   - 训练周期：50 个 epoch
   - 批量大小：8
   - 优化器：SGD（学习率=0.0002，动量=0.9，权重衰减=0.0001）
   - 学习率调度：MultiStepLR（里程碑=[8, 11]，衰减因子=0.1）
   - 输出目录：`work_dirs/mask_rcnn_voc/`（包含日志和检查点）

3. **监控训练**  
   使用 TensorBoard 查看训练进度：

   ```bash
   tensorboard --logdir work_dirs/mask_rcnn_voc/tensorboard_logs
   ```

### Sparse R-CNN 训练

1. **启动训练**  
   使用以下命令开始训练 Sparse R-CNN：

   ```bash
   python tools/train.py sparse-rcnn_r50_fpn_1x_coco.py
   ```

2. **训练参数**  
   - 配置文件：`sparse-rcnn_r50_fpn_1x_coco.py`
   - 训练周期：100 个 epoch
   - 批量大小：16
   - 优化器：AdamW（学习率=2.5e-05，权重衰减=0.0001）
   - 学习率调度：MultiStepLR（里程碑=[8, 11]，衰减因子=0.1）
   - 输出目录：`work_dirs/sparse_rcnn_voc/`（包含日志和检查点）

3. **监控训练**  
   使用 TensorBoard 查看训练进度：

   ```bash
   tensorboard --logdir work_dirs/sparse_rcnn_voc/tensorboard_logs
   ```

## 测试

### 1. 准备测试数据

将测试图像（如 `boat.jpg`、`bottle.jpg`、`cat.jpg`、`cow.jpg`）放置在 `/root/zrq/extra_photos/` 目录下。你可以根据需要修改 `test.py` 中的 `data_root` 路径。

### 2. 运行测试脚本

```bash
python test.py
```

- **功能**：  
  - 加载 Mask R-CNN 和 Sparse R-CNN 的预训练模型。
  - 对指定图像进行推理，生成检测结果。
  - 输出三类可视化图像：
    - `mask_rcnn_*.jpg`：Mask R-CNN 的 proposal 和最终预测结果。
    - `sparse_rcnn_*.jpg`：Sparse R-CNN 的预测结果。
    - `comparison_*.jpg`：Mask R-CNN 和 Sparse R-CNN 的对比结果。
- **配置文件和权重**：
  - Mask R-CNN：`mask-rcnn_r50_fpn_1x_voc.py`, `mask_rcnn_voc/best_coco_bbox_mAP_epoch_43.pth`
  - Sparse R-CNN：`sparse-rcnn_r50_fpn_1x_coco.py`, `sparse_rcnn_voc/20250520_083511/best_coco_bbox_mAP_epoch_98.pth`
- **输出目录**：`output/`（包含所有可视化结果）

### 3. 自定义测试

- 修改 `test.py` 中的 `image_paths` 列表以测试其他图像。
- 调整 `pred_score_thr`（默认 0.3）以更改检测置信度阈值。
- 将 `device` 改为 `'cuda:0'` 以使用 GPU 加速。
