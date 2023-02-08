# 开发文档

## 训练

代码变化可参考[pull/2](https://github.com/modelai/ymir-mmrotate/pull/2/files)

### 训练脚本调用链

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_training.py`

4. `ymir_training.py` 调用 `bash tools/dist_train.sh ...`

### 核心功能实现

- 数据格式转换

在 `ymir_training.py` 中首次调用 `convert_ymir_to_coco()` 进行数据格式转换，将 `det-ark:raw` 对应的 `xmin, ymin, xmax, ymax, image_quality, rotate_angle` 转换为 `qbox` 存到coco标注中的 `segmentation` 字段中。 后续调用 `convert_ymir_to_coco()` 仅获得数据集信息。

```python
for ann_line in open(ann_file, "r").readlines():
    ann_strlist = ann_line.strip().split(",")
    class_id, x1, y1, x2, y2 = [int(s) for s in ann_strlist[0:5]]
    angle = float(ann_strlist[6])

    bbox_xc = (x1 + x2) / 2
    bbox_yc = (y1 + y2) / 2
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    rbox = torch.from_numpy(np.array([[bbox_xc, bbox_yc, bbox_width, bbox_height, angle]]))
    qbox = rbox2qbox(rbox)
    # hbox = rbox2hbox(rbox)

    bbox_area = bbox_width * bbox_height
    bbox_quality = (float(ann_strlist[5]) if len(ann_strlist) > 5 and ann_strlist[5].isnumeric() else 1)
    ann_info = dict(
        bbox=[x1, y1, bbox_width, bbox_height],  # x,y,width,height
        area=bbox_area,
        score=1.0,
        bbox_quality=bbox_quality,
        rotate_angle=angle,
        iscrowd=0,
        segmentation=qbox.cpu().numpy().tolist(),
        category_id=class_id + cat_id_start,  # start from cat_id_start
        id=ann_id,
        image_id=img_id,
    )
    data["annotations"].append(ann_info)
    ann_id += 1
```

- 加载数据集

    - 参考 `configs/_base_/datasets/ymir_coco.py`， 基于 **mmdet.CocoDataset** 加载转换后的数据集， 通过 **train_pipeline**, **val_pipeline** 中的 `dict(type='ConvertMask2BoxType', box_type='rbox')` 将 segmentation 字段中的 **mask** 转换为训练要使用的 `rbox` 格式。

- 数据集评测

    - 采用常用数据集**dota**的评测指标 **dict(type='DOTAMetric', metric='mAP')**

- 加载预训练权重

    - 参考 `get_best_weight_file()`

    - 如果用户提供预训练权重， 则先其中找带 `best_` 或 `mAP_` 的权重，其次找带 `epoch_` 的权重， 最后选择其中最新的。

    - 如果用户没有提供预训练权重，则在镜像的 `/weights` 目录下， 通过超参数 `model_name` 获得 `config_file` 再通过相似度找到最相似的权重文件。

- 加载超参数

    - 参考 `modify_mmengine_config()`, 将ymir超参数覆盖 `mmengine.config.Config`

- 写进度

    - 参考 `YmirTrainingMonitorHook`, 该 hook 可实时返回进度信息， 并保存最新的权重文件到ymir中，以支持提前终止训练的功能。

- 写结果文件

    - 参考 `YmirTrainingMonitorHook` 与 `write_mmyolo_training_result()`, 其中后者支持依据超参数 `max_keep_checkpoints` 保存多个权重文件。

## 推理

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_infer.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `mmdet_result_to_ymir()` 将mmdet推理结果转换为ymir格式

    - 调用 `rw.write_infer_result()` 保存推理结果

## 挖掘

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_mining.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `compute_score()` 计算挖掘分数

    - 调用 `rw.write_mining_result()` 保存挖掘结果


## FAQ

- 将pipeline最后的 `box_type` 改为 `rbox`
```
AssertionError: The boxes dimension must >= 2 and the length of the last dimension must be 5, but got boxes with shape torch.Size([1, 4]).
```
