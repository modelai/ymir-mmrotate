# ymir-mmrotate 镜像说明

- 支持任务类型： 训练， 推理， 挖掘

## 仓库/镜像地址

> 参考[open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate)

- 仓库：[modelai/ymir-mmrotate](https://github.com/modelai/ymir-mmrotate)

- docker hub镜像地址: `youdaoyzbx/ymir-executor:ymir2.1.0-mmrotate-cu113-tmi`

- 百度网盘镜像地址： TODO

## 性能表现

|  Backbone   | pretrain |  Aug  | mmAP  | mAP50 | mAP75 | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny |    IN    |  RR   | 47.37 | 75.36 | 50.64 |   4.88    |  20.45   |         4.40         |        [config](./rotated_rtmdet_tiny-3x-dota.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota_20221201_120814.json)                             |
| RTMDet-tiny |    IN    | MS+RR | 53.59 | 79.82 | 58.87 |   4.88    |  20.45   |         4.40         |      [config](./rotated_rtmdet_tiny-3x-dota_ms.py)       |                       [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota_ms/rotated_rtmdet_tiny-3x-dota_ms-f12286ff.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota_ms/rotated_rtmdet_tiny-3x-dota_ms_20221113_201235.log)                        |
|  RTMDet-s   |    IN    |  RR   | 48.16 | 76.93 | 50.59 |   8.86    |  37.62   |         4.86         |         [config](./rotated_rtmdet_s-3x-dota.py)          |                                   [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota-11f6ccf5.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota_20221124_081442.json)                                   |
|  RTMDet-s   |    IN    | MS+RR | 54.43 | 79.98 | 60.07 |   8.86    |  37.62   |         4.86         |        [config](./rotated_rtmdet_s-3x-dota_ms.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota_ms/rotated_rtmdet_s-3x-dota_ms-20ead048.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota_ms/rotated_rtmdet_s-3x-dota_ms_20221113_201055.json)                             |
|  RTMDet-m   |    IN    |  RR   | 50.56 | 78.24 | 54.47 |   24.67   |  99.76   |         7.82         |         [config](./rotated_rtmdet_m-3x-dota.py)          |                                   [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota-beeadda6.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota_20221122_011234.json)                                   |
|  RTMDet-m   |    IN    | MS+RR | 55.00 | 80.26 | 61.26 |   24.67   |  99.76   |         7.82         |        [config](./rotated_rtmdet_m-3x-dota_ms.py)        |                             [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms/rotated_rtmdet_m-3x-dota_ms-c71eb375.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms/rotated_rtmdet_m-3x-dota_ms_20221122_011234.json)                             |


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| :-: | :-: | :-: | :-: | :-: |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | :-: |
| model_name | yolov8_n | 字符串 | 模型简写, 如yolov7_tiny, yolov5_m, yolov6_t, rtmdet_m, ppyoloe_plus_s | 支持yolov5-v8, yolox, rtmdet, ppyoloe_plus |
| samples_per_gpu | 8 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | :-: |
| max_epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| args_options | '' | 字符串 | 训练命令行参数 | 参考 [ymir-mmrotate/tools/train.py](https://github.com/modelai/ymir-mmrotate/blob/ymir-1.x/tools/train.py) |
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [ymir-mmrotate/tools/train.py](https://github.com/modelai/ymir-mmrotate/blob/ymir-1.x/tools/train.py) |
| img_size | 512 | 整数 | 模型输入大小 | 采用32的倍数， 建议>=224, 一般取512, 640, 1024 |
| val_interval | 1 | 整数 | 模型在验证集上评测的周期， 以epoch为单位 | 设置为1，每个epoch可评测一次 |
| max_keep_checkpoints | 1 | 整数 | 最多保存的权重文件数量 | 设置为k, 可保存k个最优权重和k个最新的权重文件，设置为-1可保存所有权重文件。

> [cfg_options](https://ymir-executor-fork.readthedocs.io/zh/latest/algorithms/mmdet/#cfg_options)

## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| :-: | :-: | :-: | :-: | :-: |
| hyper-parameter | default value | type | note | advice |
| conf_threshold | 0.2 | 浮点数 | 推理结果置信度过滤阈值 | 设置为0可保存所有结果，设置为0.6可过滤大量结果 |
| iou_threshold | 0.65 | 浮点数 | 推理结果nms过滤阈值 | 设置为0.7可过滤大量结果，设置为0.5则过滤的结果较少 |

## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| :-: | :-: | :-: | :-: | :-: |
| hyper-parameter | default value | type | note | advice |
| mining_algorithm | entropy | 字符串 | 挖掘算法可选 entropy 和 random | 建议采用entropy |
| conf_threshold | 0.1 | 浮点数 | 推理结果置信度过滤阈值 | 设置为0可保存所有结果，设置为0.1可过滤一些推理结果，避免挖掘算法受低置信度结果影响 |
| iou_threshold | 0.65 | 浮点数 | 推理结果nms过滤阈值 | 设置为0.7可过滤大量结果，设置为0.5则过滤的结果较少 |

## FAQ

- [如何定制？](https://github.com/modelai/ymir-mmrotate/blob/ymir-1.x/ymir/readme.md)
