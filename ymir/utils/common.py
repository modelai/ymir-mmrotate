"""
utils function for ymir and yolov5
"""
import difflib
import glob
import logging
import os
import os.path as osp
import warnings
from easydict import EasyDict as edict
from mmengine.config import Config, ConfigDict
from typing import Any, Dict, Iterable, List, Union
from ymir_exc.dataset_convert import convert_ymir_to_coco
from ymir_exc.util import get_bool

from ymir.utils.ymir_training_monitor_hook import YmirTrainingMonitorHook


def recursive_modify_attribute(mmengine_cfgdict: Union[Config, ConfigDict], attribute_key: str, attribute_value: Any):
    """
    recursive modify mmcv_cfg:
        1. mmcv_cfg.attribute_key to attribute_value
        2. mmcv_cfg.xxx.xxx.xxx.attribute_key to attribute_value (recursive)
        3. mmcv_cfg.xxx[i].attribute_key to attribute_value (i=0, 1, 2 ...)
        4. mmcv_cfg.xxx[i].xxx.xxx[j].attribute_key to attribute_value
    """
    for key in mmengine_cfgdict:
        if key == attribute_key:
            mmengine_cfgdict[key] = attribute_value
            logging.info(f'modify {mmengine_cfgdict}, {key} = {attribute_value}')
        elif isinstance(mmengine_cfgdict[key], (Config, ConfigDict)):
            recursive_modify_attribute(mmengine_cfgdict[key], attribute_key, attribute_value)
        elif isinstance(mmengine_cfgdict[key], Iterable):
            for cfg in mmengine_cfgdict[key]:
                if isinstance(cfg, (Config, ConfigDict)):
                    recursive_modify_attribute(cfg, attribute_key, attribute_value)


def get_mmengine_dataset_config(ymir_cfg: edict) -> Config:
    """get mmengine dataset config

    Parameters
    ----------
    ymir_cfg : edict
        ymir merged config

    Returns
    -------
    Config
        dataset config
    """
    # modify dataset config
    data_info = convert_ymir_to_coco(cat_id_from_zero=True)
    file_cfg = Config.fromfile('configs/_base_/datasets/ymir_coco.py')

    file_cfg.data_root = ymir_cfg.ymir.input.root_dir
    img_size = int(ymir_cfg.param.get('img_size', 512))
    samples_per_gpu = int(ymir_cfg.param.samples_per_gpu)
    workers_per_gpu = int(ymir_cfg.param.workers_per_gpu)
    file_cfg.train_dataloader.batch_size = samples_per_gpu
    file_cfg.train_dataloader.num_workers = workers_per_gpu

    metainfo = dict(classes=ymir_cfg.param.class_names)
    file_cfg.metainfo = metainfo
    for split in ['train', 'val', 'test']:
        for transform in file_cfg[f'{split}_pipeline']:
            if 'scale' in transform:
                assert transform.type == 'mmdet.Resize'
                transform.scale = (img_size, img_size)
            elif 'size' in transform:
                assert transform.type == 'mmdet.Pad'
                transform.size = (img_size, img_size)

        if split == 'test':
            continue

        ymir_dataset_cfg = dict(type='mmdet.CocoDataset',
                                ann_file=data_info[split]['ann_file'],
                                metainfo=metainfo,
                                data_root=ymir_cfg.ymir.input.root_dir,
                                data_prefix=dict(img='/in/assets'),
                                test_mode=split == 'val',
                                filter_cfg=dict(filter_empty_gt=False),
                                pipeline=file_cfg[f'{split}_pipeline'])

        file_cfg[f'{split}_dataloader'].dataset.update(ymir_dataset_cfg)

    file_cfg.test_dataloader = file_cfg.val_dataloader

    return file_cfg


def modify_mmengine_config(mmengine_cfg: Config, ymir_cfg: edict) -> None:
    """
    useful for training process
    - modify dataset config
    - modify model output channel
    - modify epochs, checkpoint, tensorboard config
    """

    # validation may augment the image and use more gpu
    # so set smaller samples_per_gpu for validation
    samples_per_gpu = int(ymir_cfg.param.samples_per_gpu)
    workers_per_gpu = int(ymir_cfg.param.workers_per_gpu)
    mmengine_cfg.train_batch_size_per_gpu = samples_per_gpu
    mmengine_cfg.train_num_workers = workers_per_gpu

    if 'batch_size_per_gpu' in mmengine_cfg.optim_wrapper.optimizer:
        mmengine_cfg.optim_wrapper.optimizer.batch_size_per_gpu = samples_per_gpu

    # modify model output channel
    num_classes = len(ymir_cfg.param.class_names)
    mmengine_cfg.num_classes = num_classes
    recursive_modify_attribute(mmengine_cfg.model, 'num_classes', num_classes)

    ymir_dataset_cfg = get_mmengine_dataset_config(ymir_cfg)
    # overwrite dataset config
    for split in ['train', 'val', 'test']:
        mmengine_cfg[f'{split}_pipeline'] = ymir_dataset_cfg[f'{split}_pipeline']
        mmengine_cfg[f'{split}_dataloader'] = ymir_dataset_cfg[f'{split}_dataloader']

        if split != 'train':
            mmengine_cfg[f'{split}_evaluator'] = ymir_dataset_cfg[f'{split}_evaluator']
    # update dataset_type, data_root and other
    mmengine_cfg.update(ymir_dataset_cfg)

    # modify max_epochs
    if ymir_cfg.param.get('max_epochs', None):
        max_epochs = int(ymir_cfg.param.max_epochs)
        mmengine_cfg.train_cfg.max_epochs = max_epochs
        mmengine_cfg.max_epochs = max_epochs

    # modify checkpoint
    mmengine_cfg.default_hooks.checkpoint['out_dir'] = ymir_cfg.ymir.output.models_dir
    mmengine_cfg.default_hooks.checkpoint['save_best'] = 'auto'

    # modify tensorboard
    tensorboard_logger = dict(type='TensorboardVisBackend', save_dir=ymir_cfg.ymir.output.tensorboard_dir)
    if len(mmengine_cfg.visualizer.vis_backends) <= 1:
        mmengine_cfg.visualizer.vis_backends.append(tensorboard_logger)
    else:
        mmengine_cfg.visualizer.vis_backends[1].update(tensorboard_logger)

    # TODO save only the best top-k model weight files.
    # modify evaluation and interval
    val_interval: int = int(ymir_cfg.param.get('val_interval', 1))
    if val_interval > 0:
        val_interval = min(val_interval, mmengine_cfg.train_cfg.max_epochs)
    else:
        val_interval = 1

    mmengine_cfg.save_epoch_intervals = val_interval
    mmengine_cfg.train_cfg.val_interval = val_interval

    # save best top-k model weights files
    # max_keep_ckpts <= 0  # save all checkpoints
    max_keep_ckpts: int = int(ymir_cfg.param.get('max_keep_checkpoints', 1))
    mmengine_cfg.default_hooks.checkpoint.interval = val_interval
    mmengine_cfg.default_hooks.checkpoint.max_keep_ckpts = max_keep_ckpts

    # fix DDP error or make training faster?
    mmengine_cfg.find_unused_parameters = get_bool(ymir_cfg, 'find_unused_parameters', False)

    # learning rate
    if ymir_cfg.param.get('learning_rate', None):
        mmengine_cfg.base_lr = float(ymir_cfg.param.learning_rate)
        mmengine_cfg.optim_wrapper.optimizer.lr = float(ymir_cfg.param.learning_rate)

    # set training log interval (iter)
    with open(ymir_cfg.ymir.input.training_index_file, 'r') as fp:
        train_dataset_size = len(fp.readlines())
    gpu_id: str = str(ymir_cfg.param.get("gpu_id", ''))
    num_gpus = len(gpu_id.split(","))
    max_interval = train_dataset_size // (samples_per_gpu * num_gpus)

    log_interval = max(1, min(50, max_interval))
    mmengine_cfg.default_hooks.logger.interval = log_interval

    # add YmirTrainingMonitorHook
    # HOOKS.register_module(module=YmirTrainingMonitorHook)
    ymir_hook = dict(type='YmirTrainingMonitorHook', interval=log_interval)
    mmengine_cfg.custom_hooks.append(ymir_hook)

    # set work dir
    mmengine_cfg.work_dir = ymir_cfg.ymir.output.models_dir

    args_options = ymir_cfg.param.get("args_options", '')
    cfg_options = ymir_cfg.param.get("cfg_options", '')

    # auto load offered weight file if not set by user!
    if (args_options.find('--resume-from') == -1 and args_options.find('--load-from') == -1
            and cfg_options.find('load_from') == -1 and cfg_options.find('resume_from') == -1):  # noqa: E129

        weight_file = get_best_weight_file(ymir_cfg)
        if weight_file:
            if cfg_options:
                cfg_options += f' load_from={weight_file}'
            else:
                cfg_options = f'load_from={weight_file}'
        else:
            logging.warning('no weight file used for training!')


def get_best_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.pretrained_model_params or cfg.param.model_params_path
    load coco-pretrained weight for yolox
    """
    if cfg.ymir.run_training:
        model_params_path: List[str] = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path = cfg.param.get('model_params_path', [])

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path
        if osp.exists(osp.join(model_dir, p)) and p.endswith(('.pth', '.pt'))
    ]

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    epoch_pth_files = [f for f in model_params_path if osp.basename(f).startswith(('epoch_', 'iter_'))]
    best_pth_files = [f for f in model_params_path if f not in epoch_pth_files]

    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    if cfg.ymir.run_training:
        weight_files = [f for f in glob.glob('/weights/**/*', recursive=True) if f.endswith(('.pth', '.pt'))]

        # load pretrained model weight for target model
        config_file = get_config_file(cfg)
        if len(weight_files) > 0:
            # use basename to avoid directory name change match result
            matched_weight_files = difflib.get_close_matches(osp.basename(config_file), weight_files)
            if len(matched_weight_files) > 0:
                logging.info(f'load yolox pretrained weight {matched_weight_files[0]}')
                return matched_weight_files[0]
    return ""


def get_topk_checkpoints(files: List[str], k: int) -> List[str]:
    """
    keep topk checkpoint files, remove other files.

    1. keep topk best checkpoint for ensembel
    2. keep topk latest checkpoint for quantization
    """
    checkpoints_files = [f for f in files if f.endswith(('.pth', '.pt'))]

    epoch_pth_files = [f for f in checkpoints_files if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        topk_epoch_pth_files = sorted(epoch_pth_files, key=os.path.getctime, reverse=True)
    else:
        topk_epoch_pth_files = []

    # best weight files may not name with best, use the other instead
    best_pth_files = [f for f in checkpoints_files if f not in epoch_pth_files]
    if len(best_pth_files) > 0:
        # newest first
        topk_best_pth_files = sorted(best_pth_files, key=os.path.getctime, reverse=True)
    else:
        topk_best_pth_files = []

    # python will check the length of list
    if k < 0:
        return checkpoints_files
    else:
        return topk_best_pth_files[0:k] + topk_epoch_pth_files[0:k]


def get_id_for_config_files() -> dict:
    """
    use id instead of config_file:
    rtmdet_tiny: configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms.py

    note: '-' will be replace with '_'
    """

    py_files = glob.glob(osp.join('configs', '*', '*dota.py'))

    config_files = [f for f in py_files if f.split('/')[1] not in ['_base_', 'deploy']]

    id_dict: Dict[str, str] = {}
    for f in config_files:
        f_name = osp.basename(f).replace('-', '_')
        splits = f_name.split('_')

        # remove rotated keywords
        if splits[0] == 'rotated':
            splits = splits[1:]

        # retinanet, fcos, rtmdet, ...
        for x in range(2, len(splits)):
            idx = '_'.join(splits[0:x])
            id_dict[idx] = f

        id_dict[f] = f

    return id_dict


def get_config_file(cfg: edict) -> str:
    """get config file path from ymir config

    for training task, get config file from hyper-parameter model_name
    for mining and infer task, get config file from weight files

    Parameters
    ----------
    cfg : easydict.EasyDict
        ymir merged config

    Returns
    -------
    str
        config file path for mmengine model

    Raises
    ------
    Exception
        KeyError('config_id not in dicts')
        Exception('no config_file found')
    """
    if cfg.ymir.run_training:
        # for training task, get config file from hyper-parameter model_name
        model_name = cfg.param.get("model_name")
        config_files_map = get_id_for_config_files()
        config_id = model_name.lower().replace('-', '_')
        if config_id not in config_files_map:
            raise KeyError(f'{config_id} not in {config_files_map}')

        config_file = config_files_map[config_id]
        return config_file
    else:
        # for mining and infer task, get config file from weight files
        model_params_path: List = cfg.param.get('model_params_path', [])  # type: ignore

        model_dir = cfg.ymir.input.models_dir
        config_files = [
            osp.join(model_dir, p) for p in model_params_path
            if osp.exists(osp.join(model_dir, p)) and p.endswith(('.py'))
        ]

        if len(config_files) > 0:
            if len(config_files) > 1:
                warnings.warn(f'multiple config file found! use {config_files[0]}')
            return config_files[0]
        else:
            raise Exception(f'no config_file found in {model_dir} and {model_params_path}')
