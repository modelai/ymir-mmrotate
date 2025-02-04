import glob
import logging
import os
import os.path as osp
import re
import subprocess
import sys
from easydict import EasyDict as edict
from mmengine.config import Config
from ymir_exc.util import (YmirStage, find_free_port, get_merged_config,
                           write_ymir_monitor_process,
                           write_ymir_training_result)

from ymir.utils.common import (get_best_weight_file, get_config_file,
                               get_topk_checkpoints)
from ymir.utils.dataset_convert import convert_ymir_to_coco


def _parse_log_line(line):
    """
    # normal evaluation line
    02/07 11:05:25 - mmengine - INFO - Epoch(val) [1][232/232]  dota/mAP: 0.0000  dota/AP50: 0.0000
    """
    epoch = int(re.findall(r'\[\d+\]', line)[0][1:-1])
    maps = [float(x) for x in re.findall(r'\d+\.\d+', line)]

    N = len('dota/')
    pattern = r'dota/\w*:'

    keys = [x[N:-1] for x in re.findall(pattern, line)]
    info = {key: map for key, map in zip(keys, maps)}
    return epoch, info


def write_mmyolo_training_result(cfg: edict) -> None:
    """
    save the best checkpoint for ymir after training.
    keep the same id with YmirTrainingMonitorHook, make sure the saved checkpoint exist.
    """
    out_dir = cfg.ymir.output.models_dir
    log_files = glob.glob(osp.join(out_dir, '*', '*.log'))

    assert len(log_files) > 0
    log_file = max(log_files, key=osp.getctime)

    # only one log file
    with open(log_file, 'r') as fp:
        lines = fp.readlines()

    log_info_dict = {}
    for line in lines:
        if line.find('dota/mAP:') > -1:
            epoch, info = _parse_log_line(line)
            log_info_dict[epoch] = info

    # for the best files
    cfg_files = glob.glob(osp.join(out_dir, '*.py'))
    best_ckpts = glob.glob(osp.join(out_dir, 'best_dota', '*.pth'))

    topk = cfg.param.max_keep_checkpoints
    # skip the newest ckpt, note YmirTrainingMonitorHook will save newest ckpt.
    topk_best_ckpts = get_topk_checkpoints(best_ckpts, topk)[1:]

    mmengine_cfg = Config.fromfile(cfg_files[0])
    evaluate_config = dict(iou_thr=mmengine_cfg.model.test_cfg.nms.iou_threshold,
                           conf_thr=mmengine_cfg.model.test_cfg.score_thr)

    for ckpt in topk_best_ckpts:
        epoch = int(re.findall(r'\d+', ckpt)[0])
        if epoch not in log_info_dict:
            continue
        write_ymir_training_result(cfg,
                                   files=[ckpt] + cfg_files,
                                   id=f'best_{epoch}',
                                   evaluation_result=log_info_dict[epoch],
                                   evaluate_config=evaluate_config)

    # save the last ckpt only, note YmirTrainingMonitorHook will save the last ckpt too.
    last_ckpt = max(glob.glob(osp.join(out_dir, 'epoch_*.pth')), key=osp.getctime)
    last_epoch = int(re.findall(r'\d+', last_ckpt)[0])
    if last_epoch in log_info_dict:
        write_ymir_training_result(cfg,
                                   files=[last_ckpt] + cfg_files,
                                   id='last',
                                   evaluation_result=log_info_dict[last_epoch],
                                   evaluate_config=evaluate_config)


def main(cfg: edict) -> int:
    # default ymir config
    gpu_id: str = str(cfg.param.get("gpu_id", ''))
    num_gpus = len(gpu_id.split(","))

    classes = cfg.param.class_names
    num_classes = len(classes)
    logging.info(f'num_classes = {num_classes}')

    # convert dataset before ddp
    data_info = convert_ymir_to_coco(cat_id_from_zero=True)
    logging.info(f'convert dataset to {data_info}')

    # mmcv args config
    config_file = get_config_file(cfg)
    # config_file = cfg.param.get("config_file")
    args_options = cfg.param.get("args_options", '')
    cfg_options = cfg.param.get("cfg_options", '')

    # auto load offered weight file if not set by user!
    if (args_options.find('--resume-from') == -1) and ((cfg_options.find('load_from') == -1
                                                        and cfg_options.find('resume_from') == -1)):

        weight_file = get_best_weight_file(cfg)
        if weight_file:
            if cfg_options:
                cfg_options += f' load_from={weight_file}'
            else:
                cfg_options = f'load_from={weight_file}'
        else:
            logging.warning('no weight file used for training!')

    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=0.2, stage=YmirStage.POSTPROCESS)

    work_dir = cfg.ymir.output.models_dir
    if num_gpus == 0:
        # view https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#training-on-cpu
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', "-1")
        cmd = f"python3 tools/train.py {config_file} " + \
            f"--work-dir {work_dir}"
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        port = find_free_port()
        os.environ.setdefault('PORT', str(port))
        cmd = f"bash ./tools/dist_train.sh {config_file} {num_gpus} " + \
            f"--work-dir {work_dir}"

    if args_options:
        cmd += f" {args_options}"

    if cfg_options:
        cmd += f" --cfg-options {cfg_options}"

    logging.info(f"training command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    # save the last checkpoint
    write_mmyolo_training_result(cfg)
    return 0


if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(main(cfg))
