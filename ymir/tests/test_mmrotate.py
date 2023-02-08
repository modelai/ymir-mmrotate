from mmdet.apis import inference_detector, init_detector

from mmrotate.utils import register_all_modules

if __name__ == '__main__':
    register_all_modules(init_default_scope=True)
    config_file = 'configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'
    checkpoint_file = 'ymir/weights/rotated_rtmdet_tiny-3x-dota-9d821076.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    inference_detector(model, 'demo/demo.jpg')
