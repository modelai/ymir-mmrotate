# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = '/in'
file_client_args = dict(backend='disk')

# TODO for dark image, pad_val = 0; for other image, pad_val = 114
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='rbox'),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='rbox'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'instances'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

metainfo = dict(classes=('ship', ))

train_dataloader = dict(batch_size=2,
                        num_workers=2,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_sampler=None,
                        dataset=dict(type=dataset_type,
                                     metainfo=metainfo,
                                     data_root=data_root,
                                     ann_file='/out/ymir_dataset/ymir_train.json',
                                     data_prefix=dict(img='/in/assets'),
                                     filter_cfg=dict(filter_empty_gt=True),
                                     pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   metainfo=metainfo,
                                   data_root=data_root,
                                   ann_file='/out/ymir_dataset/ymir_val.json',
                                   data_prefix=dict(img='/in/assets'),
                                   test_mode=True,
                                   pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
# val_evaluator = dict(type='RotatedCocoMetric', metric='bbox')

test_evaluator = val_evaluator
