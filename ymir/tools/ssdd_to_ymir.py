"""
convert SAR ship detection dataset (SSDD) dataset to ymir detection import format

view https://github.com/open-mmlab/mmrotate/blob/main/tools/data/ssdd/README.md for detail

SSDD
├── test
│   ├── all
│   │   ├── images
│   │   ├── labelTxt
│   │   └── test.json
│   ├── inshore
│   │   ├── images
│   │   └── labelTxt
│   └── offshore
│       ├── images
│       └── labelTxt
└── train
    ├── images
    ├── labelTxt
    └── train.json

YMIR-SSDD
└── train
    ├── gt
    └── images
└── val
    ├── gt
    └── images
"""

import argparse
import cv2
import json
import numpy as np
import os
import os.path as osp
import random
import shutil
from typing import List


def qbox2rbox(qbox: List[float]) -> List[float]:
    """_summary_

    Parameters
    ----------
    qbox : _type_
        _description_

    view `from mmrotate.structures import qbox2rbox`
    """
    assert len(qbox) == 8, f'qbox = {qbox}'
    pts = np.array(qbox, dtype=np.float32).reshape(4, 2)
    (x, y), (w, h), angle = cv2.minAreaRect(pts)
    return [x, y, w, h, angle / 180 * np.pi]


def get_args():
    parser = argparse.ArgumentParser("convert SAR ship detection dataset (SSDD) to ymir")
    parser.add_argument("--root_dir", help="root dir for SSDD dataset")
    parser.add_argument("--split", choices=['train', 'test'], default="train", help="split for dataset")
    parser.add_argument("--out_dir", help="the output directory", default="./out")
    parser.add_argument("--num", help="sample number for dataset", default=0, type=int)

    return parser.parse_args()


def coco_to_xml(img, anns, xml_path):
    with open(xml_path, 'w') as fp:
        fp.write(f"""<annotation>
	<folder>images</folder>
	<filename>{img['file_name']}</filename>
	<source>
		<database>SAR ship detection dataset (SSDD)</database>
		<annotation>PASCAL VOC2007</annotation>
	</source>
	<size>
		<width>{img['width']}</width>
		<height>{img['height']}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>""")

        for ann in anns:
            qbox = ann['segmentation'][0]
            assert len(ann['segmentation']) == 1, f'ann = {ann}'
            xc, yc, w, h, angle = qbox2rbox(qbox)
            xmin = round(xc - w / 2)
            xmax = round(xc + w / 2)
            ymin = round(yc - h / 2)
            ymax = round(yc + h / 2)

            fp.write(f"""
	<object>
		<name>SAR ship</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
            <rotate_angle>{angle}</rotate_angle>
		</bndbox>
	</object>""")

        fp.write("\n</annotation>")


if __name__ == '__main__':
    args = get_args()

    if args.split == 'train':
        json_file = osp.join(args.root_dir, args.split, 'train.json')
    else:
        json_file = osp.join(args.root_dir, args.split, 'all', 'test.json')

    with open(json_file, 'r') as fp:
        data = json.load(fp)

    if args.num > 0:
        num = min(len(data['images']), args.num)
        images = random.choices(data['images'], k=num)
    else:
        images = data['images']

    out_dir = osp.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'gt'), exist_ok=True)

    for img in images:
        if args.split == 'train':
            img_path = osp.join(args.root_dir, args.split, 'images', img['file_name'])
        else:
            img_path = osp.join(args.root_dir, args.split, 'all', 'images', img['file_name'])

        if not osp.exists(img_path):
            img_path = osp.splitext(img_path)[0] + '.png'
        assert osp.exists(img_path), f'{img_path} not exist'

        des_path = osp.join(out_dir, 'images', img['file_name'])
        assert not osp.exists(des_path)
        shutil.copy(img_path, des_path)

        anns = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]

        xml_path = osp.join(out_dir, 'gt', osp.splitext(img['file_name'])[0] + '.xml')
        assert not osp.exists(xml_path)
        coco_to_xml(img, anns, xml_path)
