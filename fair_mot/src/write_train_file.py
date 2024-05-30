# 编写.train, .val文件等
from email.mime import image
import os
from unittest.mock import patch


def mot17():
    image_path = '/data/resources/mot17-detection/images'
    train_file = '/data/JDE/data/mot17.train'
    with open(train_file, 'w') as f:
        img_list = os.listdir(image_path)
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(image_path, img_name)
            f.write(str(img_path) + '\n')


def gta_attr_detection():
    image_path = '/data/resources/gta-attr-dataset-02/images'
    train_file = '/data/JDE/data/gta-attr-detection-02.train'
    with open(train_file, 'w') as f:
        img_list = os.listdir(image_path)
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(image_path, img_name)
            f.write(str(img_path) + '\n')


def gta_detection():
    image_path = '/data/resources/gta-detection-dataset/images'
    train_file = '/data/JDE/data/gta-detection.train'
    with open(train_file, 'w') as f:
        img_list = os.listdir(image_path)
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(image_path, img_name)
            f.write(str(img_path) + '\n')


def citypersons():
    image_root = '/data/resources/Citypersons/images/train'
    train_file = '/data/JDE/data/citypersons.train'
    scenes = os.listdir(image_root)
    scenes.sort()
    with open(train_file, 'w') as f:
        for scene in scenes:
            scene_root = os.path.join(image_root, scene)
            img_list = os.listdir(scene_root)
            img_list.sort()
            for img_name in img_list:
                img_path = os.path.join(scene_root, img_name)
                f.write(str(img_path) + '\n')


def caltech():
    image_path = '/data/resources/Caltech/images'
    train_file = '/data/JDE/data/caltech.train'
    with open(train_file, 'w') as f:
        img_list = os.listdir(image_path)
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(image_path, img_name)
            f.write(str(img_path) + '\n')


def caltech_10k_val():
    train_file_path = '/data/JDE/data/caltech-10k-val.train'
    src_file = '/data/JDE/data/caltech.10k.val'
    path = '/data/resources/Caltech/images'
    train_file = open(train_file_path, 'w')
    with open(src_file, mode='r') as src_file:
        for line in src_file:
            idx = line.rindex('/')
            img_path = path + line[idx:]
            train_file.write(img_path)
    train_file.close()       


if __name__ == '__main__':
    caltech_10k_val()
