# 从视频序列中生成检测数据集
from operator import le
import os
import shutil
import tqdm
import numpy as np
import PIL.Image as Image
from alignment import model_dict


id_strs = []


def get_id(features):
    id_str = str(features[2]) + '-'
    id_str = id_str + str(features[-21])
    id_str = id_str + str(features[-15]) + str(features[-14]) + str(features[-13]) + str(features[-12])
    if id_str not in id_strs:
        id_strs.append(id_str)
    return str(id_strs.index(id_str))


attributes = ['Female', 'Body Fat', 'Body Medium', 'Body Thin', 'Hair Bold', 'Hair Short', 'Hair Long',
              'Short Sleeve', 'Skirt', 'Lower Short', 'Backpack', 'Hat', 'Boots', 'Upper Long',
              'Upper White', 'Upper Black', 'Upper Gray', 'Upper Purple', 'Upper Red', 'Upper Yellow', 'Upper Blue',
              'Upper Green', 'Upper Brown',
              'Lower White', 'Lower Black', 'Lower Gray', 'Lower Purple', 'Lower Red', 'Lower Yellow', 'Lower Blue',
              'Lower Green', 'Lower Brown']


def get_feature_str(features):
    ped_hash = int(features[2])
    ped_hash = f'0x{ped_hash:X}'
    
    features = [0] * 32
    face_key = str(features[-21]) + str(features[-20])
    hair_key = str(features[-17]) + str(features[-16])
    upper_key = str(features[-15]) + str(features[-14])
    lower_key = str(features[-13]) + str(features[-12])
    accs_key = str(features[-7]) + str(features[-6])

    if ped_hash in model_dict:
        ped_dict = model_dict.get(ped_hash)
    else:
        return ''
    feature = ped_dict.get('model')

    for f in feature:
        if f not in attributes:
            continue
        features[attributes.index(f)] = 1
    if 'face' in ped_dict:
        face_dict = ped_dict.get('face')
        if face_key in face_dict:
            feature = face_dict.get(face_key)
            for f in feature:
                if f not in attributes:
                    continue
                features[attributes.index(f)] = 1
        else:
            return ''
    if 'hair' in ped_dict:
        hair_dict = ped_dict.get('hair')
        if hair_key in hair_dict:
            feature = hair_dict.get(hair_key)
            for f in feature:
                if f not in attributes:
                    continue
                features[attributes.index(f)] = 1
        else:
            return ''
    upper_dict = ped_dict.get('upper')
    if upper_key in upper_dict:
        feature = upper_dict.get(upper_key)
        for f in feature:
            if f not in attributes:
                continue
            features[attributes.index(f)] = 1
    else:
        return ''
    lower_dict = ped_dict.get('lower')
    if lower_key in lower_dict:
        feature = lower_dict.get(lower_key)
        for f in feature:
            if f not in attributes:
                continue
            features[attributes.index(f)] = 1
    else:
        return ''
    if 'accs' in ped_dict:
        accs_dict = ped_dict.get('accs')
        if accs_key in accs_dict:
            feature = accs_dict.get(accs_key)
            for f in feature:
                if f not in attributes:
                    continue
                features[attributes.index(f)] = 1
    feature_str = ' '.join(str(f) for f in features)
    return feature_str
    


def generate(use_attr):
    det_path = "./detection-attr-dataset"
    data_path = "./450-seq-dataset"
    img_num = 1

    img_path = det_path + "images"
    label_path = det_path + "labels_with_ids"
    attr_path = det_path + 'attr_labels'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    if use_attr and not os.path.exists(attr_path):
        os.mkdir(attr_path)
    seq_dirs = os.listdir(data_path)
    for seq_dir in seq_dirs:
        seq_path = data_path + seq_dir
        data = np.genfromtxt(seq_path + "/" + "det.txt", dtype=[int, int, float, float, float, float,
                                                                int, int, int], delimiter=",")
        feature_data = np.loadtxt(os.path.join(seq_path, 'feature.txt'), delimiter = ",")
        seq_idx = int(seq_dir[-3:])
        if seq_idx < 110:
            feature_data = feature_data.reshape((-1, 30))
        else:
            feature_data = feature_data.reshape((-1, 25))
        for i in tqdm.tqdm(range(0, 1805, 5), desc="generating from %s" % seq_dir):
            img_dst = os.path.join(img_path, str(img_num).zfill(7) + '.jpg')
            label_dst = os.path.join(label_path, str(img_num).zfill(7) + '.txt')
            attr_dst = os.path.join(attr_path, str(img_num).zfill(7) + '.txt')

            img_src = os.path.join(seq_path, str(i) + '.jpeg')
            img = Image.open(img_src)

            file = open(label_dst, mode='w')
            if use_attr:
                attr_file = open(attr_dst, mode='w')
            use_frame = True
            for idx in range(len(data)):
                if data[idx][0] < i + 1:
                    continue
                elif data[idx][0] == i + 1:
                    x = data[idx][2]
                    y = data[idx][3]
                    w = data[idx][4]
                    h = data[idx][5]
                    x = x + w / 2
                    y = y + h / 2
                    x = x / img.width
                    y = y / img.height
                    w = w / img.width
                    h = h / img.height
                    if x < 0 or x > 1 or y < 0 or y > 1:
                        continue
                    if use_attr:
                        feature_str = get_feature_str((feature_data[idx]))
                        if feature_str == '':
                            use_frame = False
                            break
                        attr_file.write(feature_str + '\n')
                    id_str = get_id(feature_data[idx])
                    file.write("0 " + id_str + ' ')
                    file.write('{:.6f}'.format(x) + " " + '{:.6f}'.format(y) + " " +
                               '{:.6f}'.format(w) + " " + '{:.6f}'.format(h) + "\n")
                else:
                    break
            if not use_frame:
                continue
            file.close()
            if use_attr:
                attr_file.close()
            dImg = img.resize((1920, 1080), Image.ANTIALIAS)
            dImg.save(img_dst)
            img_num += 1
    # 生成一下train.txt和val.txt
    train_txt = open(os.path.join(det_path, 'gta-detection.train'), mode='w')
    for i in range(1, img_num):
        train_txt.write('./detection-attr-dataset/images/' + str(i).zfill(7) + '.jpg' + '\n')
    train_txt.close()


if __name__ == '__main__':
    generate(True)
