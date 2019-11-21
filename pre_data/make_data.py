import os
import shutil
import cv2
import numpy as np
import json


def horizontal_flip(image, axis):
    # 以50%的可能性翻转图片，axis 0 垂直翻转，1水平翻转
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def random_crop(rgb_img, crop_size):
    height, width, _ = rgb_img.shape
    w_range = (width - crop_size[0]) // 2 if width > crop_size[0] else 0
    h_range = (height - crop_size[1]) // 2 if height > crop_size[1] else 0

    w_offset = 0 if w_range == 0 else np.random.randint(w_range)
    h_offset = 0 if h_range == 0 else np.random.randint(h_range)

    cropped_rgb = rgb_img[h_offset:h_offset + crop_size[0], w_offset:w_offset + crop_size[1], :]
    cropped_rgb = horizontal_flip(cropped_rgb, 1)

    return cropped_rgb


with open("train_list.txt", "r+") as f:
    lines = f.readlines()
label_image = {}
for line in lines:
    name, label = line.split(" ")
    label = label[:-1]
    if label in label_image.keys():
        label_image[label].append(name)
    else:
        label_image[label] = [name]

with open('data.json', 'w') as f:
    json.dump(label_image, f)

numbers = dict()

ori_dir = '/home/zxh/datasets/NAICReID/bounding_box_train'
mid_dir = 'mid_train'

###############
#
# for k, v in label_image.items():
#     print("k", k)
#     if len(v) > 1:
#         for vv in v:
#             name = '{}_c1s1_{}_00.png'.format(str(k).zfill(4), vv.split('/')[1].split('.')[0])
#             img_path = os.path.join(ori_dir, name)
#             shutil.copy(img_path, mid_dir)
    # else:
    #     ind = 0
    #     while ind <= 10:
    #         for i, vv in enumerate(v):
    #             name = '{}_c1s1_{}_00.png'.format(str(k).zfill(4), vv.split('/')[1].split('.')[0])
    #             img_path = os.path.join(ori_dir, name)
    #             if ind > len(v):
    #                 ori_img = cv2.resize(cv2.imread(img_path), (256, 256))
    #                 crop_img = cv2.resize(random_crop(ori_img, (256, 128)), (128, 256))
    #                 new_name = '{}_c1s1_{}{}_00.png'.format(str(k).zfill(4), 'new'+str(ind), vv.split('/')[1].split('.')[0])
    #                 cv2.imwrite(os.path.join(mid_dir, new_name), crop_img)
    #             else:
    #                 shutil.copy(img_path, mid_dir)
    #             ind += 1
##############

out_list = []
for k, v in label_image.items():
    if len(v) > 4 and len(v)<=10:
        out_list.append(k)

with open("number_more4less10.txt", 'w') as f:
    f.write(str(out_list))

with open("number_more4less10.txt", 'r') as f:
    line = eval(f.readline().strip())
print(len(line))
# max = 0
# id = 0
# for k, v in label_image.items():
#     a = [767, 383, 1374, 1350, 1055, 1273, 1477, 1147, 514, 760, 651, 112, 174, 1161, 968]
#     if int(k) in a:
#         print(k)
#         if len(v) > max:
#             max = len(v)
#             id = k
# print(id)
#



