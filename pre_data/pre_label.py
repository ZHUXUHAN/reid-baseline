import shutil
import os


def pre_train_label():
    img_root = '/home/zxh/datasets/bounding_box_train'
    copy_to_root = '/home/zxh/datasets/copy_train'

    with open("/home/zxh/train_list.txt") as f:
        lines = f.readlines()
    max = 0
    for line in lines:
        line = line.strip().split(' ')
        imgname = line[0].split('/')[1]
        imgid = line[1]
        newimgname = '{}_c1s1_{}_00.png'.format(imgid, imgname.split('.')[0])
        print(newimgname)

        shutil.copy(os.path.join(img_root, imgname), os.path.join(copy_to_root, newimgname))


def pre_test_label():
    img_root = '/home/zxh/datasets/bounding_box_test'
    copy_to_root = '/home/zxh/datasets/copy_test'
    files = os.listdir(img_root)
    for file in files:
        newimgname = '{}_c1s1_{}_00.png'.format('0000', file.split('.')[0])
        print(newimgname)
        shutil.copy(os.path.join(img_root, file), os.path.join(copy_to_root, newimgname))

pre_test_label()
