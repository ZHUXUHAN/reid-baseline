import shutil
import os

img_root = '/home/zxh/database/2019-NAIC-ReID/初赛A榜测试集/query_a'
copy_to_root = '/home/zxh/ReID/deep-person-reid/data/NAICReID/query'

with open("/home/zxh/database/2019-NAIC-ReID/初赛A榜测试集/query_a_list.txt") as f:
    lines = f.readlines()
max = 0
for line in lines:
    line = line.strip().split(' ')
    imgname = line[0].split('/')[1]
    imgid = line[1]
    newimgname = '0000_c1s1_{}_00.png'.format(imgname.split('.')[0])
    print(newimgname)
    shutil.copy(os.path.join(img_root, imgname), os.path.join(copy_to_root, newimgname))
