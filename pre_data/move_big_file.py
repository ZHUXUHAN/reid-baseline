import os
import shutil

mv_to_dir = '/home/zxh/datasets/NAICReID/top2_bounding_box_train'

files = os.listdir('/home/zxh/datasets/NAICReID/bounding_box_train')
for file in files:
    shutil.copy(os.path.join('/home/zxh/datasets/NAICReID/bounding_box_train', file), mv_to_dir)
