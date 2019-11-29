# import os
# import os
# import shutil
# query_list = os.listdir('./query')
# new_dir = './new_query'
#
# for i, qq in enumerate(query_list):
#     newname = qq.replace('0000', str(i+4768).zfill(4))
#
#     shutil.copy(os.path.join('./query', qq), os.path.join(new_dir, newname))

import os
import shutil

# files_3 = os.listdir('/home/zxh/datasets/NAICReID/newgallery_b_rank3')
# files = os.listdir('/home/zxh/datasets/NAICReID/newgallery_b_rank4')
# print(len(files_3))
# print(len(files))
files = os.listdir('/home/zxh/my_bounding_box_train')
# for file in files:
#     if 'png' not in file:
#         print(file)

for file in files:
    shutil.copy(os.path.join('/home/zxh/my_bounding_box_train', file), '/home/zxh/datasets/NAICReID/my_rank1_bounding_box_train')
