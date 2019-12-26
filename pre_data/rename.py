import os
import shutil

query_path = '/home/zxh/datasets/NAICReID/labeled_query'
gallery_path = '/home/zxh/datasets/NAICReID/labeled_gallery'
query_newpath = '/home/zxh/datasets/NAICReID/relabeled_query'
gallery_newpath = '/home/zxh/datasets/NAICReID/relabeled_gallery'

query_files = os.listdir(query_path)
gallery_files = os.listdir(gallery_path)

for file in query_files:
    query_id = file.split('_')[0]
    query_newid = int(query_id)+9968
    query_newname = file.replace(file[:4], str(query_newid))
    print(query_newname)
    ori_path = os.path.join(query_path, file)
    new_path = os.path.join(query_newpath, query_newname)

    shutil.copy(ori_path, new_path)


for file in gallery_files:
    gallery_id = file.split('_')[0]
    gallery_newid = int(gallery_id)+9968
    gallery_newname = file.replace(file[:4], str(gallery_newid))
    print(query_newname)
    ori_path = os.path.join(gallery_path, file)
    new_path = os.path.join(gallery_newpath, gallery_newname)

    shutil.copy(ori_path, new_path)
