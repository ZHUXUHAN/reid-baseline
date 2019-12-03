import os
import  shutil
# files = os.listdir('/home/zxh/datasets/NAICReID/cleannew_bounding_box_test')
# new_dir = '/home/zxh/datasets/NAICReID/new_cleannew_gallery'
# print(len(files))
# for i, file in enumerate(files):
#     newname = file.replace(file[:4], str(i + 4768).zfill(4))
#     # print(newname)
#     shutil.copy(os.path.join('/home/zxh/datasets/NAICReID/cleannew_bounding_box_test', file), os.path.join(new_dir, newname))


# files = os.listdir('/home/zxh/datasets/NAICReID/query')
# ori_files = os.listdir('/home/zxh/datasets/NAICReID/cleannew_query')
# print(len(files))
# print(len(ori_files))
# ff = []
# for file in ori_files:
#     name = file.split('_')[2]
#     ff.append(name)
# not_name = []
# for file_1 in files:
#     # name_1 = file_1.split('.')[0]
#     name_1 = file_1.split('_')[2]
#     if name_1 not in ff:
#         not_name.append(name_1)
# print(len(not_name))
#
# top_list = []
# top_file = os.listdir('/home/zxh/datasets/NAICReID/top_bounding_box_train')
# for file in top_file:
#     name = file.split('_')[2]
#     top_list.append(name)
#
# print(len(top_list))
# file_res = {}
# ids = {}
# for f in not_name:
#     for file in top_file:
#         id = file.split('_')[0]
#         name = file.split('_')[2]
#         if f == name:
#             ids[f] = id
# print(len(ids))
# # for key, id_ in ids.items():
# for ii, key in enumerate(ids.keys()):
#     id_ = ids[key]
#     file_res[ii] = []
#     for file in top_file:
#         id = file.split('_')[0]
#         if id == id_:
#             file_res[ii].append(file)
# for id, files in file_res.items():
#     for file in files:
#         new_name = file.replace(file[:4], str(int(id) + 5829).zfill(4))
#         print(new_name)
#         shutil.copy(os.path.join('/home/zxh/datasets/NAICReID/top_bounding_box_train', file), os.path.join('/home/zxh/datasets/NAICReID/my_bounding_box_train', new_name))



# gallery_list = os.listdir('/home/zxh/datasets/NAICReID/cleannew_bounding_box_test')
# query_list = os.listdir('/home/zxh/datasets/NAICReID/cleannew_query')
#
#
# res_file = []
# for i, q in enumerate(query_list):
#     q_id = q.split('_')[0]
#     for g in gallery_list:
#         g_id = g.split('_')[0]
#         if q_id == g_id:
#             print(q_id)
#             new_name_q = q.replace(q[:4], str(i + 4768).zfill(4))
#             new_name_g = g.replace(g[:4], str(i + 4768).zfill(4))
#             shutil.copy(os.path.join('/home/zxh/datasets/NAICReID/cleannew_query', q), os.path.join('/home/zxh/datasets/NAICReID/my_bounding_box_train', new_name_q))
#             shutil.copy(os.path.join('/home/zxh/datasets/NAICReID/cleannew_bounding_box_test', g),
#                         os.path.join('/home/zxh/datasets/NAICReID/my_bounding_box_train', new_name_g))

# files = os.listdir('/home/zxh/datasets/NAICReID/my_bounding_box_train')
#
# img_4768_5828 = []
# img_5828_6116 = {}
# for file in files:
#     id = file.split('_')[0]
#     img_5828_6116[id] = []
#     name = file.split('_')[2]
#     if int(id)<=5828 and int(id) >= 4768:
#         img_4768_5828.append(name)
#     if int(id) > 5828:
#         img_5828_6116[id].append(name)
#
# del_lsit = []
# for id, names in img_5828_6116.items():
#     for name in names:
#         if name in img_4768_5828:
#             del_lsit.append(name)
# print(del_lsit)
# for file in files:
#     name = file.split('_')[2]
#     id = file.split('_')[0]
#     if int(id) > 5828:
#         if name in del_lsit:
#             os.remove(os.path.join('/home/zxh/datasets/NAICReID/my_bounding_box_train', file))

files = os.listdir('/home/zxh/datasets/NAICReID/query')
files_1 = os.listdir('/home/zxh/datasets/NAICReID/my_bounding_box_train')
# l= []
# for file in files:
#     for top_file in files_1:
#       name_1 = file.split('_')[2]
#       name_2 = top_file.split('_')[2]
#       if name_1==name_2:
#           l.append(file)
# print(len(l))


names =[]
for file in files_1:
    id = file.split('_')[0]
    name = file.split('_')[2]
    if int(id) >= 4768:
        names.append(name)


from collections import Counter
list=names
result=Counter(list)
for k, v in result.items():
    if v>1:
        print(k, v)
