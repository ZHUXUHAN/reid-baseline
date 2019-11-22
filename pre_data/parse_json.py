import json
import shutil
import os

m_file = './json/submission8501.json'
query_json = './query.json'
ori_gallery_path = ''
new_gallery_path = ''
ori_query_path = ''
new_query_path = ''
train_all_classes = 4768
query_all_classes = 1347

with open(m_file, 'r') as f:
    res = json.load(f)
with open(query_json, 'r') as f:
    query_name_id = json.load(f)

keys = res.keys()
len_keys = len(set(keys))
if len_keys < 1348:
    assert "WRONG"

max = 0
for key, value in res.items():
    len_value = len(set(value))
    if len_value != 200:
        assert "WRONG"
    query_id = int(query_name_id[key.split('.')[0]]) + 4768
    query_newname = '{}_c1s1_{}_00.png'.format(str(query_id), key.split('.')[0])
    query_oriname = '0000_c1s1_{}_00.png'.format(key.split('.')[0])
    shutil.copy(os.path.join(ori_query_path, query_oriname), os.path.join(new_query_path, query_newname))
    if query_id > max:
        max = query_id
    top4 = value[:4]
    for i in range(4):
        gallery_imgname = top4[i]
        gallery_oriname = '0000_c1s1_{}_00.png'.format(gallery_imgname.split('.')[0])
        gallery_newname = '{}_c1s1_{}_00.png'.format(str(query_id), gallery_imgname.split('.')[0])
        shutil.copy(os.path.join(ori_gallery_path, gallery_oriname), os.path.join(new_gallery_path, gallery_newname))

if max!= train_all_classes+query_all_classes:
    assert "WRONG"

