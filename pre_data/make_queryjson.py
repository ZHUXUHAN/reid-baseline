import json
import os

query_json = {}

files = os.listdir('/home/zxh/datasets/NAICReID/query')

for file in files:
    id = file.split('_')[0]
    imgname = file.split('_')[2]
    query_json[imgname] = id

with open(os.path.join('./', 'query.json'), 'w', encoding='utf-8') as f:
    json.dump(query_json, f)
