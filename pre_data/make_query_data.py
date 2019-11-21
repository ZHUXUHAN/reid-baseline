import os
import shutil
query_list = os.listdir('./query')
new_dir = './new_query'

for i, qq in enumerate(query_list):
    newname = qq.replace('0000', str(i+4768).zfill(4))

    shutil.copy(os.path.join('./query', qq), os.path.join(new_dir, newname))
