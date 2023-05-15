import sys
import os
from collections import defaultdict

ori_eprime = sys.argv[1]
ori_eeg = sys.argv[2]
tgt_dir = sys.argv[3]

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.cnt'):
                fullname = os.path.join(root, f)
                yield fullname

def main():
    base = './base/'
    for i in findAllFile(base):
        print(i)

#重命名脑电数据
dictmp = defaultdict()
for path, dir_list, file_list in os.walk(ori_eprime):
    for file_name in file_list:
        if file_name.endswith('.edat2') and file_name.split('_')[0].isdigit():
            version_id = path.split('/')[-1][0]+path.split('/')[-1][-2:]
            sub_id = file_name.split('-')[1]
            tmp_name = file_name.split('-')[0]
            new_name = 's'+sub_id+'_'+version_id+'_'+'b'+tmp_name
            block_id = file_name.split('_')[0]
            key = sub_id+'_'+block_id
            dictmp[key] = new_name  #key:sub1_1
        else:
            continue

print(dictmp)
count = 0 
for path, dir_list, file_list in os.walk(ori_eeg):
    for file_name in file_list:
        print(path, file_name)
        #if file_name.endswith('.cnt'):
        if file_name.endswith('.cdt'):
            #print(path, file_name)
            #tgt_path = tgt_dir+'/convert_to_cnt/'+path.split('/')[-1]
            tgt_path = tgt_dir+'/ori_cdt/'+path.split('/')[-1]
            if not os.path.exists(tgt_path):
                os.makedirs(tgt_path)
            #sub_id = file_name.split('_')[0][1:-2]  #cnt->sub_id
            sub_id = file_name.split('.')[0][1:-2]  #cdt->sub_id
            block_id = file_name.split('.')[0][-1]
            ori_path = os.path.join(path, file_name) 
            print(sub_id,block_id)
            key = sub_id+'_'+block_id
            #new_name = dictmp[key]+'.cnt'
            new_name = dictmp[key]+'.cdt'
            print('<<<<<', sub_id,block_id,ori_path,tgt_path,new_name)
            count += 1
            os.system('cp {} {}/{}'.format(ori_path, tgt_path, new_name))
print('the all numer is ', count)
#ori_path = os.path.join(path, file_name) 
#            os.system('cp {} {}/{}'.format(ori_path, tgt_path, new_name))
