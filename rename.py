import sys
import os
#/Volumes/Yeye/听觉注意/eprime_data
ori_dir = sys.argv[1]
tgt_dir = sys.argv[2]

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
#重命名eprime数据
count =0
for path, dir_list, file_list in os.walk(ori_dir):
    for file_name in file_list:
        if file_name.endswith('.edat2') and file_name.split('_')[0].isdigit():
            print(path, file_name)
            version_id = path.split('/')[-1][0]+path.split('/')[-1][-2:]
            sub_id = file_name.split('-')[1]
            #sub_id = path.split('/')[2][0]+path.split('/')[2][-2:]
            tgt_path = tgt_dir+'/SUB'+sub_id
            print(tgt_path, sub_id)
            if not os.path.exists(tgt_path):
                os.makedirs(tgt_path)
            tmp_name = file_name.split('-')[0]
            new_name = 's'+sub_id+'_'+version_id+'_'+'b'+tmp_name+'.edat2'
            ori_path = os.path.join(path, file_name) 
            print('<<<<',version_id, sub_id, ori_path, new_name)
            count += 1
            os.system('cp {} {}/{}'.format(ori_path, tgt_path, new_name))
        else:
            continue
print('the all numer is ', count)
'''
#重命名脑电数据
for path, dir_list, file_list in os.walk(ori_dir):
    for file_name in file_list:
        if file_name.endswith('.txt'):
            version_id = path.split('/')[1][0]+path.split('/')[1][-2:]
            sub_id = path.split('/')[2][0]+path.split('/')[2][-2:]
            tgt_path = tgt_dir+'/SUB'+path.split('/')[2][-2:]
            print(tgt_path)
            if not os.path.exists(tgt_path):
                os.makedirs(tgt_path)
            new_name = sub_id+'_'+version_id+'_'+'b'+file_name
            print('<<<<',version_id, sub_id, new_name,os.path.join(path, file_name))
            ori_path = os.path.join(path, file_name) 
            os.system('cp {} {}/{}'.format(ori_path, tgt_path, new_name))
        else:
            continue
'''