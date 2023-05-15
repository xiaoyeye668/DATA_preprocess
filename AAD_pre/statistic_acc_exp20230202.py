#LOAD NECESSARY LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import math


#LOAD DATASET

# load data
df = pd.read_excel('exp20230202/data/6subs.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
filter_list = ['ExperimentName','Subject', 'QuesDirection.ACC', 'QuesDirection.RT',
'QuesNumber.ACC','QuesNumber.RT','CorrectRes01','CorrectRes02','Condition1','sequence','SoundFile']
dataframe = df.loc[:,filter_list]
dataframe = dataframe[dataframe['Condition1']!=55]
ori_data = dataframe[:].values

##有/无提示变量
taskA_list = []
taskB_list = []
for i in range(len(ori_data)):
    task_id = ori_data[i][0].split('_')[1]
    if task_id == 'ExpA':
        taskA_list.append(ori_data[i])
    else:
        taskB_list.append(ori_data[i])
taskA_data, taskB_data = np.array(taskA_list), np.array(taskB_list)
print('原始数据 有/无线索提示(任务A/B)数据维度：',taskA_data.shape, taskB_data.shape)

def split_by_direction(data):
    left_list, front_list, right_list = [],[],[]
    for i in range(len(data)):
        direction_id = data[i][6]
        if direction_id == 'v':
            left_list.append(data[i])
        elif direction_id == 'b':
            front_list.append(data[i])
        else:
            right_list.append(data[i])
    left_data, front_data, right_data = np.array(left_list), np.array(front_list), np.array(right_list)
    #print('左/前中/右方位数据维度：',left_data.shape, front_data.shape, right_data.shape)
    return left_data, front_data, right_data

#方位(左/前中/右)变量
left_list, front_list, right_list = split_by_direction(ori_data)
left_data, front_data, right_data = np.array(left_list), np.array(front_list), np.array(right_list)
print('左/前中/右方位数据维度：',left_data.shape, front_data.shape, right_data.shape)

subjects = [1,4,5,6,99,100]
#subjects = [11]
print('<<<<<<<<<<<<<<< 准确率 统计分析<<<<<<<<<<<<<<<<<<<<')
for sub in subjects:   
    print('<<<<<<<<<<<<<<< Subject {} <<<<<<<<<<<<<<<<<<<<'.format(sub))
    #有/无提示
    data_sub_A = taskA_data[taskA_data[:,1]==sub]
    #print(data_sub_A.shape,len(data_sub_A[data_sub_A[:,2]==1]),len(data_sub_A))
    print('高注意状态-方位判断准确率:', len(data_sub_A[data_sub_A[:,2]==1]) / len(data_sub_A) * 100)
    data_sub_B = taskB_data[taskB_data[:,1]==sub]
    #print(data_sub_B.shape,len(data_sub_B[data_sub_B[:,2]==1]),len(data_sub_B))
    print('低注意状态-方位判断准确率:', len(data_sub_B[data_sub_B[:,2]==1]) / len(data_sub_B) * 100)

    print('高注意状态-数字判断准确率:', len(data_sub_A[data_sub_A[:,4]==1]) / len(data_sub_A) * 100)
    print('低注意状态-数字判断准确率:', len(data_sub_B[data_sub_B[:,4]==1]) / len(data_sub_B) * 100)    
    
    data_sub_left_A, data_sub_front_A, data_sub_right_A = split_by_direction(data_sub_A)
    #高注意状态+方位判断
    #print(data_sub_left_A.shape, data_sub_front_A.shape, data_sub_right_A.shape)
    print('高注意状态-方位‘左’-方位判断准确率:', len(data_sub_left_A[data_sub_left_A[:,2]==1]) / len(data_sub_left_A) * 100)
    print('高注意状态-方位‘前中‘-方位判断准确率:', len(data_sub_front_A[data_sub_front_A[:,2]==1]) / len(data_sub_front_A) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('高注意状态-方位‘右’-方位判断准确率:', len(data_sub_right_A[data_sub_right_A[:,2]==1]) / len(data_sub_right_A) * 100)
    
    #高注意状态+数字判断
    print('高注意状态-方位‘左’-数字判断准确率:', len(data_sub_left_A[data_sub_left_A[:,4]==1]) / len(data_sub_left_A) * 100)
    print('高注意状态-方位‘前中‘-数字判断准确率:', len(data_sub_front_A[data_sub_front_A[:,4]==1]) / len(data_sub_front_A) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('高注意状态-方位‘右’-数字判断准确率:', len(data_sub_right_A[data_sub_right_A[:,4]==1]) / len(data_sub_right_A) * 100)

    data_sub_left_B, data_sub_front_B, data_sub_right_B = split_by_direction(data_sub_B)
    #print(data_sub_left_B.shape, data_sub_front_B.shape, data_sub_right_B.shape)
    #低注意状态+方位判断
    print('低注意状态-方位‘左’-方位判断准确率:', len(data_sub_left_B[data_sub_left_B[:,2]==1]) / len(data_sub_left_B) * 100)
    #data_sub_front = front_data[front_data[:,1]==sub]
    print('低注意状态-方位‘前中‘-方位判断准确率:', len(data_sub_front_B[data_sub_front_B[:,2]==1]) / len(data_sub_front_B) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('低注意状态-方位‘右’-方位判断准确率:', len(data_sub_right_B[data_sub_right_B[:,2]==1]) / len(data_sub_right_B) * 100)

    #低注意状态+数字判断
    print('低注意状态-方位‘左’-数字判断准确率:', len(data_sub_left_B[data_sub_left_B[:,4]==1]) / len(data_sub_left_B) * 100)
    #data_sub_front = front_data[front_data[:,1]==sub]
    print('低注意状态-方位‘前中‘-数字判断准确率:', len(data_sub_front_B[data_sub_front_B[:,4]==1]) / len(data_sub_front_B) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('低注意状态-方位‘右’-数字判断准确率:', len(data_sub_right_B[data_sub_right_B[:,4]==1]) / len(data_sub_right_B) * 100)
    print('\n')
  
print('<<<<<<<<<<<<<<< ALL Subject <<<<<<<<<<<<<<<<<<<<')
#有/无提示
data_sub_A = taskA_data
#print(data_sub_A.shape)
print('高注意状态-方位判断准确率:', len(data_sub_A[data_sub_A[:,2]==1]) / len(data_sub_A) * 100)
data_sub_B = taskB_data
#print(data_sub_B.shape)
print('低注意状态-方位判断准确率:', len(data_sub_B[data_sub_B[:,2]==1]) / len(data_sub_B) * 100)
print('高注意状态-数字判断准确率:', len(data_sub_A[data_sub_A[:,4]==1]) / len(data_sub_A) * 100)
print('低注意状态-数字判断准确率:', len(data_sub_B[data_sub_B[:,4]==1]) / len(data_sub_B) * 100)

#线索提示+方位判断
data_sub_left_A, data_sub_front_A, data_sub_right_A = split_by_direction(data_sub_A)
print('高注意状态-方位‘左’-方位判断准确率:', len(data_sub_left_A[data_sub_left_A[:,2]==1]) / len(data_sub_left_A) * 100)
print('高注意状态-方位‘前中‘-方位判断准确率:', len(data_sub_front_A[data_sub_front_A[:,2]==1]) / len(data_sub_front_A) * 100)
print('高注意状态-方位‘右’-方位判断准确率:', len(data_sub_right_A[data_sub_right_A[:,2]==1]) / len(data_sub_right_A) * 100)
#高注意状态+数字判断
print('高注意状态-方位‘左’-数字判断准确率:', len(data_sub_left_A[data_sub_left_A[:,4]==1]) / len(data_sub_left_A) * 100)
print('高注意状态-方位‘前中‘-数字判断准确率:', len(data_sub_front_A[data_sub_front_A[:,4]==1]) / len(data_sub_front_A) * 100)
print('高注意状态-方位‘右’-数字判断准确率:', len(data_sub_right_A[data_sub_right_A[:,4]==1]) / len(data_sub_right_A) * 100)

data_sub_left_B, data_sub_front_B, data_sub_right_B = split_by_direction(data_sub_B)
print('低注意状态-方位‘左’-方位判断准确率:', len(data_sub_left_B[data_sub_left_B[:,2]==1]) / len(data_sub_left_B) * 100)
print('低注意状态-方位‘前中‘-方位判断准确率:', len(data_sub_front_B[data_sub_front_B[:,2]==1]) / len(data_sub_front_B) * 100)
print('低注意状态-方位‘右’-方位判断准确率:', len(data_sub_right_B[data_sub_right_B[:,2]==1]) / len(data_sub_right_B) * 100)
#低注意状态+数字判断
print('低注意状态-方位‘左’-数字判断准确率:', len(data_sub_left_B[data_sub_left_B[:,4]==1]) / len(data_sub_left_B) * 100)
print('低注意状态-方位‘前中‘-数字判断准确率:', len(data_sub_front_B[data_sub_front_B[:,4]==1]) / len(data_sub_front_B) * 100)
print('低注意状态-方位‘右’-数字判断准确率:', len(data_sub_right_B[data_sub_right_B[:,4]==1]) / len(data_sub_right_B) * 100)
print('\n')

#剔除回答错误的trial
#print(ori_data.shape)
err_list = []
print('<<<<<<<<<<<<<<< 反应错误 剔除的trials <<<<<<<<<<<<<<<<<<<<')
err_list.append(ori_data[ori_data[:,2]==0])
err_list.append(ori_data[ori_data[:,4]==0])
data = ori_data[ori_data[:,2]==1]
data = data[data[:,2]==1]

#err_list.append(data[data[:,4]==0])
#data = data[data[:,4]==1]
#err_list.append(data[data[:,6]==0])
#data = data[data[:,6]==1]
#print(err_list)
#print(err_list[0].shape[0],err_list[1].shape[0],err_list[2].shape[0])

with open('exp20230202/analysis/dele_list.txt', 'w') as f:
    f.writelines(','.join(filter_list))
    f.writelines('\n')
    for i in range(len(err_list)):
        for j in range(len(err_list[i])):
            f.writelines(str(err_list[i][j]))
            f.writelines('\n')

#计算反应时均值与标准差
taskA_list = []
taskB_list = []

for i in range(len(data)):
    task_id = data[i][0].split('_')[1]
    #version_id = data[i][0].split('_')[1]
    block_id = data[i][0].split('_')[0]
    if task_id == 'ExpA':
        taskA_list.append(data[i])
    else:
        taskB_list.append(data[i])
taskA_data, taskB_data = np.array(taskA_list), np.array(taskB_list)
print('剔除反应错误试次后 任务A/B数据维度：',taskA_data.shape, taskB_data.shape)


print('<<<<<<<<<<<<<<< Task A <<<<<<<<<<<<<<<<<<<<')
print('方位判断反应时均值、标准差、阈值:')
direction_rt_means = np.mean(taskA_data[:,3])
direction_rt_std = np.std(taskA_data[:,3])
A_low_th_dir, A_high_th_dir = direction_rt_means - 2.5*direction_rt_std, direction_rt_means + 2.5*direction_rt_std
print(direction_rt_means, direction_rt_std, A_low_th_dir, A_high_th_dir)
print('数字判断反应时均值、标准差、阈值:')
number_rt_means = np.mean(taskA_data[:,5])
number_rt_std = np.std(taskA_data[:,5])
A_low_th_num, A_high_th_num = number_rt_means - 2.5*number_rt_std, number_rt_means + 2.5*number_rt_std
print(number_rt_means, number_rt_std, A_low_th_num, A_high_th_num)
#f = open('exp20221121/analysis/dele_list_new.txt', 'a')
#提出方位反应时极值
#if len(taskA_data[taskA_data[:,3] < A_low_th]):
#    f.write(str(taskA_data[taskA_data[:,3] < A_low_th]))
#    f.write('\n')
taskA_rt_filtered = taskA_data[A_low_th_dir <= taskA_data[:,3]]
#if len(taskA_rt_filtered[taskA_rt_filtered[:,3] > A_high_th]):
#    f.write(str(taskA_rt_filtered[taskA_rt_filtered[:,3] > A_high_th]))
#    f.write('\n')
taskA_rt_filtered = taskA_rt_filtered[taskA_rt_filtered[:,3] <= A_high_th_dir]
#if len(taskA_data[taskA_data[:,3] < A_low_th]):
#    f.write(str(taskA_data[taskA_data[:,3] < A_low_th]))
#    f.write('\n')
#剔除数字反应时极值
taskA_rt_filtered = taskA_rt_filtered[A_low_th_num <= taskA_rt_filtered[:,5]]
#if len(taskA_rt_filtered[taskA_rt_filtered[:,3] > A_high_th]):
#    f.write(str(taskA_rt_filtered[taskA_rt_filtered[:,3] > A_high_th]))
#    f.write('\n')
taskA_rt_filtered = taskA_rt_filtered[taskA_rt_filtered[:,5] <= A_high_th_num]
#print(taskA_rt_filtered.shape)
taskA_rt_filtered = np.squeeze(taskA_rt_filtered)
#print(taskA_rt_filtered)
taskA_rt_filtered[:,0] = 'ExpA'
#print(taskA_rt_filtered[:10,0])
print('剔除极值后的数据维度:', taskA_rt_filtered.shape)

print('<<<<<<<<<<<<<<< Task B <<<<<<<<<<<<<<<<<<<<')
print('反应时均值、标准差、阈值 :')
direction_rt_means = np.mean(taskB_data[:,3])
direction_rt_std = np.std(taskB_data[:,3])
B_low_th_dir, B_high_th_dir = direction_rt_means - 2.5*direction_rt_std, direction_rt_means + 2.5*direction_rt_std
print(direction_rt_means, direction_rt_std, B_low_th_dir, B_high_th_dir)
print('数字判断反应时均值、标准差、阈值:')
number_rt_means = np.mean(taskB_data[:,5])
number_rt_std = np.std(taskB_data[:,5])
B_low_th_num, B_high_th_num = number_rt_means - 2.5*number_rt_std, number_rt_means + 2.5*number_rt_std
print(number_rt_means, number_rt_std, B_low_th_num, B_high_th_num)
#剔除方位反应时极值
#taskB_rt_filtered = taskB_data[B_low_th <= taskB_data[:,3].all() <= B_high_th]
#if len(taskB_data[taskB_data[:,3] < B_low_th]):
#    f.write(str(taskB_data[taskB_data[:,3] < B_low_th]))
#    f.write('\n')
taskB_rt_filtered = taskB_data[B_low_th_dir <= taskB_data[:,3]]
#if len(taskB_rt_filtered[taskB_rt_filtered[:,3] > B_high_th]):
#    f.write(str(taskB_rt_filtered[taskB_rt_filtered[:,3] > B_high_th]))
#    f.write('\n')
taskB_rt_filtered = taskB_rt_filtered[taskB_rt_filtered[:,3] <= B_high_th_dir]
#if len(taskB_data[taskB_data[:,3] < B_low_th]):
#    f.write(str(taskB_data[taskB_data[:,3] < B_low_th]))
#    f.write('\n')
#剔除数字反应时极值
taskB_rt_filtered = taskB_rt_filtered[B_low_th_num <= taskB_rt_filtered[:,5]]
#if len(taskB_rt_filtered[taskB_rt_filtered[:,3] > B_high_th]):
#    f.write(str(taskB_rt_filtered[taskB_rt_filtered[:,3] > B_high_th]))
#    f.write('\n')
taskB_rt_filtered = taskB_rt_filtered[taskB_rt_filtered[:,5] <= B_high_th_num]

taskB_rt_filtered = np.squeeze(taskB_rt_filtered)
taskB_rt_filtered[:,0] = 'ExpB'
#print(taskB_rt_filtered)
print('剔除极值后的数据维度:', taskB_rt_filtered.shape)

print('\n')
print('<<<<<<<<<<<<<<<反应时 统计分析<<<<<<<<<<<<<<<<<<<<')
all_filtered = np.concatenate((taskA_rt_filtered, taskB_rt_filtered),axis=0)
#print(all_filtered.shape)

for sub in subjects:
    data_sub_A = taskA_rt_filtered[taskA_rt_filtered[:,1]==sub]
    print('<<<<<<<<<<<<<<< Subject {} <<<<<<<<<<<<<<<<<<<<'.format(sub))
    #print(data_sub_A.shape)
    rt_means_dir = np.mean(data_sub_A[:,3])
    rt_std_dir = np.std(data_sub_A[:,3])
    print('高注意状态-方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(data_sub_A[:,5])
    rt_std_num = np.std(data_sub_A[:,5])
    print('高注意状态-方位判断反应时均值、标准差:', rt_means_num, rt_std_num)

    data_sub_B = taskB_rt_filtered[taskB_rt_filtered[:,1]==sub]
    #print(data_sub_B.shape)
    rt_means_dir = np.mean(data_sub_B[:,3])
    rt_std_dir = np.std(data_sub_B[:,3])
    print('低注意状态-方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(data_sub_B[:,5])
    rt_std_num = np.std(data_sub_B[:,5])
    print('低注意状态-方位判断反应时均值、标准差:', rt_means_num, rt_std_num)

    #线索提示+方位判断
    left_A_rt, front_A_rt, right_A_rt = split_by_direction(data_sub_A)
    rt_means_dir = np.mean(left_A_rt[:,3])
    rt_std_dir = np.std(left_A_rt[:,3])
    print('高注意状态-方位‘左’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(left_A_rt[:,5])
    rt_std_num = np.std(left_A_rt[:,5])
    print('高注意状态-方位‘左’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    rt_means_dir = np.mean(front_A_rt[:,3])
    rt_std_dir = np.std(front_A_rt[:,3])
    print('高注意状态-方位‘前中’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(front_A_rt[:,5])
    rt_std_num = np.std(front_A_rt[:,5])
    print('高注意状态-方位‘前中’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    rt_means_dir = np.mean(right_A_rt[:,3])
    rt_std_dir = np.std(right_A_rt[:,3])
    print('高注意状态-方位‘右’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(right_A_rt[:,5])
    rt_std_num = np.std(right_A_rt[:,5])
    print('高注意状态-方位‘右’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    left_B_rt, front_B_rt, right_B_rt = split_by_direction(data_sub_B)
    rt_means_dir = np.mean(left_B_rt[:,3])
    rt_std_dir = np.std(left_B_rt[:,3])
    print('低注意状态-方位‘左’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(left_B_rt[:,5])
    rt_std_num = np.std(left_B_rt[:,5])
    print('低注意状态-方位‘左’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    rt_means_dir = np.mean(front_B_rt[:,3])
    rt_std_dir = np.std(front_B_rt[:,3])
    print('低注意状态-方位‘前中’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(front_B_rt[:,5])
    rt_std_num = np.std(front_B_rt[:,5])
    print('低注意状态-方位‘前中’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    rt_means_dir = np.mean(right_B_rt[:,3])
    rt_std_dir = np.std(right_B_rt[:,3])
    print('低注意状态-方位‘右’ -方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
    rt_means_num = np.mean(right_B_rt[:,5])
    rt_std_num = np.std(right_B_rt[:,5])
    print('低注意状态-方位‘右’ -数字判断反应时均值、标准差:', rt_means_num, rt_std_num)

    print('\n')


print('<<<<<<<<<<<<<<< ALL Subject RT <<<<<<<<<<<<<<<<<<<<')
data_sub_A = taskA_rt_filtered
print(data_sub_A.shape)
rt_means_dir = np.mean(data_sub_A[:,3])
rt_std_dir = np.std(data_sub_A[:,3])
#print(data_sub_A[:,3].shape)
print('高注意状态-方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
rt_means_num = np.mean(data_sub_A[:,5])
rt_std_num = np.std(data_sub_A[:,5])
#print(data_sub_A[:,5].shape)
print('高注意状态-方位判断反应时均值、标准差:', rt_means_num, rt_std_num)

data_sub_B = taskB_rt_filtered
print(data_sub_B.shape)
rt_means_dir = np.mean(data_sub_B[:,3])
rt_std_dir = np.std(data_sub_B[:,3])
print('低注意状态-方位判断反应时均值、标准差:', rt_means_dir, rt_std_dir)
rt_means_num = np.mean(data_sub_B[:,5])
rt_std_num = np.std(data_sub_B[:,5])
print('低注意状态-数组判断反应时均值、标准差:', rt_means_num, rt_std_num)

#线索提示+方位判断
left_A_rt, front_A_rt, right_A_rt = split_by_direction(data_sub_A)
rt_means = np.mean(left_A_rt[:,3])
rt_std = np.std(left_A_rt[:,3])
print('高注意状态-方位‘左’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(left_A_rt[:,5])
rt_std = np.std(left_A_rt[:,5])
print('高注意状态-方位‘左’-数字判断 反应时均值、标准差:', rt_means, rt_std)

rt_means = np.mean(front_A_rt[:,3])
rt_std = np.std(front_A_rt[:,3])
print('高注意状态-方位‘前中’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(front_A_rt[:,5])
rt_std = np.std(front_A_rt[:,5])
print('高注意状态-方位‘前中’-数字判断 反应时均值、标准差:', rt_means, rt_std)

rt_means = np.mean(right_A_rt[:,3])
rt_std = np.std(right_A_rt[:,3])
print('高注意状态-方位‘右’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(right_A_rt[:,5])
rt_std = np.std(right_A_rt[:,5])
print('高注意状态-方位‘右’-数字判断 反应时均值、标准差:', rt_means, rt_std)

left_B_rt, front_B_rt, right_B_rt = split_by_direction(data_sub_B)
rt_means = np.mean(left_B_rt[:,3])
rt_std = np.std(left_B_rt[:,3])
print('低注意状态-方位‘左’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(left_B_rt[:,5])
rt_std = np.std(left_B_rt[:,5])
print('低注意状态-方位‘左’-素质判断 反应时均值、标准差:', rt_means, rt_std)

rt_means = np.mean(front_B_rt[:,3])
rt_std = np.std(front_B_rt[:,3])
print('低注意状态-方位‘前中’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(front_B_rt[:,5])
rt_std = np.std(front_B_rt[:,5])
print('低注意状态-方位‘前中’-数字判断 反应时均值、标准差:', rt_means, rt_std)

rt_means = np.mean(right_B_rt[:,3])
rt_std = np.std(right_B_rt[:,3])
print('低注意状态-方位‘右’-方位判断 反应时均值、标准差:', rt_means, rt_std)
rt_means = np.mean(right_B_rt[:,5])
rt_std = np.std(right_B_rt[:,5])
print('低注意状态-方位‘右’-数字判断 反应时均值、标准差:', rt_means, rt_std)

'''
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

#formula = 'c~ a + b '
import pandas as pd
filter_list = ['ExperimentName','Subject', 'QuesDirectionACC', 'QuesDirectionRT',
                'QuesNumber1ACC','QuesNumber1RT','QuesNumber2ACC','QuesNumber2RT',
                 'CorrectRes01','CorrectRes02','CorrectRes03','sequence','SoundFile']
df_new = pd.DataFrame(all_filtered,columns=filter_list)
print(df_new.shape)
#print(df_new['ExperimentName'],df_new['CorrectRes01'],df_new['QuesDirectionRT'])
df_new['ExperimentName'] = df_new['ExperimentName'].map(str)
df_new['CorrectRes01'] = df_new['CorrectRes01'].map(str)
df_new['QuesDirectionRT'] = df_new['QuesDirectionRT'].map(int)
#print(df_new.info())
formula = 'QuesDirectionRT ~ C(CorrectRes01) + C(ExperimentName) + C(CorrectRes01):C(ExperimentName)'
model = ols(formula, df_new).fit()
anovaResults = anova_lm(model)
 # --- >>> STOP stats <<< ---
print(anovaResults)
#anova_results = anova_lm(ols(formula,df).fit())
#print(anova_results)
'''

