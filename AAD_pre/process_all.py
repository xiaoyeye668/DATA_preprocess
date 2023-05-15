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


#LOAD DATASET

# load data
df = pd.read_excel('/Users/yeye/Downloads/RA文档/EEG_AAD/exp20221121/analysis/merge2.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
filter_list = ['ExperimentName','Subject', 'QuesDirection.ACC', 'QuesDirection.RT',
                 'QuesNumber1.ACC','QuesNumber1.RT','QuesNumber2.ACC','QuesNumber2.RT',
                 'CorrectRes01','CorrectRes02','CorrectRes03','sequence','SoundFile']
dataframe = df.loc[:,filter_list]
#dataframe = dataframe[dataframe['Subject']==1]
ori_data = dataframe[:].values

##有/无提示变量
taskA_list = []
taskB_list = []
for i in range(len(ori_data)):
    task_id = ori_data[i][0].split('_')[0]
    if task_id == 'ExpA':
        taskA_list.append(ori_data[i])
    else:
        taskB_list.append(ori_data[i])
taskA_data, taskB_data = np.array(taskA_list), np.array(taskB_list)
print('原始数据 有/无线索提示(任务A/B)数据维度：',taskA_data.shape, taskB_data.shape)

def split_by_direction(data):
    left_list, front_list, right_list = [],[],[]
    for i in range(len(data)):
        direction_id = data[i][-5]
        if direction_id == 1:
            left_list.append(data[i])
        elif direction_id == 2:
            front_list.append(data[i])
        else:
            right_list.append(data[i])
    left_data, front_data, right_data = np.array(left_list), np.array(front_list), np.array(right_list)
    #print('左/前中/右方位数据维度：',left_data.shape, front_data.shape, right_data.shape)
    return left_data, front_data, right_data

def split_by_first_num(data):
    num1_3_list, num1_7_list, num1_9_list = [],[],[]
    for i in range(len(data)):
        first_num = data[i][-4]
        if first_num == 4:
            num1_3_list.append(data[i])
        elif first_num == 5:
            num1_7_list.append(data[i])
        else:
            num1_9_list.append(data[i])
    num1_3_data, num1_7_data, num1_9_data = np.array(num1_3_list), np.array(num1_7_list), np.array(num1_9_list)
    #print('第一次数字3/7/9 数据维度：',num1_3_data.shape, num1_7_data.shape, num1_9_data.shape)
    return num1_3_data, num1_7_data, num1_9_data

def split_by_second_num(data):
    num2_3_list, num2_7_list, num2_9_list = [],[],[]
    for i in range(len(data)):
        second_num = data[i][-3]
        if second_num == 4:
            num2_3_list.append(data[i])
        elif second_num == 5:
            num2_7_list.append(data[i])
        else:
            num2_9_list.append(data[i])
    num2_3_data, num2_7_data, num2_9_data = np.array(num2_3_list), np.array(num2_7_list), np.array(num2_9_list)
    #print('原始数据 第二次数字3/7/9 数据维度：',num2_3_data.shape, num2_7_data.shape, num2_9_data.shape)
    return num2_3_data, num2_7_data, num2_9_data


#方位(左/前中/右)变量
left_list, front_list, right_list = [],[],[]
for i in range(len(ori_data)):
    direction_id = ori_data[i][-5]
    if direction_id == 1:
        left_list.append(ori_data[i])
    elif direction_id == 2:
        front_list.append(ori_data[i])
    else:
        right_list.append(ori_data[i])
left_data, front_data, right_data = np.array(left_list), np.array(front_list), np.array(right_list)
print('原始数据 左/前中/右方位数据维度：',left_data.shape, front_data.shape, right_data.shape)

#第一次数字内容(3/7/9)统计
num1_3_list, num1_7_list, num1_9_list = [],[],[]
for i in range(len(ori_data)):
    first_num = ori_data[i][-4]
    if first_num == 4:
        num1_3_list.append(ori_data[i])
    elif first_num == 5:
        num1_7_list.append(ori_data[i])
    else:
        num1_9_list.append(ori_data[i])
num1_3_data, num1_7_data, num1_9_data = np.array(num1_3_list), np.array(num1_7_list), np.array(num1_9_list)
print('原始数据 第一次数字3/7/9 数据维度：',num1_3_data.shape, num1_7_data.shape, num1_9_data.shape)

#第二次数字内容(3/7/9)统计
num2_3_list, num2_7_list, num2_9_list = [],[],[]
for i in range(len(ori_data)):
    second_num = ori_data[i][-3]
    if second_num == 4:
        num2_3_list.append(ori_data[i])
    elif second_num == 5:
        num2_7_list.append(ori_data[i])
    else:
        num2_9_list.append(ori_data[i])
num2_3_data, num2_7_data, num2_9_data = np.array(num2_3_list), np.array(num2_7_list), np.array(num2_9_list)
print('原始数据 第二次数字3/7/9 数据维度：',num2_3_data.shape, num2_7_data.shape, num2_9_data.shape)

subjects = [1,2,3,4,5,6]
print('<<<<<<<<<<<<<<< 准确率 统计分析<<<<<<<<<<<<<<<<<<<<')
for sub in subjects:   
    print('<<<<<<<<<<<<<<< Subject {} <<<<<<<<<<<<<<<<<<<<'.format(sub))
    #有/无提示
    data_sub_A = taskA_data[taskA_data[:,1]==sub]
    #print(data_sub_A.shape)
    print('任务A（有提示） 准确率:', len(data_sub_A[data_sub_A[:,2]==1]) / len(data_sub_A) * 100)
    data_sub_B = taskB_data[taskB_data[:,1]==sub]
    #print(data_sub_B.shape)
    print('任务B（无提示） 准确率:', len(data_sub_B[data_sub_B[:,2]==1]) / len(data_sub_B) * 100)

    #方位判断
    data_sub_left = left_data[left_data[:,1]==sub]
    #print(left_data.shape,data_sub_left.shape)
    print('方位‘左’ 判断准确率:', len(data_sub_left[data_sub_left[:,2]==1]) / len(data_sub_left) * 100)
    data_sub_front = front_data[front_data[:,1]==sub]
    print('方位‘前中‘ 判断准确率:', len(data_sub_front[data_sub_front[:,2]==1]) / len(data_sub_front) * 100)
    data_sub_right = right_data[right_data[:,1]==sub]
    print('方位‘右’ 判断准确率:', len(data_sub_right[data_sub_right[:,2]==1]) / len(data_sub_right) * 100)

    #线索提示+方位判断
    data_sub_left_A, data_sub_front_A, data_sub_right_A = split_by_direction(data_sub_A)
    print('任务A（有提示）方位‘左’ 判断准确率:', len(data_sub_left_A[data_sub_left_A[:,2]==1]) / len(data_sub_left_A) * 100)
    print('任务A（有提示）方位‘前中‘ 判断准确率:', len(data_sub_front_A[data_sub_front_A[:,2]==1]) / len(data_sub_front_A) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('任务A（有提示）方位‘右’ 判断准确率:', len(data_sub_right_A[data_sub_right_A[:,2]==1]) / len(data_sub_right_A) * 100)

    data_sub_left_B, data_sub_front_B, data_sub_right_B = split_by_direction(data_sub_B)
    print('任务B（无提示）方位‘左’ 判断准确率:', len(data_sub_left_B[data_sub_left_B[:,2]==1]) / len(data_sub_left_B) * 100)
    #data_sub_front = front_data[front_data[:,1]==sub]
    print('任务B（无提示）方位‘前中‘ 判断准确率:', len(data_sub_front_B[data_sub_front_B[:,2]==1]) / len(data_sub_front_B) * 100)
    #data_sub_right = right_data[right_data[:,1]==sub]
    print('任务B（无提示）方位‘右’ 判断准确率:', len(data_sub_right_B[data_sub_right_B[:,2]==1]) / len(data_sub_right_B) * 100)


    #数字内容
    data_sub_first_3 = num1_3_data[num1_3_data[:,1]==sub]
    #print(num1_3_data.shape, data_sub_first_3.shape)
    print('第一次数字‘3’ 判断准确率:', len(data_sub_first_3[data_sub_first_3[:,4]==1]) / len(data_sub_first_3) * 100)
    data_sub_first_7 = num1_7_data[num1_7_data[:,1]==sub]
    print('第一次数字‘7’ 判断准确率:', len(data_sub_first_7[data_sub_first_7[:,4]==1]) / len(data_sub_first_7) * 100)
    data_sub_first_9 = num1_9_data[num1_9_data[:,1]==sub]
    print('第一次数字‘9’ 判断准确率:', len(data_sub_first_9[data_sub_first_9[:,4]==1]) / len(data_sub_first_9) * 100)

    data_sub_second_3 = num2_3_data[num2_3_data[:,1]==sub]
    print('第二次数字‘3’ 判断准确率:', len(data_sub_second_3[data_sub_second_3[:,6]==1]) / len(data_sub_second_3) * 100)
    data_sub_second_7 = num2_7_data[num2_7_data[:,1]==sub]
    print('第二次数字‘7’ 判断准确率:', len(data_sub_second_7[data_sub_second_7[:,6]==1]) / len(data_sub_second_7) * 100)
    data_sub_second_9 = num2_9_data[num2_9_data[:,1]==sub]
    print('第二次数字‘9’ 判断准确率:', len(data_sub_second_9[data_sub_second_9[:,6]==1]) / len(data_sub_second_9) * 100)

    print('\n')
  
print('<<<<<<<<<<<<<<< ALL Subject <<<<<<<<<<<<<<<<<<<<')
#有/无提示
data_sub_A = taskA_data
#print(data_sub_A.shape)
print('任务A（有提示） 准确率:', len(data_sub_A[data_sub_A[:,2]==1]) / len(data_sub_A) * 100)
data_sub_B = taskB_data
#print(data_sub_B.shape)
print('任务B（无提示） 准确率:', len(data_sub_B[data_sub_B[:,2]==1]) / len(data_sub_B) * 100)

#方位判断
data_sub_left = left_data
#print(left_data.shape,data_sub_left.shape)
print('方位‘左’ 判断准确率:', len(data_sub_left[data_sub_left[:,2]==1]) / len(data_sub_left) * 100)
data_sub_front = front_data
print('方位‘前中‘ 判断准确率:', len(data_sub_front[data_sub_front[:,2]==1]) / len(data_sub_front) * 100)
data_sub_right = right_data
print('方位‘右’ 判断准确率:', len(data_sub_right[data_sub_right[:,2]==1]) / len(data_sub_right) * 100)

#线索提示+方位判断
data_sub_left_A, data_sub_front_A, data_sub_right_A = split_by_direction(data_sub_A)
print('任务A（有提示）方位‘左’ 判断准确率:', len(data_sub_left_A[data_sub_left_A[:,2]==1]) / len(data_sub_left_A) * 100)
print('任务A（有提示）方位‘前中‘ 判断准确率:', len(data_sub_front_A[data_sub_front_A[:,2]==1]) / len(data_sub_front_A) * 100)
print('任务A（有提示）方位‘右’ 判断准确率:', len(data_sub_right_A[data_sub_right_A[:,2]==1]) / len(data_sub_right_A) * 100)

data_sub_left_B, data_sub_front_B, data_sub_right_B = split_by_direction(data_sub_B)
print('任务B（无提示）方位‘左’ 判断准确率:', len(data_sub_left_B[data_sub_left_B[:,2]==1]) / len(data_sub_left_B) * 100)
print('任务B（无提示）方位‘前中‘ 判断准确率:', len(data_sub_front_B[data_sub_front_B[:,2]==1]) / len(data_sub_front_B) * 100)
print('任务B（无提示）方位‘右’ 判断准确率:', len(data_sub_right_B[data_sub_right_B[:,2]==1]) / len(data_sub_right_B) * 100)

#数字内容
data_sub_first_3 = num1_3_data
#print(num1_3_data.shape, data_sub_first_3.shape)
print('第一次数字‘3’ 判断准确率:', len(data_sub_first_3[data_sub_first_3[:,4]==1]) / len(data_sub_first_3) * 100)
data_sub_first_7 = num1_7_data
print('第一次数字‘7’ 判断准确率:', len(data_sub_first_7[data_sub_first_7[:,4]==1]) / len(data_sub_first_7) * 100)
data_sub_first_9 = num1_9_data
print('第一次数字‘9’ 判断准确率:', len(data_sub_first_9[data_sub_first_9[:,4]==1]) / len(data_sub_first_9) * 100)

data_sub_second_3 = num2_3_data
print('第二次数字‘3’ 判断准确率:', len(data_sub_second_3[data_sub_second_3[:,6]==1]) / len(data_sub_second_3) * 100)
data_sub_second_7 = num2_7_data
print('第二次数字‘7’ 判断准确率:', len(data_sub_second_7[data_sub_second_7[:,6]==1]) / len(data_sub_second_7) * 100)
data_sub_second_9 = num2_9_data
print('第二次数字‘9’ 判断准确率:', len(data_sub_second_9[data_sub_second_9[:,6]==1]) / len(data_sub_second_9) * 100)
print('\n')


#剔除回答错误的trial
#print(ori_data.shape)
err_list = []
print('<<<<<<<<<<<<<<< 反应错误 剔除的trials <<<<<<<<<<<<<<<<<<<<')
err_list.append(ori_data[ori_data[:,2]==0])
data = ori_data[ori_data[:,2]==1]
err_list.append(data[data[:,4]==0])
data = data[data[:,4]==1]
err_list.append(data[data[:,6]==0])
data = data[data[:,6]==1]
#print(err_list)
#print(err_list[0].shape[0],err_list[1].shape[0],err_list[2].shape[0])
'''
with open('exp20221121/analysis/dele_list.txt', 'w') as f:
    f.writelines(','.join(filter_list))
    f.writelines('\n')
    for i in range(len(err_list)):
        for j in range(len(err_list[i])):
            f.writelines(str(err_list[i][j]))
            f.writelines('\n')
'''

#计算反应时均值与标准差
taskA_list = []
taskB_list = []

for i in range(len(data)):
    task_id = data[i][0].split('_')[0]
    version_id = data[i][0].split('_')[1]
    block_id = data[i][0].split('_')[2]
    if task_id == 'ExpA':
        taskA_list.append(data[i])
    else:
        taskB_list.append(data[i])
taskA_data, taskB_data = np.array(taskA_list), np.array(taskB_list)
print('剔除反应错误试次后 任务A/B数据维度：',taskA_data.shape, taskB_data.shape)


print('<<<<<<<<<<<<<<< Task A <<<<<<<<<<<<<<<<<<<<')
print('反应时均值、标准差、阈值:')
direction_rt_means = np.mean(taskA_data[:,3])
direction_rt_std = np.std(taskA_data[:,3])
A_low_th, A_high_th = direction_rt_means - 2.5*direction_rt_std, direction_rt_means + 2.5*direction_rt_std
print(direction_rt_means, direction_rt_std, A_low_th, A_high_th)
taskA_rt_filtered = taskA_data[A_low_th <= taskA_data[:,3].all() <= A_high_th]
#print(taskA_rt_filtered.shape)
taskA_rt_filtered = np.squeeze(taskA_rt_filtered)
#print(taskA_rt_filtered)
print('剔除极值后的数据维度:', taskA_rt_filtered.shape)

print('<<<<<<<<<<<<<<< Task B <<<<<<<<<<<<<<<<<<<<')
print('反应时均值、标准差、阈值 :')
direction_rt_means = np.mean(taskB_data[:,3])
direction_rt_std = np.std(taskB_data[:,3])
B_low_th, B_high_th = direction_rt_means - 2.5*direction_rt_std, direction_rt_means + 2.5*direction_rt_std
print(direction_rt_means, direction_rt_std, A_low_th, A_high_th)
taskB_rt_filtered = taskB_data[B_low_th <= taskB_data[:,3].all() <= B_high_th]
taskB_rt_filtered = np.squeeze(taskB_rt_filtered)
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
    rt_means = np.mean(data_sub_A[:,3])
    rt_std = np.std(data_sub_A[:,3])
    print('任务A（有提示）反应时均值、标准差:', rt_means, rt_std)

    data_sub_B = taskB_rt_filtered[taskB_rt_filtered[:,1]==sub]
    #print(data_sub_B.shape)
    rt_means = np.mean(data_sub_B[:,3])
    rt_std = np.std(data_sub_B[:,3])
    print('任务B（无提示）反应时均值、标准差:', rt_means, rt_std)

    #方位判断
    left_rt, front_rt, right_rt = split_by_direction(all_filtered)
    data_sub_left = left_rt[left_rt[:,1]==sub]
    rt_means = np.mean(data_sub_left[:,3])
    rt_std = np.std(data_sub_left[:,3])
    print('方位‘左’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_front = front_rt[front_rt[:,1]==sub]
    rt_means = np.mean(data_sub_front[:,3])
    rt_std = np.std(data_sub_front[:,3])
    print('方位‘前中’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_right = right_rt[right_rt[:,1]==sub]
    rt_means = np.mean(data_sub_right[:,3])
    rt_std = np.std(data_sub_right[:,3])
    print('方位‘右’ 反应时均值、标准差:', rt_means, rt_std)

    #线索提示+方位判断
    left_A_rt, front_A_rt, right_A_rt = split_by_direction(data_sub_A)
    rt_means = np.mean(left_A_rt[:,3])
    rt_std = np.std(left_A_rt[:,3])
    print('任务A（有提示）时方位‘左’ 反应时均值、标准差:', rt_means, rt_std)

    rt_means = np.mean(front_A_rt[:,3])
    rt_std = np.std(front_A_rt[:,3])
    print('任务A（有提示）时方位‘前中’ 反应时均值、标准差:', rt_means, rt_std)

    rt_means = np.mean(right_A_rt[:,3])
    rt_std = np.std(right_A_rt[:,3])
    print('任务A（有提示）时方位‘右’ 反应时均值、标准差:', rt_means, rt_std)

    left_B_rt, front_B_rt, right_B_rt = split_by_direction(data_sub_B)
    rt_means = np.mean(left_B_rt[:,3])
    rt_std = np.std(left_B_rt[:,3])
    print('任务B（无提示）时方位‘左’ 反应时均值、标准差:', rt_means, rt_std)

    rt_means = np.mean(front_B_rt[:,3])
    rt_std = np.std(front_B_rt[:,3])
    print('任务B（无提示）时方位‘前中’ 反应时均值、标准差:', rt_means, rt_std)

    rt_means = np.mean(right_B_rt[:,3])
    rt_std = np.std(right_B_rt[:,3])
    print('任务B（无提示）时方位‘右’ 反应时均值、标准差:', rt_means, rt_std)

    #第一次数字内容
    num1_3_rt, num1_7_rt, num1_9_rt = split_by_first_num(all_filtered)
    data_sub_num1_3 = num1_3_rt[num1_3_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num1_3[:,5])
    rt_std = np.std(data_sub_num1_3[:,5])
    print('第一次数字‘3’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_num1_7 = num1_7_rt[num1_7_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num1_7[:,5])
    rt_std = np.std(data_sub_num1_7[:,5])
    print('第一次数字‘7’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_num1_9 = num1_9_rt[num1_9_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num1_9[:,5])
    rt_std = np.std(data_sub_num1_9[:,5])
    print('第一次数字‘9’ 反应时均值、标准差:', rt_means, rt_std)

    #第二次数字内容
    num2_3_rt, num2_7_rt, num2_9_rt = split_by_second_num(all_filtered)
    data_sub_num2_3 = num2_3_rt[num2_3_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num2_3[:,7])
    rt_std = np.std(data_sub_num2_3[:,7])
    print('第二次数字‘3’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_num2_7 = num2_7_rt[num2_7_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num2_7[:,7])
    rt_std = np.std(data_sub_num2_7[:,7])
    print('第二次数字‘7’ 反应时均值、标准差:', rt_means, rt_std)

    data_sub_num2_9 = num2_9_rt[num2_9_rt[:,1]==sub]
    rt_means = np.mean(data_sub_num2_9[:,7])
    rt_std = np.std(data_sub_num2_9[:,7])
    print('第二次数字‘9’ 反应时均值、标准差:', rt_means, rt_std)

    print('\n')
