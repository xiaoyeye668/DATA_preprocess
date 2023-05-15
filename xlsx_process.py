import os
import sys
import re
import xlrd  # 导入库
# 打开文件

tgt_file = open(sys.argv[1], 'w')

#xlsx = xlrd.open_workbook('words.xlsx')
xlsx=xlrd.open_workbook('newWords.xlsx')
# 查看所有sheet列表
print('All sheets: %s' % xlsx.sheet_names())

#sheet1 = xlsx.sheets()[0]    # 获得第1张sheet，索引从0开始
sheet1 = xlsx.sheets()[1]
sheet1_name = sheet1.name    # 获得名称
sheet1_cols = sheet1.ncols   # 获得列数
sheet1_nrows = sheet1.nrows  # 获得行数
print('Sheet1 Name: %s\nSheet1 cols: %s\nSheet1 rows: %s' % (sheet1_name, sheet1_cols, sheet1_nrows))
words = []
for i in range(sheet1_cols):
	for j in range(sheet1_nrows):
		#print(sheet1.row(j)[i].value.split('\n'))
		word = re.split('\n|,|;|；| ', sheet1.row(j)[i].value)
		word = [word for word in word if word != '']
		words += word
for word in words:
	tgt_file.write('{}\n'.format(word.upper()))

tgt_file.close()

#sheet1_nrows4 = sheet1.row_values(4)  # 获得第4行数据
#sheet1_cols2 = sheet1.col_values(2)   # 获得第2列数据
#cell23 = sheet1.row(2)[3].value       # 查看第3行第4列数据
#print('Row 4: %s\nCol 2: %s\nCell 1: %s\n' % (sheet1_nrows4, sheet1_cols2, cell23))
