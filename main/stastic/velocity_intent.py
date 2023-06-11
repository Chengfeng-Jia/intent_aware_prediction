# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : velocity_intent.py
# Time       ：2021/5/7 10:28
# Author     ：J ▄︻┻┳═一
# version    ：python 3.6
# Description：
"""
import pandas as pd
import numpy as np
from dateutil.parser import parse
import random
# import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

font2 = font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simfang.TTF', size=15)
csv_loc=r'D:\Marine\desk\beifen\m\trajectory_prediction\TF_TCN-master_j\copymem\try_data\all1209\after_warning.csv'
df = pd.read_csv(csv_loc)
# print(df)
df0=df.loc[df['label'] == 0].head(20000)
print(df0)
df_arr0=np.array(df0[['id','x','y','v','a','time', 'rankc','label']])

df1=df.loc[df['label'] == 1].head(20000)
print(df1)
df_arr1=np.array(df1[['id','x','y','v','a','time', 'rankc','label']])

df2=df.loc[df['label'] == 2].head(20000)
print(df2)
df_arr2=np.array(df2[['id','x','y','v','a','time', 'rankc','label']])

delta_t=[]
averange_v=[]
for line in df_arr2:
    # print(i)
    if line[6]==1:
        start=line[5].split('\'')[1]
        # print(start)
        a=parse(start)
        sum_v=[line[3]]
    if line[6]==0:
        sum_v.append(line[3])
        end=line[5].split('\'')[1]
        # print(end)
        b=parse(end)
        t=(b-a).seconds
        # print(t)
        delta_t.append(t)
        mean_v=np.mean(sum_v)
        averange_v.append(mean_v)
    else:sum_v.append(line[3])
print(delta_t)

print(averange_v)
delta_t=[t+random.randint(0, 100) for t in delta_t]
averange_v=[1/v for v in averange_v]
ax=sns.jointplot(x = delta_t, y =averange_v,kind = 'reg',color='crimson')
# plt.xlabel('Time (s)')
# plt.ylabel('Average of the speed (km/h)')
# ax.set_axis_labels('Time (s)', 'Average of the speed (km/h)',fontproperties=font2)
ax.set_axis_labels('时间（秒）', '平均速度（海里/小时）',fontproperties=font2)

plt.savefig(r"F:\wangpan\OneDrive\桌面\right_xiaoheng.png",dpi = 800,bbox_inches = 'tight')
plt.show()
