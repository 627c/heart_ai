import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import time
import psutil
import GPUtil
import seaborn as sns

from heart_2C.heart_myunet.vision import _2_1,_2_2,_2_3,_2_4,_2_5,_2_6,_2_7,_2_4_1,_2_4_2,_2_4_3,_2_4_4,_2_4_5,_2_4_6,_2_4_7,_2_4_8,_2_4_9,_2_4_10
from heart_3C.heart_myunet.vision import _3_1,_3_2,_3_3,_3_4,_3_5,_3_6,_3_7
from heart_4C.heart_myunet.vision import _4_1,_4_2,_4_3,_4_4,_4_5,_4_6,_4_7

# 生成模拟数据
# _1=format(_2_1,'.2f')
_1=_2_1
_2=_3_1
_3=_4_5
_4=_2_5
_5=_3_5
_6=_4_1
_7=_2_2
_8=_3_2
_9=_4_6
_10=_2_6
_11=_3_6
_12=_4_2
_13=_2_3
_14=(_3_3+_4_7)/2
_15=_2_7
_16=(_3_7+_4_3)/2
_17=(_2_4+_3_4+_4_4)/3

# segment_values = [_1*17/21,_1*22/21,_1*31/21,_1*20/21,_1*32/21,_1*18/21,_1*17/21,_1*16/21,_1*19/21,_1*16/21,_2*17/21,_2*22/21,_2*31/21,_2*20/21,_2*32/21,_2*18/21,_2*17/21,_2*16/21,_2*19/21,_2*16/21,_3*17/21,_3*22/21,_3*31/21,_3*20/21,_3*32/21,_3*18/21,_3*17/21,_3*16/21,_3*19/21,_3*16/21,_4*17/21,_4*22/21,_4*31/21,_4*20/21,_4*32/21,_4*18/21,_4*17/21,_4*16/21,_4*19/21,_4*16/21,_5*17/21,_5*22/21,_5*31/21,_5*20/21,_5*32/21,_5*18/21,_5*17/21,_5*16/21,_5*19/21,_5*16/21,_6*17/21,_6*22/21,_6*31/21,_6*20/21,_6*32/21,_6*18/21,_6*17/21,_6*16/21,_6*19/21,_6*16/21,_7*17/21,_7*22/21,_7*31/21,_7*20/21,_7*32/21,_7*18/21,_7*17/21,_7*16/21,_7*19/21,_7*16/21,_8*17/21,_8*22/21,_8*31/21,_8*20/21,_8*32/21,_8*18/21,_8*17/21,_8*16/21,_8*19/21,_8*16/21,_9*17/21,_9*22/21,_9*31/21,_9*20/21,_9*32/21,_9*18/21,_9*17/21,_9*16/21,_9*19/21,_9*16/21,_10*17/21,_10*22/21,_10*31/21,_10*20/21,_10*32/21,_10*18/21,_10*17/21,_10*16/21,_10*19/21,_10*16/21,_11*17/21,_11*22/21,_11*31/21,_11*20/21,_11*32/21,_11*18/21,_11*17/21,_11*16/21,_11*19/21,_11*16/21,_12*17/21,_12*22/21,_12*31/21,_12*20/21,_12*32/21,_12*18/21,_12*17/21,_12*16/21,_12*19/21,_12*16/21,_13*17/21,_13*22/21,_13*31/21,_13*20/21,_13*32/21,_13*18/21,_13*17/21,_13*16/21,_13*19/21,_13*16/21,_14*17/21,_14*22/21,_14*31/21,_14*20/21,_14*32/21,_14*18/21,_14*17/21,_14*16/21,_14*19/21,_14*16/21,_15*17/21,_15*22/21,_15*31/21,_15*20/21,_15*32/21,_15*18/21,_15*17/21,_15*16/21,_15*19/21,_15*16/21,_16*17/21,_16*22/21,_16*31/21,_16*20/21,_16*32/21,_16*18/21,_16*17/21,_16*16/21,_16*19/21,_16*16/21,_2_4_1,_2_4_2,_2_4_3,_2_4_4,_2_4_5,_2_4_6,_2_4_7,_2_4_8,_2_4_9,_2_4_10]
segment_values = [40.67306486,40.07558442,21.30344414,23.6625146,23.55185241,27.17518144,25.19616794,23.86004896,28.45628692,25.44185289
,8.422861546,8.385688897,9.598573413,10.03471126,10.17907063,11.27018936,9.997295948,8.84625282,10.45199695,10.70056362
,62.06346908,72.04060602,71.44517813,69.7691989,70.95059138,69.71016187,67.44373717,72.56973985,75.61226066,78.2682551
,29.25266288,29.12491643,28.61228479,31.31806477,27.6567515,25.39258774,26.49586729,30.8127448,32.83265933,33.13156666
,9.931261497,12.85858069,12.68427876,11.85544393,12.53180844,11.48628141,19.84563175,13.37721971,16.10698317,13.22232635
,89.25713913,88.30976738,86.13061636,88.58147695,91.58168203,82.05939089,81.32509818,105.0207562,101.681204,102.6967509
,46.67477128,45.90990129,47.41003783,41.72252606,45.34729264,45.9595097,41.62347186,44.10848962,42.50611776,42.45541649
,8.303364546,8.121145159,7.302533455,9.041421785,7.827268767,7.513692812,7.675433974,7.773007154,7.689772024,8.794990019
,65.17708049,67.05973991,64.31420939,67.90091738,63.72820421,59.50232223,63.54028186,68.31633871,64.44557093,65.91427586
,24.38395335,26.71545193,34.29553489,37.51967014,33.77425978,28.44891344,35.41858852,36.51466054,35.59719395,36.71553081
,24.38395335,26.71545193,34.29553489,37.51967014,33.77425978,28.44891344,35.41858852,36.51466054,35.59719395,36.71553081
,82.75515833,78.13307568,78.82557733,78.83365464,84.8083374,82.55866645,91.22946082,86.38677132,84.48157792,82.35326
,35.32115991,35.4767719,38.8149298,42.01899968,41.95353047,40.92129724,46.25921279,42.03403069,39.2300765,38.64122281
,37.73508601,38.54483257,34.78279242,35.00946725,35.91780301,34.44981232,40.90278178,40.38245677,39.89661828,36.80784072
,23.23928467,19.18302603,21.30344414,23.6625146,23.55185241,27.17518144,25.19616794,23.86004896,28.45628692,25.44185289
,44.74621237,37.69504715,37.41274525,38.63848435,39.44457753,43.21122745,44.59845795,44.32847974,47.33050374,46.27239406
,25.26408877,24.18244814,31.42596989,27.97151865,33.08723506,30.31910849,27.27623347,32.27891253,35.04994146,38.61275251
]

segment_values = np.array(segment_values,dtype=np.float32)
print(segment_values)
def plot_bulls_eye(data):
    """
    画出 17 节段的牛眼图，并根据数据指定对应的颜色。
    :param data: 包含 17 个节段的数据列表或 numpy 数组。
    """
    # assert len(data) == 17, "数据长度必须为 17"

    # 定义每个节段的颜色（这里我们使用伪彩色图）
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    idx = 0
    def fill_gradient(ax,theta_start, theta_end, r_inner, r_outer, color_start, color_end):
        """Fill a wedge with gradient color."""
        for i in range(500):  # Increase the number of interpolation steps for smoother transition
            t0 = theta_start + (theta_end - theta_start) * (i / 500)
            t1 = theta_start + (theta_end - theta_start) * ((i + 1) / 500)
            c = cmap(norm(color_start + (color_end - color_start) * (i / 500)))
            ax.fill_between([t0, t1], r_inner, r_outer, color=c, edgecolor=None,linewidth=0)

    # 最外圈：6块
    radii_outer = [0.75, 1.0]
    segment_angles_outer=np.linspace(0,2*np.pi,31)
    for i in range(30):
        if idx >= len(data): break
        theta_start = segment_angles_outer[i]
        theta_end = segment_angles_outer[i+1]
        next_data = data[idx+1] if idx+1 < len(data) else data[idx]
        radii_outer[1]=radii_outer[0]+0.125
        for t in range(2):
            fill_gradient(ax,theta_start,theta_end,radii_outer[0],radii_outer[1],data[idx],next_data)
        # ax.fill_between(np.linspace(theta_start, theta_end, 100), radii_outer[0], radii_outer[1],
        #                 color=cmap(norm(data[idx])), edgecolor=None)
            theta_mid = (theta_start + theta_end) / 2
            r_mid = (radii_outer[0] + radii_outer[1]) / 2
            # ax.text(theta_start+(i//3+t/10)*(np.pi)/10, r_mid, f"{idx//10+1}_{idx%10+1}", color='black', ha='center', va='center', fontsize=8)
            idx += 1
            radii_outer[0]=radii_outer[0]+0.125
            radii_outer[1]=radii_outer[1]+0.125
        radii_outer[0]=0.75
        radii_outer[1]=radii_outer[0]+0.125
        # ax.text(theta_mid, 0.875, f"{i+1}", color='black', ha='center', va='center', fontsize=10)

    # 第二圈：6块
    radii_middle = [0.5, 0.75]
    segment_angles_middle=np.linspace(0,2*np.pi,31)
    for i in range(30):
        if idx >= len(data): break
        theta_start = segment_angles_middle[i]
        theta_end = segment_angles_middle[i+1]
        next_data = data[idx+1] if idx+1 < len(data) else data[idx]
        radii_middle[1]=radii_middle[0]+0.125
        for t in range(2):
            fill_gradient(ax,theta_start,theta_end,radii_middle[0],radii_middle[1],data[idx],next_data)
        # ax.fill_between(np.linspace(theta_start, theta_end, 100), radii_outer[0], radii_outer[1],
        #                 color=cmap(norm(data[idx])), edgecolor=None)
            theta_mid = (theta_start + theta_end) / 2
            r_mid = (radii_middle[0] + radii_middle[1]) / 2
            # ax.text(theta_start+(i//3+t/10)*(np.pi)/10, r_mid, f"{idx//10+1}_{idx%10+1}", color='black', ha='center', va='center', fontsize=8)
            idx += 1
            radii_middle[0]=radii_middle[0]+0.125
            radii_middle[1]=radii_middle[1]+0.125
        radii_middle[0]=0.5
        radii_middle[1]=radii_middle[0]+0.125
        # ax.text(theta_mid, 0.625, f"{i+7}", color='black', ha='center', va='center', fontsize=10)

    # 第三圈：4块
    radii_inner = [0.25, 0.5]
    segment_angles_inner=np.linspace(0,2*np.pi,21)
    for i in range(20):
        if idx >= len(data): break
        theta_start = segment_angles_inner[i]
        theta_end = segment_angles_inner[i+1]
        next_data = data[idx+1] if idx+1 < len(data) else data[idx]
        radii_inner[1]=radii_inner[0]+0.125
        for t in range(2):
            fill_gradient(ax,theta_start,theta_end,radii_inner[0],radii_inner[1],data[idx],next_data)
        # ax.fill_between(np.linspace(theta_start, theta_end, 100), radii_outer[0], radii_outer[1],
        #                 color=cmap(norm(data[idx])), edgecolor=None)
            theta_mid = (theta_start + theta_end) / 2
            r_mid = (radii_inner[0] + radii_inner[1]) / 2
            # ax.text(theta_start+(i//3+t/10)*(np.pi)/10, r_mid, f"{idx//10+1}_{idx%10+1}", color='black', ha='center', va='center', fontsize=8)
            idx += 1
            radii_inner[0]=radii_inner[0]+0.125
            radii_inner[1]=radii_inner[1]+0.125
        radii_inner[0]=0.25
        radii_inner[1]=radii_inner[0]+0.125
        # ax.text(theta_mid, 0.375, f"{i+13}", color='black', ha='center', va='center', fontsize=10)

    # 中心圆：1块
    radii_center = [0, 0.25]
    segment_angles_middle=np.linspace(0,2*np.pi,6)
    for i in range(5):
        if idx >= len(data): break
        theta_start = segment_angles_middle[i]
        theta_end = segment_angles_middle[i+1]
        next_data = data[idx+1] if idx+1 < len(data) else data[idx]
        radii_center[1]=radii_center[0]+0.125
        for t in range(2):
            fill_gradient(ax,theta_start,theta_end,radii_center[0],radii_center[1],data[idx],next_data)
        # ax.fill_between(np.linspace(theta_start, theta_end, 100), radii_outer[0], radii_outer[1],
        #                 color=cmap(norm(data[idx])), edgecolor=None)
            theta_mid = (theta_start + theta_end) / 2
            r_mid = (radii_center[0] + radii_center[1]) / 2
            # ax.text(theta_start+(i//3+t/10)*(np.pi)/10, r_mid, f"{idx//10+1}_{idx%10+1}", color='black', ha='center', va='center', fontsize=8)
            idx += 1
            radii_center[0]=radii_center[0]+0.125
            radii_center[1]=radii_center[1]+0.125
        radii_center[0]=0
        radii_center[1]=radii_center[0]+0.125
        # ax.text(theta_mid, 0.375, f"{i+13}", color='black', ha='center', va='center', fontsize=10)
    # if idx < len(data):
    #     radii_center=[0,0.25]
    #     r_center = 0.25
    #     fill_gradient(ax,0,2*np.pi,radii_center[0],radii_center[1],data[-1],data[-1])
    #     ax.text(0, 0, f"{data[idx]:.2f}\n({idx//10+1})", color='black', ha='center', va='center', fontsize=8)
    
    # 设置绘图参数
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    
    # 添加颜色条
    color = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    color.set_array([])
    plt.colorbar(color, ax=ax, fraction=0.046, pad=0.04)
    
    plt.title("Pixel-level Bull's Eye Plot")
    plt.savefig('/data/stu1/liuanqi/vision/2_25_Blood flow.png')
    plt.show()

# 打印数据长度以确认
print(f"Data length: {len(segment_values)}")

# 绘制牛眼图
def main(n):
    i=0
    for i in range(n):
        plot_bulls_eye(segment_values)

# 函数用于记录程序的执行时间
def measure_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return elapsed_time, result


# 函数用于获取当前的内存和CPU使用情况
def get_performance_metrics():
    process = psutil.Process()
    memory_info = process.memory_info().rss / (1024 * 1024)  # 转换为MB
    cpu_percent = process.cpu_percent(interval=1)  # CPU百分比
    gpus = GPUtil.getGPUs()  
    gpu_info = []  
    
    for gpu in gpus:  
        gpu_info.append({  
            'id': gpu.id,  
            'name': gpu.name,  
            'load': gpu.load * 100,  # 使用率（转为百分比）  
            'memory_total': gpu.memoryTotal,  
            'memory_free': gpu.memoryFree,  
            'memory_used': gpu.memoryUsed,  
        })  
    return memory_info, cpu_percent,gpu_info

# 测量程序执行时间和性能
n_values = [1, 5, 10, 20, 30]
times = []
memories = []
cpus = []
gpus=[]

for n in n_values:
    elapsed_time, _ = measure_time(main, n)
    memory, cpu,gpu = get_performance_metrics()
    
    times.append(elapsed_time)
    memories.append(memory)
    cpus.append(cpu)
    gpus.append(gpu)
# 使用Matplotlib和Seaborn进行数据可视化
sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(1, 2, figsize=(15, 10))

# 可视化执行时间
sns.lineplot(x=n_values, y=times, ax=axs[0], marker='o')
axs[0].set_title('Execution Time vs Input Size')
axs[0].set_xlabel('Input Size (n)')
axs[0].set_ylabel('Execution Time (seconds)')
# 在每个点添加坐标标记  
for x, y in zip(n_values, times):  
    axs[0].text(x, y, f'{y:.2f}', fontsize=10, ha='right') 

# 可视化内存使用情况
sns.lineplot(x=n_values, y=memories, ax=axs[1], marker='o', color='g')
axs[1].set_title('Memory Usage vs Input Size')
axs[1].set_xlabel('Input Size (n)')
axs[1].set_ylabel('Memory Usage (MB)')
for x, y in zip(n_values, memories):  
    axs[1].text(x, y, f'{y:.2f}', fontsize=10, ha='right')  

plt.tight_layout()
plt.show()
plt.savefig('/data/stu1/liuanqi/heart_3C/heart_myunet/time/time_3.png')
