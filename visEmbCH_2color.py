# 读取csv文件中的embedding结果，用t-SNE降维，用matplotlib.pyplot绘制成双色散点图，并用箭头连接

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

# 从csv文件中读取数据
data = pd.read_csv('wangwei_szss_emb.csv')  # 将这里的'your_data.csv'替换成你的文件名

# 将字符串转化为列表
data['embedding'] = data['embedding'].apply(ast.literal_eval)

# 将列表转化为numpy数组
embeddings = np.stack(data['embedding'].values)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=3, random_state=100)
tsne_obj= tsne.fit_transform(embeddings)

# 将t-SNE的结果添加到我们的数据框中
data['X'] = tsne_obj[:,0]
data['Y'] = tsne_obj[:,1]

# 创建一个新的图形
plt.figure(figsize=(6, 5))

# 创建色彩映射
cmap_blue = plt.cm.Blues
cmap_red = plt.cm.Reds

# 画出前7行数据，并用由浅及深的蓝色表示
for i in range(round(len(data['X'])/2)):
    plt.scatter(data['X'].values[i], data['Y'].values[i], color=cmap_red(100), label='Group 1' if i == 0 else "")
    # plt.text(data['X'].values[i], data['Y'].values[i], data['Text'].values[i], fontproperties=font)
    if i > 0:  # 从第二个点开始，用箭头连接到上一个点
        plt.arrow(data['X'].values[i - 1], data['Y'].values[i - 1], data['X'].values[i] - data['X'].values[i - 1],
                  data['Y'].values[i] - data['Y'].values[i - 1], shape='full', lw=1, length_includes_head=True,
                  head_width=2, color='red')

# 画出后7行数据，并用由浅及深的绿色表示
for i in range(round(len(data['X'])/2), len(data['X'])):
    plt.scatter(data['X'].values[i], data['Y'].values[i], color=cmap_blue(100), label='Group 2' if i == 7 else "")
    # plt.text(data['X'].values[i], data['Y'].values[i], data['Text'].values[i], fontproperties=font)
    if i > round(len(data['X'])/2):  # 从第二个点开始，用箭头连接到上一个点
        plt.arrow(data['X'].values[i - 1], data['Y'].values[i - 1], data['X'].values[i] - data['X'].values[i - 1],
                  data['Y'].values[i] - data['Y'].values[i - 1], shape='full', lw=1, length_includes_head=True,
                  head_width=2, color='blue')

# 添加图例
# plt.legend()

# 显示图形
plt.show()