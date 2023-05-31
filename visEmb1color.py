# 读取csv文件中的embedding结果，用t-SNE降维，用matplotlib.pyplot绘制成单色散点图，保存到Filename_string文件

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = 'Arial Unicode MS'
font = FontProperties(fname="/System/Library/Fonts/supplemental/Arial Unicode.ttf", size=14)

# 从csv文件中读取数据
fileName = '4mindwords10Lan' + '_emb.csv'
data = pd.read_csv(fileName)  # 将这里的'your_data.csv'替换成你的文件名

# 将字符串转化为列表
data['embedding'] = data['embedding'].apply(ast.literal_eval)

# 将列表转化为numpy数组
embeddings = np.stack(data['embedding'].values)

# 使用t-SNE进行降维
PerValue = 5
RandValue = 40
tsne = TSNE(n_components=2, perplexity=PerValue, random_state=RandValue)
tsne_obj= tsne.fit_transform(embeddings)

# 将t-SNE的结果添加到我们的数据框中
data['X'] = tsne_obj[:,0]
data['Y'] = tsne_obj[:,1]

# 创建一个新的图形
plt.figure(figsize=(6, 5))

# 创建色彩映射
cmap_blue = plt.cm.Blues
cmap_red = plt.cm.Reds

# 画出前10行数据，并用红色表示
for i in range(len(data['X'])):
    plt.scatter(data['X'].values[i], data['Y'].values[i], color=cmap_red(100), label='Group 1' if i == 0 else "")
    plt.text(data['X'].values[i], data['Y'].values[i], data['Text'].values[i], fontproperties=font, fontsize="9")
    #if i > 0:  # 从第二个点开始，用箭头连接到上一个点
    #    plt.arrow(data['X'].values[i - 1], data['Y'].values[i - 1], data['X'].values[i] - data['X'].values[i - 1],
    #              data['Y'].values[i] - data['Y'].values[i - 1], shape='full', lw=1, length_includes_head=True,
    #              head_width=2, color='red')

# 画出第11个数据，并用蓝色表示
#for i in range(10, 11):
#    plt.scatter(data['X'].values[i], data['Y'].values[i], color=cmap_blue(100), label='Group 2' if i == 10 else "")
#    plt.text(data['X'].values[i], data['Y'].values[i], data['Text'].values[i], fontproperties=font)
    #if i > round(len(data['X'])/2):  # 从第二个点开始，用箭头连接到上一个点
    #    plt.arrow(data['X'].values[i - 1], data['Y'].values[i - 1], data['X'].values[i] - data['X'].values[i - 1],
    #              data['Y'].values[i] - data['Y'].values[i - 1], shape='full', lw=1, length_includes_head=True,
    #              head_width=2, color='blue')

# 添加图例
plt.legend()

# 保存图形到文件
Filename_string = "fig_" + fileName + str(PerValue) + "_" + str(RandValue) + ".jpg"
plt.savefig(Filename_string, dpi=300, bbox_inches='tight')

# 显示图形
plt.show()