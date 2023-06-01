import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import ast

# 从csv文件中读取数据
data = pd.read_csv('noName_lsy_embeddings.csv')  # 将这里的'your_data.csv'替换成你的文件名

# 将字符串转化为列表
data['embedding'] = data['embedding'].apply(ast.literal_eval)

# 将列表转化为numpy数组
embeddings = np.stack(data['embedding'].values)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
tsne_obj= tsne.fit_transform(embeddings)

# 将t-SNE的结果添加到我们的数据框中
data['X'] = tsne_obj[:,0]
data['Y'] = tsne_obj[:,1]

# 创建一个新的图形
plt.figure(figsize=(6, 5))

# 画出前7行数据，并用一种颜色标记
plt.scatter(data['X'].values[:7], data['Y'].values[:7], color='r', label='Group 1')

# 画出后7行数据，并用另一种颜色标记
plt.scatter(data['X'].values[7:14], data['Y'].values[7:14], color='b', label='Group 2')

# 添加图例
plt.legend()

# 显示图形
plt.show()