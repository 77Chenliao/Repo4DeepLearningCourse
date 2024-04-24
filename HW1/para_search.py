import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import NeuralNetwork
from sklearn.model_selection import ParameterSampler
import json
from tqdm import tqdm

# 加载数据
X_train = np.load('./data/X_train.npy')
y_train = np.load('./data/y_train.npy')
X_val = np.load('./data/X_test.npy')
y_val = np.load('./data/y_test.npy')

# 定义超参数空间
param_distributions = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'hidden_size': [64, 128, 256, 512],
    'reg_lambda': [0.001, 0.01, 0.1, 1.0]
}

# 初始化最佳值记录器和性能记录列表
best_val_loss = np.inf
best_params = {}
all_results = []

# 创建参数的随机组合
n_iter = 40
epochs = 50
batch_size = 32
param_list = list(ParameterSampler(param_distributions, n_iter=n_iter))
for params in tqdm(param_list):
    # 创建神经网络模型
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=params['hidden_size'],
                       output_size=10, activation='relu', reg_lambda=params['reg_lambda'])

    # 训练模型
    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            y_pred = nn.forward(X_batch)
            loss = nn.compute_loss(y_pred, y_batch)
            nn.backward(X_batch, y_batch, lr=params['learning_rate'])

        # 在验证集上评估
        y_pred_val = nn.forward(X_val)
        val_loss = nn.compute_loss(y_pred_val, y_val)
        val_acc = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1))

    all_results.append({'params': params, 'val_acc': val_acc})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params

# 3D 可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 提取每个参数和准确率
xs = [r['params']['learning_rate'] for r in all_results]
ys = [r['params']['hidden_size'] for r in all_results]
zs = [r['params']['reg_lambda'] for r in all_results]
colors = [r['val_acc'] for r in all_results]

p = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', depthshade=True)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Hidden Size')
ax.set_zlabel('Reg Lambda')
fig.colorbar(p, ax=ax, label='Validation Accuracy')

plt.show()

# 输出和保存最佳参数
print(f"Best validation loss: {best_val_loss}")
print(f"Best parameters: {best_params}")
with open('./data/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
