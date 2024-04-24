import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 更新学习率
def update_learning_rate(initial_lr, epoch, decay_rate=0.95):
    # 指数衰减学习率
    return initial_lr * (decay_rate ** epoch)