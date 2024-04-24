import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
from utils import update_learning_rate

# 读取数据
X_train = np.load('./data/X_train.npy')
y_train = np.load('./data/y_train.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

# model
# 训练模型
# 初始化神经网络
input_size = X_train.shape[1]
hidden_size = 64
output_size = 10
nn = NeuralNetwork(input_size, hidden_size, output_size, activation='relu')

# 参数设置
epochs = 50
batch_size = 16
learning_rate = 0.001
reg_lambda = 0.001
best_test_loss = np.inf
best_test_acc = 0

# 可视化训练过程
train_losses = []
test_losses = []
test_accuracies = []

# 训练模型
for epoch in range(epochs):
    learning_rate = update_learning_rate(learning_rate, epoch)

    # 随机打乱数据
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = []
    # Mini-batch训练
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # 前向传播
        y_pred = nn.forward(X_batch)

        # 计算损失
        loss = nn.compute_loss(y_pred, y_batch)
        epoch_loss.append(loss)

        # 反向传播
        nn.backward(X_batch, y_batch, lr=learning_rate)

    # 计算平均损失
    train_loss = np.mean(epoch_loss)
    train_losses.append(train_loss)

    # 在验证集上计算损失和准确率
    y_pred_test = nn.forward(X_test)
    test_loss = nn.compute_loss(y_pred_test, y_test)
    test_losses.append(test_loss)
    test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
    test_accuracies.append(test_acc)

    # 保存最优模型
    if test_acc> best_test_acc:
        best_test_accuracy = test_acc
        np.savez('data/best_model.npz', weights1=nn.weights1, bias1=nn.bias1, weights2=nn.weights2, bias2=nn.bias2)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, test Loss: {test_loss:.4f}, test Accuracy: {test_acc:.4f}')

# 可视化训练过程
# 绘制训练和验证损失
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制验证准确率
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Test Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
