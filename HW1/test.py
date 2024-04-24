import numpy as np
from model import NeuralNetwork

# 读取数据
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

# model
input_size = X_test.shape[1]
hidden_size = 256
output_size = 10
nn = NeuralNetwork(input_size, hidden_size, output_size, activation='relu',weights_file='./data/best_model.npz')

# 在测试集上计算损失和准确率
y_pred_test = nn.forward(X_test)
test_loss = nn.compute_loss(y_pred_test, y_test)
test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
print('Test Loss:', test_loss, 'Test Accuracy:', test_acc)