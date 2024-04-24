from utils import sigmoid, sigmoid_derivative, relu, relu_derivative
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid',reg_lambda=0.01,weights_file=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.reg_lambda = reg_lambda
        
        # 初始化权重和偏差
        if weights_file:
            weights = np.load(weights_file)
            self.weights1 = weights['weights1']
            self.bias1 = weights['bias1']
            self.weights2 = weights['weights2']
            self.bias2 = weights['bias2']
        else:
            self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
            self.bias1 = np.zeros((1, hidden_size))
            self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
            self.bias2 = np.zeros((1, output_size))
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation_func = relu
            self.activation_derivative = relu_derivative
        
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.activation_func(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.a2 /= np.sum(self.a2, axis=1, keepdims=True)  # Softmax
        return self.a2
    
    def backward(self, X, y, lr=0.01):
        # 计算输出层梯度
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) + self.reg_lambda * self.weights2 
        db2 = np.sum(delta2, axis=0)
        
        # 计算隐藏层梯度
        delta1 = np.dot(delta2, self.weights2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) + self.reg_lambda * self.weights1
        db1 = np.sum(delta1, axis=0)
        
        # 更新参数
        self.weights1 -= lr * dW1
        self.bias1 -= lr * db1
        self.weights2 -= lr * dW2
        self.bias2 -= lr * db2
    def compute_loss(self, y_pred, y_true,reg_lambda=0.01):
        m = y_true.shape[0]
        log_likelihoods = -np.log(y_pred[range(m), y_true.argmax(axis=1)]+1e-8)
        loss = np.sum(log_likelihoods) / m # Cross Entropy Loss
        Loss = loss + self.reg_lambda / 2 * (np.sum(np.square(self.weights1)) + np.sum(np.square(self.weights2))) # L2 Regularization
        return Loss