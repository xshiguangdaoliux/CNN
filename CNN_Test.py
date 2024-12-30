import torch#PyTorch库的核心，用于处理张量和实现深度学习。
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 检查CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
batch_size = 32 #每次训练中处理的样本数量。
epochs = 10#训练数据的完整迭代次数
learning_rate = 0.001 #优化器的学习率，用于控制模型参数的更新步长。


# 数据预处理与增强
"""transforms.Compose: 将多个预处理步骤按顺序组合。
Resize((128, 128)): 将所有图像大小调整为128x128。
RandomHorizontalFlip(): 随机水平翻转图像（数据增强）。
ToTensor(): 将图像转为张量，并将像素值归一化为[0, 1]。
Normalize(mean, std): 使用给定均值和标准差对图像数据进行标准化"""
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

"""CIFAR-10: 一个常用的小型图像分类数据集，包含10个类别，每张图像大小为32x32像素。
train=True: 表示加载训练数据集；train=False表示加载测试数据集。
transform=transform: 指定前面定义的预处理方法。
download=True: 如果本地没有数据集，会从互联网下载。"""
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""分别是两个卷积层。
Conv2d(in_channels, out_channels, kernel_size, padding)：
in_channels=3 表示输入图像有3个通道（RGB）。
out_channels=32 和 64 表示输出特征图的通道数。
kernel_size=3 是卷积核大小为3x3。
padding=1 是在边界处填充1个像素，保证卷积后特征图大小不变。
"""
"""最大池化层，MaxPool2d(kernel_size=2, stride=2)。
池化窗口大小为2x2，步长为2，作用是减少特征图的尺寸，降低计算量。
"""
"""fc1 将展平后的特征转为128维向量。
fc2 输出为10维（对应CIFAR-10的10个类别）。"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()#Input size=3×128×128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)#输出特征图大小为：32×128×128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    """描述了数据通过模型的前向传播过程：
卷积 + 激活 + 池化+展平+全连接层+输出"""
    """总结每层后的数据维度
输入：3×128×128
卷积1（conv1）：32×128×128。
池化1（pool）：32×64×64。
卷积2（conv2）：64×64×64。
池化2（pool）：64×32×32。
展平：65536。
全连接1（fc1）：128。
全连接2（fc2）：10"""
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()#交叉熵损失（CrossEntropyLoss），适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""模型训练模式:
调用model.train()，启用训练模式（如启用Dropout）。
迭代过程:
遍历训练集train_loader的每个批次（images, labels）。
将数据加载到设备（images.to(device)）。
前向传播:
将输入数据传入模型，得到输出。
计算损失值（loss）。
反向传播:
清零梯度（optimizer.zero_grad()）。
计算梯度（loss.backward()）。
更新参数（optimizer.step()）。
记录损失:
累积每个批次的损失，计算平均损失。"""
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

"""模型测试模式:
调用model.eval()，禁用训练特性（如Dropout）。
评估过程:
遍历测试集test_loader的每个批次。
在torch.no_grad()上下文中，停止梯度计算以节省内存和计算资源。
使用torch.max获取预测的类别。
比较预测值与真实标签，计算准确率。"""
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_model()
    evaluate_model()
