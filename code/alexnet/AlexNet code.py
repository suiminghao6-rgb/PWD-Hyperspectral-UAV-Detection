import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
import random
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 高光谱数据集
class HyperspectralDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'label': self.labels[idx]}
        return sample

# AlexNet 构建
class AlexNet1D(nn.Module):
    def __init__(self, input_channels=1, input_sample_points=224, num_classes=4):  # 根据标签类别数量定义num_classes
        super(AlexNet1D, self).__init__()
        self.input_channels = input_channels
        self.input_sample_points = input_sample_points
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(6),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        if x.size(1) != self.input_channels or x.size(2) != self.input_sample_points:
            raise Exception('输入数据维度错误, 输入维度应为[Batch_size, {}, {}], 实际输入维度为{}'.format(self.input_channels, self.input_sample_points, x.size()))
        x = self.features(x)
        x = x.view(-1, 1536)
        x = self.classifier(x)
        return x

# 加载数据集
csv_file = r"G:\Users\YS\Desktop\松材线虫\输入数据\Alexnet\Alexnet band data_Healthy & Disease.csv"
data = pd.read_csv(csv_file)

# 检查标签
labels = data.iloc[:, 0].values
print(np.unique(labels))  # 确认标签数量

dataset = HyperspectralDataset(csv_file)

# 确定输入特征的长度
input_sample_points = dataset[0]['features'].shape[0]

# 划分训练集与测试集集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型、损失函数和优化器
model = AlexNet1D(input_channels=1, input_sample_points=input_sample_points, num_classes=4)  # 依据标签数量修改num_classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['features'], data['label']
        inputs = inputs.unsqueeze(1)  # 增加一个通道维度
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
    scheduler.step()

print('Finished Training')

# 模型评估
model.eval()
accuracy = MulticlassAccuracy(num_classes=2)  # 依据标签数量修改num_classes
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data['features'], data['label']
        inputs = inputs.unsqueeze(1)  # 增加一个通道维度
        labels = labels.long()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy.update(preds, labels)

print(f'Test Accuracy: {accuracy.compute().item() * 100:.2f}%')
torch.save(model,'alexnet.pth')

# 对原始影像进行分类预测
tif_file = ""
with rasterio.open(tif_file) as src:
    hyperspectral_data = src.read().astype(np.float32)
    profile = src.profile

assert hyperspectral_data.shape[0] == input_sample_points, "输入数据的波段数与模型不匹配"

# 进行分类预测
model.eval()
height, width = hyperspectral_data.shape[1], hyperspectral_data.shape[2]
predicted_classes = np.zeros((height, width), dtype=np.uint8)

with torch.no_grad():
    for i in range(height):
        for j in range(width):
            pixel_data = hyperspectral_data[:, i, j]
            inputs = torch.tensor(pixel_data).unsqueeze(0).unsqueeze(0)  # 调整输入维度为[1, 1, 波段数]
            outputs = model(inputs)
            _, predicted_class = torch.max(outputs, 1)
            predicted_classes[i, j] = predicted_class.item()

print(f'Predicted Class Matrix: \n{predicted_classes}')

# 预测结果可视化
with rasterio.open(tif_file) as src:
    show(src, title="Hyperspectral Data")
plt.imshow(predicted_classes, cmap='tab20', interpolation='nearest')
plt.title("Predicted Classes")
plt.colorbar()
plt.show()

# 存储分类结果
profile.update(count=1, dtype=rasterio.uint8)

with rasterio.open("", 'w', **profile) as dst:
    dst.write(predicted_classes, 1)
