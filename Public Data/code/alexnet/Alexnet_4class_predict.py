# 首先设置环境变量解决OpenMP冲突
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
from matplotlib.colors import ListedColormap
import rasterio


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# 定义高光谱数据集
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


# 修改后的 AlexNet
class AlexNet1D(nn.Module):
    def __init__(self, input_channels=1, input_sample_points=240, num_classes=4):
        super(AlexNet1D, self).__init__()
        self.input_channels = input_channels
        self.input_sample_points = input_sample_points
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(6),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        if x.size(1) != self.input_channels or x.size(2) != self.input_sample_points:
            raise Exception(
                '输入数据维度错误, 输入维度应为[Batch_size, {}, {}], 实际输入维度为{}'.format(self.input_channels,
                                                                                              self.input_sample_points,
                                                                                              x.size()))
        x = self.features(x)
        x = x.view(-1, 1536)
        x = self.classifier(x)
        return x


# 加载数据集
csv_file = r"E:\研一\论文修改\Public Data\datasets\Alexnet\Alexnet band data_Healthy & Disease & Shadow & Edge.csv"
data = pd.read_csv(csv_file)

# 检查标签
labels = data.iloc[:, 0].values
print(np.unique(labels))

dataset = HyperspectralDataset(csv_file)

# 确定输入特征的长度
input_sample_points = dataset[0]['features'].shape[0]
print(f"输入波段数: {input_sample_points}")

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = AlexNet1D(input_channels=1, input_sample_points=input_sample_points, num_classes=4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 训练模型
num_epochs = 80
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['features'], data['label']
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.long().to(device)

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

# 评估模型
model.eval()
accuracy = MulticlassAccuracy(num_classes=4).to(device)
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data['features'], data['label']
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.long().to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy.update(preds, labels)

print(f'Test Accuracy: {accuracy.compute().item() * 100:.2f}%')

# 保存模型
model_save_path = r"E:\研一\论文修改\Public Data\输出\alexnet_4class_ok_unfilter.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到: {model_save_path}")

# ==================== 优化后的预测部分 ====================
print("\n开始预测TIF影像...")

# 对TIF数据进行分类预测 - 优化版本
tif_file = r"C:\Users\Shane\Desktop\datatest\456456.tif"
with rasterio.open(tif_file) as src:
    hyperspectral_data = src.read().astype(np.float32)
    profile = src.profile

print(f"TIF影像尺寸: {hyperspectral_data.shape}")
print(f"波段数: {hyperspectral_data.shape[0]}")
print(f"像素值范围: [{hyperspectral_data.min():.2f}, {hyperspectral_data.max():.2f}]")

# 确保数据维度与模型输入一致
assert hyperspectral_data.shape[
           0] == input_sample_points, f"输入数据的波段数({hyperspectral_data.shape[0]})与模型不匹配({input_sample_points})"

# 进行批量分类预测 - 优化版
model.eval()
height, width = hyperspectral_data.shape[1], hyperspectral_data.shape[2]

# 将数据重塑为 [height*width, channels]
pixels = hyperspectral_data.reshape(input_sample_points, -1).T
pixels_tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(1)

# 批量预测
batch_size = 1024
predicted_classes = np.zeros(height * width, dtype=np.uint8)

print("开始批量预测...")
with torch.no_grad():
    for i in range(0, len(pixels_tensor), batch_size):
        batch = pixels_tensor[i:i + batch_size].to(device)
        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)
        predicted_classes[i:i + batch_size] = predicted.cpu().numpy()

        if (i // batch_size) % 10 == 0:
            progress = (i / len(pixels_tensor)) * 100
            print(f"预测进度: {progress:.1f}%")

predicted_classes = predicted_classes.reshape(height, width)

print(f'预测完成！')
print(f'预测结果尺寸: {predicted_classes.shape}')

# 统计各类别数量
unique, counts = np.unique(predicted_classes, return_counts=True)
print("\n类别分布:")
class_names = ['Healthy', 'Disease', 'Shadow', 'Edge']
for cls, count in zip(unique, counts):
    percentage = (count / (height * width)) * 100
    print(f"  类别 {cls} ({class_names[cls]}): {count} 像素 ({percentage:.2f}%)")

# 可视化预测结果 - 使用新的颜色方案
colors = ['#8000FF', '#00FFFF', '#00FF00', '#FF0000']  # 深紫色、青色、绿色、红色
cmap = ListedColormap(colors)

plt.figure(figsize=(12, 10))
im = plt.imshow(predicted_classes, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
plt.title("Predicted Classes\n0:Healthy(Purple) | 1:Disease(Cyan) | 2:Shadow(Green) | 3:Edge(Red)",
          fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ticks=[0, 1, 2, 3], shrink=0.8)
cbar.ax.set_yticklabels(['0-Healthy', '1-Disease', '2-Shadow', '3-Edge'])
plt.tight_layout()
plt.show()

# 保存结果
profile.update(count=1, dtype=rasterio.uint8)
output_file = r"C:\Users\Shane\Desktop\datatest\4564564result4classes.tif"
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(predicted_classes, 1)

print(f"\n分类结果已保存到: {output_file}")