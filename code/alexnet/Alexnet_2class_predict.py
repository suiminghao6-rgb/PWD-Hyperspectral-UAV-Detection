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
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 高光谱数据集
class HyperspectralDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sample = {'features': features, 'label': label}
        return sample


# AlexNet 构建
class AlexNet1D(nn.Module):
    def __init__(self, input_channels=1, input_sample_points=224, num_classes=2):
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
            raise Exception('输入数据维度错误, 输入维度应为[Batch_size, {}, {}], 实际输入维度为{}'.format(
                self.input_channels, self.input_sample_points, x.size()))
        x = self.features(x)
        x = x.view(-1, 1536)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    set_seed(42)

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    csv_file = r"E:\研一\论文修改\Public Data\datasets\Alexnet\Alexnet band data_Healthy & Disease.csv"
    data = pd.read_csv(csv_file)

    # 检查标签
    labels = data.iloc[:, 0].values
    print(f"标签类别: {np.unique(labels)}")

    dataset = HyperspectralDataset(csv_file)
    input_sample_points = dataset[0]['features'].shape[0]
    print(f"输入波段数: {input_sample_points}")

    # 划分训练集与测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # 定义模型、损失函数和优化器
    model = AlexNet1D(input_channels=1, input_sample_points=input_sample_points, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练模型
    num_epochs = 40
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['features'].to(device), data['label'].to(device)
            inputs = inputs.unsqueeze(1)

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

    print('训练完成')

    # 模型评估
    model.eval()
    accuracy = MulticlassAccuracy(num_classes=2).to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['features'].to(device), data['label'].to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            accuracy.update(preds, labels)

    print(f'测试准确率: {accuracy.compute().item() * 100:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'alexnet_weights.pth')
    print("模型权重已保存")

    # ==================== 优化的预测部分 ====================

    # 对原始影像进行分类预测
    tif_file = r"C:\Users\Shane\Desktop\datatest\456456.tif"
    print(f"正在读取影像: {tif_file}")

    with rasterio.open(tif_file) as src:
        hyperspectral_data = src.read().astype(np.float32)
        profile = src.profile

    # 归一化
    hyperspectral_data = (hyperspectral_data - hyperspectral_data.min()) / (
                hyperspectral_data.max() - hyperspectral_data.min())

    print(f"影像形状: {hyperspectral_data.shape}")
    assert hyperspectral_data.shape[0] == input_sample_points, "输入数据的波段数与模型不匹配"

    # 批量预测 - 大幅加速
    print("开始分类预测（批量处理）...")
    model.eval()

    # 重塑数据: (bands, height, width) -> (height*width, bands)
    bands, height, width = hyperspectral_data.shape
    pixels = hyperspectral_data.reshape(bands, -1).T  # shape: (height*width, bands)

    # 批量预测
    batch_size = 2048  # 可以根据GPU内存调整，越大越快
    predicted_classes = []

    with torch.no_grad():
        for i in tqdm(range(0, len(pixels), batch_size), desc="预测进度"):
            batch = pixels[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)

            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_classes.extend(preds.cpu().numpy())

    # 重塑回原始形状
    predicted_classes = np.array(predicted_classes, dtype=np.uint8).reshape(height, width)

    print("分类完成!")
    print(f"分类结果统计: {np.unique(predicted_classes, return_counts=True)}")

    # 可视化分类结果（黑白二值图，class 0=白色，class 1=黑色）
    plt.figure(figsize=(12, 5))

    # 显示原始影像（使用第一个波段）
    plt.subplot(1, 2, 1)
    with rasterio.open(tif_file) as src:
        show(src.read(1), ax=plt.gca(), title="Original Image", cmap='gray')

    # 显示分类结果（黑白二值图 - class 0=白色，class 1=黑色）
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_classes, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.title("Classification Result")
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.savefig(r"C:\Users\Shane\Desktop\datatest\classification_result.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("可视化结果已保存")

    # 存储分类结果
    profile.update(count=1, dtype=rasterio.uint8)
    output_path = r"C:\Users\Shane\Desktop\datatest\456456result2classes.tif"

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predicted_classes, 1)

    print(f"分类结果已保存至: {output_path}")