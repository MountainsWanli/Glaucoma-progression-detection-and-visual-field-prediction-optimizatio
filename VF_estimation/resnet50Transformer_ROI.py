import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import tqdm
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# Define the CNN+transformer model
import torch
import torch.nn as nn
import torchvision.models as models

class resnet50Transformer(nn.Module):
    def __init__(self, num_classes=59, num_heads=8, num_layers=2, hidden_dim=256):
        super(resnet50Transformer, self).__init__()

        # 使用 torchvision 加载 ResNet50，移除分类头
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉 fc 层，保留到 global avgpool 输出为 (B, 2048, 1, 1)
        # 降维到 Transformer 输入维度
        self.fc_cnn = nn.Linear(2048, hidden_dim)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 回归头
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()  # 输入形状: (B, T, C, H, W)
        # 展平成 (B*T, C, H, W)
        x = x.view(batch_size * timesteps, C, H, W)
        # 提取 CNN 特征 (B*T, 2048, 1, 1)
        features = self.cnn_backbone(x)
        # 去掉多余维度 -> (B*T, 2048)
        features = features.view(batch_size, timesteps, -1)
        # 降维到 Transformer 输入维度 (B, T, hidden_dim)
        features = self.fc_cnn(features)
        # Transformer 编码
        t_out = self.transformer(features)  # (B, T, hidden_dim)
        # 使用最后一个时间步的输出
        out = t_out[:, -1, :]  # (B, hidden_dim)
        # 输出回归值/分类值
        out = self.fc(out)  # (B, num_classes)
        return out

Batch = 8
Epoch = 50
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

data_path = '/home/itaer2/zxy/shixi/project_1/'
excel_data = pd.read_excel(data_path + 'vf_3.xlsx', sheet_name=0)
excel_data = np.array(excel_data)
ids = excel_data[1:, 16]
image_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])  # (432, 432)

input_format = {0: 'CFPs/', 1: 'ROI_images_3/', 2: 'Annotated Images/'}
format_index = 1  # format_index can be 0, 1, 2
images = []
for count, index in enumerate(ids):
    images.append(image_transform(Image.open(data_path + input_format[format_index] + index)))
images = torch.stack(images, dim=0)
labels = np.delete(excel_data[1:, 19:], (40, 51), axis=1).astype(np.float64)

np.random.seed(0)
random_indices = np.random.permutation(len(images))
split_index = int(len(images) * 0.8)
train_data, train_label = images[random_indices[:split_index]], labels[random_indices[:split_index]]
test_data, test_label = images[random_indices[split_index:]], labels[random_indices[split_index:]]

scaler = preprocessing.StandardScaler()
scaler.fit(train_label)
train_label = scaler.transform(train_label)
train_label, test_label = torch.Tensor(train_label), torch.Tensor(test_label)
train_set = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=Batch, shuffle=True)

val_index = len(test_data)
val_set = torch.utils.data.TensorDataset(test_data[:val_index], test_label[:val_index])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=Batch, shuffle=False)
test_set = torch.utils.data.TensorDataset(test_data[:val_index], test_label[:val_index])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=Batch, shuffle=False)

model = resnet50Transformer()
model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=(len(train_set) // Batch + 1 if len(train_set) % Batch else len(train_set) // Batch) * Epoch)
loss_list_epoch = []
best_loss=float('inf')
for epoch in tqdm.tqdm(range(Epoch)):
    loss_list_batch = []
    for data in train_loader:
        image, label = data[0].to(device), data[1].to(device)
        image=image.unsqueeze(1)
        pre = model(image)
        loss = loss_function(pre, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if best_loss>loss:
            torch.save(model.state_dict(),'./checkpoints/resnet50Transformer_ROI/best.pt')
        best_loss=loss
        loss_list_batch.append(loss.data.item() * len(pre) / len(train_set))
    loss_list_epoch.append(sum(loss_list_batch))

plt.figure()
plt.plot(loss_list_epoch)
plt.show()

# load model from the provided checkpoints
# model.load_state_dict(torch.load('./checkpoints/' + input_format[format_index] + 'model.pt'))
model.load_state_dict(torch.load('./checkpoints/resnet50Transformer_ROI/' + 'best.pt'))
model.eval()
test_pre = []
with torch.no_grad():
    for data in test_loader:
        image = data[0].to(device)
        image = image.unsqueeze(1)  # Add sequence dimension (batch, seq_len=1, C, H, W)
        pre = model(image)
        test_pre.append(pre)
test_pre = torch.cat(test_pre)
test_pre = scaler.inverse_transform(test_pre.cpu().numpy())
test_pre = np.clip(test_pre, -1, 35)
rmse = mean_squared_error(test_label[:val_index], test_pre, squared=False)
mae = mean_absolute_error(test_label[:val_index], test_pre)
r2 = r2_score(test_label[:val_index], test_pre)

print('Results:', 'RMSE:', (round(rmse, 3)), '\t',
      'MAE', (round(mae, 3)), '\t',
      'R2', (round(r2, 3)))
