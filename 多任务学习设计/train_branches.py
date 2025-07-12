import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# ========== 1. 参数配置 ==========
class Args:
    data_path = '../boost_converter_dataset.csv'  # 数据文件路径
    batch_size = 128                            # 每批训练样本数
    epochs = 200                                # 训练轮数
    hidden_dim = 32                             # 隐藏层神经元个数
    lr = 1e-3                                   # 学习率
    val_frac = 0.3                              # 验证集比例
    save_dir = 'branchnet_weights'  # 模型保存目录
    target_name = 'P_mosfet'                 # 当前训练的目标变量
    lam_relative = 30                           # 相对误差损失权重
args = Args()

# 创建保存模型的文件夹
os.makedirs(args.save_dir, exist_ok=True)

# ========== 2. 加载与预处理数据 ==========
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
data_raw = pd.read_csv(args.data_path)

# 提取输入特征和目标变量
X = data_raw[input_features].values.astype(np.float32)
y = data_raw[[args.target_name]].values.astype(np.float32)

# 标准化输入特征
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, f'{args.save_dir}/scaler_X.pkl')

# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=args.val_frac, random_state=42)

# 自定义数据集类
class BoostBranchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(BoostBranchDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(BoostBranchDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

# ========== 3. 构建神经网络模型 ==========
class BranchNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出单个预测值
        )

    def forward(self, x):
        return self.model(x)

model = BranchNet(input_dim=X.shape[1], hidden_dim=args.hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ========== 4. 自定义损失函数 ==========
def capacitor_loss(y_pred, y_true, alpha=args.lam_relative, eps=1e-6):
    """
    总损失 = 均方误差 + alpha * 相对误差
    """
    mse = nn.functional.mse_loss(y_pred, y_true)

    rel_error = torch.abs((y_pred - y_true) / (y_true + eps))
    rel_loss = torch.mean(rel_error)

    return mse + alpha * rel_loss

# ========== 5. 模型训练过程 ==========
train_losses, val_losses = [], []

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = capacitor_loss(y_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            y_pred = model(xb)
            loss = capacitor_loss(y_pred, yb)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"第 {epoch:03d} 轮: 训练损失 = {train_losses[-1]:.4f}, 验证损失 = {val_losses[-1]:.4f}")

# ========== 6. 保存模型 ==========
model_path = os.path.join(args.save_dir, f'branchnet_{args.target_name}.pth')
torch.save(model.state_dict(), model_path)
print(f"模型已保存至 {model_path}")

# ========== 7. 绘制损失曲线 ==========
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title(f'{args.target_name} 神经网络训练曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, f'loss_curve_{args.target_name}.png'))
plt.show()




import torch
import pandas as pd
import numpy as np
import joblib
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ========== 1. 参数设置 ==========
model_path = 'branchnet_weights/branchnet_P_capacitor.pth'
scaler_path = 'branchnet_weights/scaler_X.pkl'
data_path = '../boost_converter_dataset.csv'
target_col = 'P_capacitor'

# ========== 2. 加载数据 ==========
df = pd.read_csv(data_path)
X = df[['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']].values.astype(np.float32)
y_true = df[[target_col]].values.astype(np.float32)

scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ========== 3. 模型结构 ==========
model = BranchNet(input_dim=X.shape[1], hidden_dim=32)
model.load_state_dict(torch.load(model_path))
model.eval()

# ========== 4. 预测 ==========
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# ========== 5. 分析结果 ==========
abs_error = np.abs(y_pred - y_true)
rel_error = abs_error / (y_true + 1e-6)

print(f"P_capacitor 预测性能：")
print(f"  - MAE: {mean_absolute_error(y_true, y_pred):.6f}")
print(f"  - RMSE: {mean_squared_error(y_true, y_pred):.6f}")
print(f"  - 平均相对误差: {np.mean(rel_error):.4%}")
print(f"  - 最大相对误差: {np.max(rel_error):.4%}")

# ========== 6. 可视化 ==========
plt.figure(figsize=(8, 5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('P_capacitor')
plt.title('P_capacitor: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('branchnet_p_capacitor_pred_vs_true.png')
plt.show()

# 相对误差分布
plt.figure(figsize=(6, 4))
plt.hist(rel_error, bins=50, alpha=0.7, color='orange')
plt.xlabel('Relative Error')
plt.ylabel('Count')
plt.title('Relative Error Distribution (P_capacitor)')
plt.tight_layout()
plt.savefig('branchnet_p_capacitor_rel_error_hist.png')
plt.show()
