import os
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ========== 1. 配置超参数 ==========
class Args:
    data_path = '../boost_converter_dataset.csv'
    use_YPhy = 1        # 使用物理建模值作为输入特征
    batch_size = 250
    epochs = 300
    drop_frac = 0.1
    n_nodes = 32
    n_layers = 3

    lam_phy = 0          # 不使用物理建模误差监督
    lam_cons = 50        # 总损耗守恒项权重
    lam_non_neg = 20     # 非负性惩罚项权重

    val_frac = 0.3
    save_path = './boost_pgnn_model.pth'

args = Args()

# ========== 2. 物理建模函数 ==========
def compute_physical_features(df):
    C = 100e-6
    ESR = 0.1
    R_ind = 0.01
    V_diode = 0.3
    R_ds = 0.01
    t_r = 50e-9
    t_f = 50e-9

    Vin = df['Vin']
    Iin = df['Iin']
    Vout = df['Vout']
    Iout = df['Iout']
    fs = df['fs']
    D = df['D']

    Q_sw = 0.5 * (t_r + t_f) * Iin
    df['P_phy_capacitor'] = ESR * Iout ** 2
    df['P_phy_inductor'] = R_ind * Iin ** 2
    df['P_phy_diode'] = V_diode * Iout
    df['P_phy_mosfet'] = R_ds * Iin ** 2 + fs * Q_sw * Vin
    return df

# ========== 3. 数据加载与标准化 ==========
data_raw = pd.read_csv(args.data_path)
data_with_phy = compute_physical_features(data_raw.copy())

input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
if args.use_YPhy:
    input_features += ['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']

X = data_with_phy[input_features].values.astype(np.float32)
from sklearn.preprocessing import StandardScaler
import joblib

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, 'scaler_X.pkl')

# 输出部分 = [真实器件功率 4项] + [总损耗 1项] + [物理建模功率 4项]
y_true = data_raw[['P_capacitor', 'P_inductor', 'P_diode', 'P_mosfet']].values.astype(np.float32)
y_total = data_raw[['P_loss']].values.astype(np.float32)
y_phy = data_with_phy[['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']].values.astype(np.float32)
y_parts = np.concatenate([y_true, y_total, y_phy], axis=1).astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train_parts, y_val_parts = train_test_split(X_scaled, y_parts, test_size=args.val_frac, random_state=42)

# ========== 4. 数据集类 ==========
import torch
from torch.utils.data import Dataset, DataLoader

class BoostDataset(Dataset):
    def __init__(self, X, y_parts):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_parts = torch.tensor(y_parts, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_parts[idx]

train_loader = DataLoader(BoostDataset(X_train, y_train_parts), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(BoostDataset(X_val, y_val_parts), batch_size=args.batch_size, shuffle=False)

# ========== 5. 模型结构 ResidualPGNN（残差学习） ==========
import torch.nn as nn
import torch.nn.functional as F

class ResidualPGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(ResidualPGNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 4)  # 输出 ΔP 残差值

    def forward(self, x):
        return self.output_layer(self.hidden(x))

model = ResidualPGNN(input_dim=X.shape[1], hidden_dim=args.n_nodes, num_layers=args.n_layers, dropout=args.drop_frac)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 6. 残差结构损失函数：主监督 + 守恒 + 非负 ==========
def residual_pgnn_loss(delta_pred, y_parts):
    y_true = y_parts[:, :4]
    y_total_true = y_parts[:, 4].unsqueeze(1)
    y_phy = y_parts[:, 5:]

    y_pred = delta_pred + y_phy  # 最终预测值 = 残差 + 物理建模值

    loss_main = F.mse_loss(y_pred, y_true)
    loss_cons = F.mse_loss(torch.sum(y_pred, dim=1, keepdim=True), y_total_true)
    loss_non_neg = torch.relu(-y_pred).sum()

    return loss_main + args.lam_cons * loss_cons + args.lam_non_neg * loss_non_neg

# ========== 7. 模型训练过程 ==========
train_losses, val_losses = [], []

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        delta_pred = model(xb)
        loss = residual_pgnn_loss(delta_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            delta_pred = model(xb)
            val_loss += residual_pgnn_loss(delta_pred, yb).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ========== 8. 模型保存 ==========
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")

# ========== 9. 收敛曲线可视化 ==========
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Residual PGNN Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_pgnn_loss_convergence.png")
plt.show()
