# Updated PGNN Training Script with Multi-Head Output, Residual Learning, and Enhanced Loss

import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# ========== 1. 配置超参数 ==========
class Args:
    data_path = '../boost_converter_dataset.csv'
    use_YPhy = 1
    batch_size = 250
    epochs = 300
    drop_frac = 0.1

    n_nodes = 64     # 增加隐藏节点提升拟合能力
    n_layers = 4     # 更深层模型

    lam_phy = 0      # 物理先验精度仍较低
    lam_cons = 50
    lam_non_neg = 20
    lam_main = 1.0   # 主监督项权重

    val_frac = 0.3
    save_path = './boost_multi_residual_pgnn_model.pth'
args = Args()

# ========== 2. 特征工程：添加 ΔP_phy 残差信息 ==========
def compute_physical_features_train(df):
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

    # 残差目标值（仅用于训练监督）
    df['dP_capacitor'] = df['P_capacitor'] - df['P_phy_capacitor']
    df['dP_inductor'] = df['P_inductor'] - df['P_phy_inductor']
    df['dP_diode'] = df['P_diode'] - df['P_phy_diode']
    df['dP_mosfet'] = df['P_mosfet'] - df['P_phy_mosfet']

    return df

# ========== 3. 加载数据 ==========
data_raw = pd.read_csv(args.data_path)
data_all = compute_physical_features_train(data_raw.copy())

input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D',
                  'P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']
X = data_all[input_features].values.astype(np.float32)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, 'scaler_X.pkl')

# 主监督标签为残差 ΔP
y_delta = data_all[['dP_capacitor', 'dP_inductor', 'dP_diode', 'dP_mosfet']].values.astype(np.float32)
# 总损耗和物理预测值用于能量守恒和辅助监督
y_total = data_all[['P_loss']].values.astype(np.float32)
y_phy = data_all[['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']].values.astype(np.float32)

# 拼接标签
y_parts = np.concatenate([y_delta, y_total, y_phy], axis=1).astype(np.float32)

X_train, X_val, y_train_parts, y_val_parts = train_test_split(X_scaled, y_parts, test_size=args.val_frac, random_state=42)

# ========== 4. 数据集 ==========
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

# ========== 5. 构建多头残差学习模型 ==========
class ResidualPGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.shared_layers = nn.Sequential(*layers)

        # 每个器件独立输出 ΔP
        self.out_cap = nn.Linear(hidden_dim, 1)
        self.out_ind = nn.Linear(hidden_dim, 1)
        self.out_dio = nn.Linear(hidden_dim, 1)
        self.out_mos = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared_layers(x)
        return torch.cat([
            self.out_cap(h),
            self.out_ind(h),
            self.out_dio(h),
            self.out_mos(h)
        ], dim=1)

model = ResidualPGNN(input_dim=X.shape[1], hidden_dim=args.n_nodes, num_layers=args.n_layers, dropout=args.drop_frac)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 6. 自定义损失函数 ==========
def residual_pgnn_loss(y_pred, y_parts):
    delta_true = y_parts[:, :4]          # 主监督目标 ΔP_true
    y_total_true = y_parts[:, 4].unsqueeze(1)
    y_phy = y_parts[:, 5:9]              # P_phy

    y_pred_full = y_pred + y_phy         # 模型预测 + 物理模型 = 最终功率

    # 主监督残差项
    main_loss = F.mse_loss(y_pred, delta_true)
    # 守恒损失
    cons_loss = F.mse_loss(torch.sum(y_pred_full, dim=1, keepdim=True), y_total_true)
    # 非负惩罚
    non_neg_penalty = torch.relu(-y_pred_full).sum()

    return (args.lam_main * main_loss +
            args.lam_cons * cons_loss +
            args.lam_non_neg * non_neg_penalty)

# ========== 7. 模型训练 ==========
train_losses, val_losses = [], []
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = residual_pgnn_loss(y_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            y_pred = model(xb)
            val_loss += residual_pgnn_loss(y_pred, yb).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ========== 8. 保存模型 ==========
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")

# ========== 9. 收敛曲线 ==========
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Multi Residual PGNN Loss Convergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("multi_residual_pgnn_loss_curve.png")
plt.show()