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
    use_YPhy = 1
    batch_size = 250
    epochs = 300
    drop_frac = 0.1

    n_nodes = 32
    n_layers = 3

    # 如果未来可以进一步提高物理建模的精确度的话 可以使用
    lam_phy = 0  # 物理先验建模精确度比较低 并不适合使用 暂时设置为0
    # 引导项 而非主监督项

    lam_cons = 50
    lam_non_neg = 20

    val_frac = 0.3
    save_path = './boost_pgnn_model.pth'
args = Args()

# ========== 2. 加载数据和特征工程 ==========
def compute_physical_features(df):
    # 元器件参数（来自 setup_circuit_parameters.m）
    C = 100e-6            # 电容的电容量
    ESR = 0.1             # 电容的等效串联电阻
    R_ind = 0.01          # 电感的串联电阻
    V_diode = 0.3         # 二极管的正向导通压降
    R_ds = 0.01           # MOSFET导通时的漏源极电阻
    t_r = 50e-9           # MOSFET上升时间 [s]
    t_f = 50e-9           # MOSFET下降时间 [s]

    # 输入特征（6维度）提取
    Vin = df['Vin']
    Iin = df['Iin']
    Vout = df['Vout']
    Iout = df['Iout']
    fs = df['fs']
    D = df['D']

    # 动态计算 Q_sw（单位 C）
    Q_sw = 0.5 * (t_r + t_f) * Iin

    # 物理损耗建模
    df['P_phy_capacitor'] = ESR * Iout ** 2
    df['P_phy_inductor'] = R_ind * Iin ** 2
    df['P_phy_diode'] = V_diode * Iout
    df['P_phy_mosfet'] = R_ds * Iin ** 2 + fs * Q_sw * Vin
    return df

data_raw = pd.read_csv(args.data_path)
data_with_phy = compute_physical_features(data_raw.copy())

input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
if args.use_YPhy:
    input_features += ['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']

X = data_with_phy[input_features].values.astype(np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # 替换原始的 X
joblib.dump(scaler_X, 'scaler_X.pkl')


y_true = data_raw[['P_capacitor', 'P_inductor', 'P_diode', 'P_mosfet']].values.astype(np.float32)
y_total = data_raw[['P_loss']].values.astype(np.float32)
y_phy = data_with_phy[['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']].values.astype(np.float32)

y_parts = np.concatenate([y_true, y_total, y_phy], axis=1).astype(np.float32)

X_train, X_val, y_train_parts, y_val_parts = train_test_split(X_scaled, y_parts, test_size=args.val_frac, random_state=42)

# ========== 3. 数据集 ==========
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

# ========== 4. 构建模型 ==========
class PGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(PGNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        x = self.hidden(x)
        return self.output_layer(x)

model = PGNN(input_dim=X.shape[1], hidden_dim=args.n_nodes, num_layers=args.n_layers, dropout=args.drop_frac)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 5. 定义损失函数 ==========
def pgnn_loss(y_pred, y_parts):
    y_true = y_parts[:, :4]
    y_total_true = y_parts[:, 4].unsqueeze(1)
    y_phy_parts = y_parts[:, 5:]

    # 数据经验损失
    mse = F.mse_loss(y_pred, y_true)
    # 物理先验建模损失 --- 实际上物理建模精确度较低（一阶近似）并不适合使用 能够作为界限
    phy_loss = F.mse_loss(y_pred, y_phy_parts)
    # 物理一致性 能量守恒限制损失 （重要）
    cons_loss = F.mse_loss(torch.sum(y_pred, dim=1, keepdim=True), y_total_true)
    # 非负性损失：逐分量惩罚所有负值
    non_negative_penalty = torch.relu(-y_pred)  # 对每个负值进行惩罚，非负值为0
    non_negative_loss = non_negative_penalty.sum()  # 对所有元素求和，强惩罚每个负预测

    return (mse +
            args.lam_phy * phy_loss +
            args.lam_cons * cons_loss +
            args.lam_non_neg * non_negative_loss)

# ========== 6. 训练过程 ==========
train_losses = []
val_losses = []

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = pgnn_loss(y_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            y_pred = model(xb)
            val_loss += pgnn_loss(y_pred, yb).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ========== 7. 模型保存 ==========
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")

# ========== 8. 绘制收敛曲线 ==========
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PGNN Loss Convergence Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pgnn_loss_convergence.png")
plt.show()



