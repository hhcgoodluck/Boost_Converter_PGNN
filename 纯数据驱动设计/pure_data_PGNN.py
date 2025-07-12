# Purely Data-Driven PGNN Training Script with Multi-Head Output and Physical Constraints

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

# ========== 1. Config ==========
class Args:
    data_path = '../boost_converter_dataset.csv'
    batch_size = 250
    epochs = 300
    drop_frac = 0.1
    n_nodes = 64
    n_layers = 4

    lam_cons = 20
    lam_non_neg = 35
    lam_main = 27

    val_frac = 0.3
    save_path = '../boost_data_driven_pgnn_model.pth'
args = Args()

# ========== 2. Load Data ==========
data_raw = pd.read_csv(args.data_path)
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
X = data_raw[input_features].values.astype(np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, '../../scaler_X_data_driven.pkl')

y_target = data_raw[['P_capacitor', 'P_inductor', 'P_diode', 'P_mosfet']].values.astype(np.float32)
y_total = data_raw[['P_loss']].values.astype(np.float32)
y_parts = np.concatenate([y_target, y_total], axis=1)

X_train, X_val, y_train_parts, y_val_parts = train_test_split(X_scaled, y_parts, test_size=args.val_frac, random_state=42)

# ========== 3. Dataset ==========
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

# ========== 4. Multi-Head Regression Model ==========
class MultiHeadNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.shared_layers = nn.Sequential(*layers)
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

model = MultiHeadNet(input_dim=X.shape[1], hidden_dim=args.n_nodes, num_layers=args.n_layers, dropout=args.drop_frac)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 5. Custom Loss ==========
def data_driven_loss(y_pred, y_parts, alpha=27.0, eps=1e-6):
    """
    强化每个器件功率预测的精度：
    - 使用MSE作为主监督
    - 增加相对误差惩罚，进一步降低每个器件的相对误差
    - 保留总功率守恒 + 非负约束
    """
    y_true = y_parts[:, :4]  # 真值 P_cap, P_ind, P_dio, P_mos
    y_total_true = y_parts[:, 4].unsqueeze(1)  # P_loss 总功率

    # 每个器件的损耗权重，提升capacitor分支（index=0）的影响力
    component_weights = torch.tensor([50, 1.0, 1.0, 1.0], device=y_pred.device).unsqueeze(0)  # shape: [1, 4]

    # 主监督损失（加权MSE）
    main_loss = F.mse_loss(y_pred, y_true, reduction='none')  # shape: [batch_size, 4]
    main_loss = torch.mean(component_weights * main_loss)  # 按分支加权后求平均

    # 相对误差损失（加权）
    relative_error = torch.abs((y_pred - y_true) / (y_true + eps))  # shape: [batch_size, 4]
    relative_loss = torch.mean(component_weights * relative_error)

    # 守恒损失
    cons_loss = F.mse_loss(torch.sum(y_pred, dim=1, keepdim=True), y_total_true)

    # 非负性惩罚
    non_neg_penalty = torch.relu(-y_pred).sum()

    return (args.lam_main * (main_loss + alpha * relative_loss) +
            args.lam_cons * cons_loss +
            args.lam_non_neg * non_neg_penalty)


# ========== 6. Training ==========
train_losses, val_losses = [], []
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = data_driven_loss(y_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            y_pred = model(xb)
            val_loss += data_driven_loss(y_pred, yb).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ========== 7. Save Model ==========
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")

# ========== 8. Loss Curve ==========
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Data-Driven PGNN Loss Convergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data_driven_pgnn_loss_curve.png")
plt.show()



import torch
import pandas as pd
import numpy as np
import joblib

# ===== 1. 加载测试数据 =====
test_df = pd.read_csv('../boost_converter_dataset.csv')

# ===== 2. 构造模型输入特征（纯数据驱动，不含物理项）=====
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
X_raw = torch.tensor(test_df[input_features].values, dtype=torch.float32)

# ===== 3. 标准化输入特征 =====
scaler_X = joblib.load('../../scaler_X_data_driven.pkl')  # 注意：scaler 也是在纯数据驱动模式下训练的版本
X_scaled = scaler_X.transform(X_raw)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

# ===== 4. 加载训练好的模型权重 =====
model = MultiHeadNet(input_dim=X_scaled.shape[1], hidden_dim=64, num_layers=4, dropout=0.1)
model.load_state_dict(torch.load('../boost_data_driven_pgnn_model.pth'))  # 新模型路径
model.eval()

# ===== 5. 预测 ΔP，即每个器件的功率（此时 ΔP = P_pred，因为没有 P_phy）=====
with torch.no_grad():
    P_pred = model(X_scaled).numpy()

# ===== 6. 写入结果 =====
test_df['P_capacitor_pred'] = P_pred[:, 0]
test_df['P_inductor_pred'] = P_pred[:, 1]
test_df['P_diode_pred'] = P_pred[:, 2]
test_df['P_mosfet_pred'] = P_pred[:, 3]
test_df['P_total_pred'] = np.sum(P_pred, axis=1)

# 保存文件
test_df.to_csv('prediction_results_data_driven_pgnn.csv', index=False)
print("Prediction completed and saved to prediction_results_data_driven_pgnn.csv")
