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

# ========== 1. Config remains unchanged ==========
class Args:
    data_path = 'boost_converter_dataset.csv'
    batch_size = 250
    epochs = 300
    drop_frac = 0.1
    n_nodes = 64
    n_layers = 4

    lam_cons = 20
    lam_non_neg = 35
    lam_main = 27

    val_frac = 0.3
    save_path = './boost_branch_fusion_model.pth'
args = Args()


# ========== 2. Dataset and Data Loading remains unchanged ==========
data_raw = pd.read_csv(args.data_path)
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
X = data_raw[input_features].values.astype(np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, 'scaler_X_data_driven.pkl')

y_target = data_raw[['P_capacitor', 'P_inductor', 'P_diode', 'P_mosfet']].values.astype(np.float32)
y_total = data_raw[['P_loss']].values.astype(np.float32)
y_parts = np.concatenate([y_target, y_total], axis=1)

X_train, X_val, y_train_parts, y_val_parts = train_test_split(X_scaled, y_parts, test_size=args.val_frac, random_state=42)

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


# ========== 3. Multi-Branch Network with Shared Inputs ==========
class BranchNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, p_parts):
        return self.net(p_parts)

class FullModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.branch_cap = BranchNet(input_dim)
        self.branch_ind = BranchNet(input_dim)
        self.branch_dio = BranchNet(input_dim)
        self.branch_mos = BranchNet(input_dim)
        self.fusion = FusionNet()

    def forward(self, x):
        p_cap = self.branch_cap(x)
        p_ind = self.branch_ind(x)
        p_dio = self.branch_dio(x)
        p_mos = self.branch_mos(x)
        p_parts = torch.cat([p_cap, p_ind, p_dio, p_mos], dim=1)
        p_total = self.fusion(p_parts)
        return p_parts, p_total

model = FullModel(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 4. Custom Loss ==========
def multi_branch_loss(p_parts_pred, p_total_pred, y_parts, alpha=27.0, eps=1e-6):
    y_true_parts = y_parts[:, :4]
    y_true_total = y_parts[:, 4].unsqueeze(1)

    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=p_parts_pred.device).unsqueeze(0)
    main_loss = F.mse_loss(p_parts_pred, y_true_parts, reduction='none')
    main_loss = torch.mean(weights * main_loss)

    rel_error = torch.abs((p_parts_pred - y_true_parts) / (y_true_parts + eps))
    relative_loss = torch.mean(weights * rel_error)

    cons_loss = F.mse_loss(torch.sum(p_parts_pred, dim=1, keepdim=True), y_true_total)
    fusion_loss = F.mse_loss(p_total_pred, y_true_total)

    non_neg_penalty = torch.relu(-p_parts_pred).sum()
    return (args.lam_main * (main_loss + alpha * relative_loss)
            + args.lam_cons * cons_loss
            + 0.5 * fusion_loss
            + args.lam_non_neg * non_neg_penalty)

# ========== 5. Training Loop ==========
train_losses, val_losses = [], []
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        p_parts_pred, p_total_pred = model(xb)
        loss = multi_branch_loss(p_parts_pred, p_total_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            p_parts_pred, p_total_pred = model(xb)
            val_loss += multi_branch_loss(p_parts_pred, p_total_pred, yb).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ========== 6. Save Model ==========
torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")

# ========== 7. Plot Loss ==========
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Multi-Branch PGNN Loss Convergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("multi_branch_pgnn_loss_curve.png")
plt.show()





import torch
import pandas as pd
import joblib
import numpy as np

# ===== 1. 加载模型结构和权重 =====
class BranchNet(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class FusionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, p_parts):
        return self.net(p_parts)

class FullModel(torch.nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.branch_cap = BranchNet(input_dim)
        self.branch_ind = BranchNet(input_dim)
        self.branch_dio = BranchNet(input_dim)
        self.branch_mos = BranchNet(input_dim)
        self.fusion = FusionNet()

    def forward(self, x):
        p_cap = self.branch_cap(x)
        p_ind = self.branch_ind(x)
        p_dio = self.branch_dio(x)
        p_mos = self.branch_mos(x)
        p_parts = torch.cat([p_cap, p_ind, p_dio, p_mos], dim=1)
        p_total = self.fusion(p_parts)
        return p_parts, p_total

# 加载模型
model_path = 'boost_branch_fusion_model.pth'
model = FullModel(input_dim=6)
model.load_state_dict(torch.load(model_path))
model.eval()

# ===== 2. 加载并标准化输入特征 =====
scaler_X = joblib.load('scaler_X_data_driven.pkl')
data_raw = pd.read_csv('boost_converter_dataset.csv')
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D']
X = data_raw[input_features].values.astype(np.float32)
X_scaled = scaler_X.transform(X)

# 转为 tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ===== 3. 执行预测 =====
with torch.no_grad():
    P_parts_pred, P_total_pred = model(X_tensor)

# 转为 numpy
P_parts_pred = P_parts_pred.numpy()
P_total_pred = P_total_pred.numpy()

# ===== 4. 保存预测结果 =====
df_pred = pd.DataFrame(P_parts_pred, columns=[
    'P_capacitor_pred', 'P_inductor_pred', 'P_diode_pred', 'P_mosfet_pred'
])
df_pred['P_total_pred'] = P_total_pred

# 可选：拼接真实值进行对比
df_true = data_raw[['P_capacitor', 'P_inductor', 'P_diode', 'P_mosfet', 'P_loss']]
df_result = pd.concat([df_true, df_pred], axis=1)

# 保存到 CSV
df_result.to_csv('prediction_results_branch_fusion.csv', index=False)
print("Prediction saved to prediction_results_branch_fusion.csv")
