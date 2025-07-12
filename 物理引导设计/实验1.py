import torch
import pandas as pd
import joblib  # 用于加载 StandardScaler
from 物理引导设计.train_boost_PGNN import PGNN, compute_physical_features

# ===== 1. 加载并处理原始测试数据 =====
test_df = pd.read_csv('../boost_converter_input.csv')
test_df = compute_physical_features(test_df)

# ===== 2. 定义输入特征（必须和训练时保持一致）=====
features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D',
            'P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']
X_raw = torch.tensor(test_df[features].values, dtype=torch.float32)

# ===== 3. 加载 scaler 并对输入标准化 =====
scaler_X = joblib.load('scaler_X.pkl')  # 确保和训练时保存的是同一个文件
X_test = scaler_X.transform(X_raw)      # 标准化输入
X_test = torch.tensor(X_test, dtype=torch.float32)

# ===== 4. 加载模型 =====
model = PGNN(input_dim=X_test.shape[1], hidden_dim=32, num_layers=3, dropout=0.1)
model.load_state_dict(torch.load('boost_pgnn_model.pth'))
model.eval()

# ===== 5. 进行预测 =====
y_pred = model(X_test).detach().numpy()

# ===== 6. 保存预测结果 =====
test_df['P_capacitor_pred'] = y_pred[:, 0]
test_df['P_inductor_pred'] = y_pred[:, 1]
test_df['P_diode_pred'] = y_pred[:, 2]
test_df['P_mosfet_pred'] = y_pred[:, 3]

test_df.to_csv('prediction_results.csv', index=False)
print("Prediction completed and saved to prediction_results.csv")
