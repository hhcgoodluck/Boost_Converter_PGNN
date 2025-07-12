import torch
import pandas as pd
import numpy as np
import joblib
from 其他设计.res_train_boost_PGNN import compute_physical_features, ResidualPGNN  # 注意更换模型类名

# ===== 1. 加载测试数据 =====
test_df = pd.read_csv('../boost_converter_input.csv')
test_df = compute_physical_features(test_df)

# ===== 2. 提取输入特征（与训练保持一致）=====
features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D',
            'P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']
X_raw = test_df[features].values.astype(np.float32)

# ===== 3. 加载标准化器并标准化输入 =====
scaler_X = joblib.load('scaler_X.pkl')
X_test = scaler_X.transform(X_raw)
X_test = torch.tensor(X_test, dtype=torch.float32)

# ===== 4. 加载残差学习模型 =====
model = ResidualPGNN(input_dim=X_test.shape[1], hidden_dim=32, num_layers=3, dropout=0.1)
model.load_state_dict(torch.load('boost_pgnn_model.pth'))
model.eval()

# ===== 5. 推理预测：注意这里是预测残差 ΔP =====
with torch.no_grad():
    delta_pred = model(X_test).numpy()

# ===== 6. 加上物理建模预测值，得到最终功率预测值 =====
P_phy = test_df[['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']].values
P_final = delta_pred + P_phy  # 关键操作

# ===== 7. 保存结果 =====
test_df['P_capacitor_pred'] = P_final[:, 0]
test_df['P_inductor_pred'] = P_final[:, 1]
test_df['P_diode_pred'] = P_final[:, 2]
test_df['P_mosfet_pred'] = P_final[:, 3]

test_df.to_csv('prediction_results.csv', index=False)
print("Prediction completed and saved to prediction_results.csv")
