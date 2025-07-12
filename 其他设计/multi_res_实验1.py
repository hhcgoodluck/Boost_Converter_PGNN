import torch
import pandas as pd
import joblib
from 其他设计.multi_res_boost_PGNN import ResidualPGNN  # 使用新模型

# ===== 1. 加载测试数据并计算物理建模特征 =====
def compute_physical_features_predict(df):
    """
    预测阶段使用的特征计算函数，只计算 P_phy_* 相关的物理估计值。
    不引用任何 ground-truth 数据，保证测试阶段的纯净性。
    """
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

    # 仿照训练阶段计算四个物理估计项
    df['P_phy_capacitor'] = ESR * Iout ** 2
    df['P_phy_inductor'] = R_ind * Iin ** 2
    df['P_phy_diode'] = V_diode * Iout
    df['P_phy_mosfet'] = R_ds * Iin ** 2 + fs * Q_sw * Vin

    return df


test_df = pd.read_csv('../boost_converter_input.csv')
test_df = compute_physical_features_predict(test_df)

# ===== 2. 构造模型输入特征（和训练时保持一致）=====
input_features = ['Vin', 'Iin', 'Vout', 'Iout', 'fs', 'D',
                  'P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']
X_raw = torch.tensor(test_df[input_features].values, dtype=torch.float32)

# ===== 3. 标准化输入 =====
scaler_X = joblib.load('scaler_X.pkl')  # 与训练时保持一致
X_scaled = scaler_X.transform(X_raw)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

# ===== 4. 加载训练好的模型 =====
model = ResidualPGNN(input_dim=X_scaled.shape[1], hidden_dim=64, num_layers=4, dropout=0.1)
model.load_state_dict(torch.load('boost_multi_residual_pgnn_model.pth'))
model.eval()

# ===== 5. 模型预测残差 ΔP，并加上 P_phy 得到最终 P_pred =====
with torch.no_grad():
    delta_pred = model(X_scaled).numpy()

# 拿出 P_phy
P_phy = test_df[['P_phy_capacitor', 'P_phy_inductor', 'P_phy_diode', 'P_phy_mosfet']].values

# 计算最终预测值 P_pred = ΔP + P_phy
P_pred = delta_pred + P_phy

# ===== 6. 添加到 DataFrame 并保存 =====
test_df['P_capacitor_pred'] = P_pred[:, 0]
test_df['P_inductor_pred'] = P_pred[:, 1]
test_df['P_diode_pred'] = P_pred[:, 2]
test_df['P_mosfet_pred'] = P_pred[:, 3]

test_df.to_csv('prediction_results_residual_pgnn.csv', index=False)
print("Prediction completed and saved to prediction_results_residual_pgnn.csv")
