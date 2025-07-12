import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('prediction_results_data_driven_pgnn.csv')

components = ['capacitor', 'inductor', 'diode', 'mosfet']
relative_errors = {}

for comp in components:
    true_col = f'P_{comp}'
    pred_col = f'P_{comp}_pred'
    relative_errors[comp] = np.abs((df[pred_col] - df[true_col]) / (df[true_col] + 1e-6))  # 加1e-6避免除0

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot([relative_errors[c] for c in components], tick_labels=components)
plt.ylabel("Relative Error")
plt.title("Relative Error of Predicted Power per Component")
plt.grid(True)
plt.tight_layout()
plt.savefig("relative_error_components.png")
plt.show()


# 基于预测值计算占比
P_total_pred = df['P_total_pred']
ratios = {comp: df[f'P_{comp}_pred'] / (P_total_pred + 1e-6) for comp in components}

# 堆叠面积图
plt.figure(figsize=(10, 6))
plt.stackplot(range(len(df)), [ratios[c] for c in components], labels=components)
plt.ylabel("Proportion of Total Power Loss")
plt.xlabel("Sample Index")
plt.title("Power Loss Proportion per Component")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("power_loss_proportion_components.png")
plt.show()
