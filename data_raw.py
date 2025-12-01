import pandas as pd
import numpy as np

# ============================================================
# 1. Load dữ liệu gốc
# ============================================================

input_file = "dirty_default_credit_modified.csv"
df = pd.read_csv(input_file)

print("📥 Đã load dữ liệu:", df.shape)
target_col = "defaultpaymentnextmonth"

# Numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target_col]

# Categorical columns (education, sex, marriage, etc.)
categorical_cols = ['education', 'sex', 'marriage', 'age', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# ============================================================
# 2. Chèn Missing Values 20% cho các cột PAY_0 → PAY_6
# ============================================================

print("🔧 Bước 1: Chèn Missing Values 20% cho các cột PAY_0 → PAY_6...")

df_missing = df.copy()

# Chèn 20% Missing Values vào các cột PAY_0 → PAY_6
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in pay_cols:
    mask = np.random.rand(len(df_missing)) < 0.20  # 20% Missing Values
    df_missing.loc[mask, col] = np.nan

print("   ✓ Done: Missing Values 20% cho các cột PAY_0 → PAY_6")

# ============================================================
# 3. Chèn Noise 10% cho các cột PAY_0 → PAY_6
# ============================================================

print("🔧 Bước 2: Chèn Noise 10% cho các cột PAY_0 → PAY_6...")

df_noise = df_missing.copy()

# Chèn Noise 10% vào các cột PAY_0 → PAY_6
for col in pay_cols:
    noise_idx = df_noise.sample(frac=0.10, random_state=42).index  # 10% noise
    df_noise.loc[noise_idx, col] = df_noise.loc[noise_idx, col].apply(
        lambda x: np.random.choice([x + np.random.randint(-3, 4), np.random.randint(-10, 15)])
    )

print("   ✓ Done: Noise 10% cho các cột PAY_0 → PAY_6")

# ============================================================
# 4. Xuất file dữ liệu đã sửa
# ============================================================

output_file = "dirty_default_credit_modified.csv"
df_noise.to_csv(output_file, index=False)

print("\n🎉 HOÀN THÀNH!")
print("📁 File dữ liệu đã được lưu thành:")
print(f"➡ {output_file}")
print("🔥 Dữ liệu này đã được làm mất và nhiễu theo yêu cầu.")
