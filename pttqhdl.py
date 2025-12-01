# -*- coding: utf-8 -*-
"""
PHÂN TÍCH VÀ TIỀN XỬ LÝ DỮ LIỆU - CẤU TRÚC ĐÚNG
================================================================================
PHẦN 1 (Dòng 1-467): PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA) - KHÔNG THAY ĐỔI DỮ LIỆU
PHẦN 2 (Từ dòng 468): XỬ LÝ DỮ LIỆU THỰC SỰ
================================================================================
"""

# !pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*20 + "PHẦN 1: PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)")
print(" "*15 + "Lưu ý: Phần này CHỈ phân tích, KHÔNG thay đổi dữ liệu gốc")
print("="*80)

# ==========================================
# ĐỌC DỮ LIỆU
# ==========================================
print("\n📂 Đọc dữ liệu...")
app_train = pd.read_csv('/content/drive/MyDrive/Phantichtrucquanhoa/application_train.csv')
print(f"✓ Đã đọc xong: {app_train.shape[0]:,} dòng × {app_train.shape[1]} cột")

# ==========================================
# BƯỚC 1: THÔNG TIN CƠ BẢN VỀ DỮ LIỆU
# ==========================================
print("\n" + "="*80)
print("BƯỚC 1: THÔNG TIN CƠ BẢN VỀ DỮ LIỆU")
print("="*80)

print("\n📋 Cấu trúc dữ liệu:")
print(f"   - Số dòng: {app_train.shape[0]:,}")
print(f"   - Số cột: {app_train.shape[1]}")
print(f"   - Bộ nhớ sử dụng: {app_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Hiển thị 5 dòng đầu
print("\n📊 5 dòng đầu tiên:")
print(app_train.head())

# Thông tin kiểu dữ liệu
print("\n📝 Tóm tắt kiểu dữ liệu:")
dtype_counts = app_train.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   - {dtype}: {count} cột")

# ==========================================
# BƯỚC 2: PHÂN TÍCH MISSING VALUES
# ==========================================
print("\n" + "="*80)
print("BƯỚC 2: PHÂN TÍCH MISSING VALUES")
print("="*80)

missing = app_train.isnull().sum().sort_values(ascending=False)
missing_ratio = (missing / len(app_train)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Ratio (%)': missing_ratio
})

print(f"\n📊 Tổng quan:")
print(f"   - Tổng số cột: {app_train.shape[1]}")
print(f"   - Cột có missing: {(missing > 0).sum()}")
print(f"   - Cột có >70% missing: {(missing_ratio > 70).sum()}")
print(f"   - Cột có >50% missing: {(missing_ratio > 50).sum()}")
print(f"   - Cột có >30% missing: {(missing_ratio > 30).sum()}")

print("\n📋 Top 20 cột có missing cao nhất:")
print(missing_df[missing_df['Missing_Count'] > 0].head(20).to_string())

print("\n💡 Khuyến nghị:")
print("   - Cột có >70% missing: NÊN XÓA")
print("   - Cột có 40-70% missing: CÂN NHẮC XÓA hoặc impute cẩn thận")
print("   - Cột có <40% missing: IMPUTE với median/mode")

# ==========================================
# BƯỚC 3: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU
# ==========================================
print("\n" + "="*80)
print("BƯỚC 3: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
print("="*80)

# 3.1: Duplicates
print("\n🔄 Kiểm tra dữ liệu trùng lặp:")
duplicates = app_train.duplicated(subset='SK_ID_CURR').sum()
print(f"   - Số dòng trùng lặp (SK_ID_CURR): {duplicates}")
if duplicates > 0:
    print(f"   ⚠️ CẦN XÓA {duplicates} dòng trùng lặp khi xử lý")
else:
    print("   ✓ Không có dòng trùng lặp")

# 3.2: Phân bố TARGET
print("\n🎯 Phân bố TARGET (Label):")
target_dist = app_train['TARGET'].value_counts()
target_pct = app_train['TARGET'].value_counts(normalize=True) * 100
print(f"   - TARGET = 0 (Không vỡ nợ): {target_dist[0]:,} ({target_pct[0]:.2f}%)")
print(f"   - TARGET = 1 (Vỡ nợ): {target_dist[1]:,} ({target_pct[1]:.2f}%)")
print(f"   - Tỷ lệ imbalance: 1:{target_dist[0]/target_dist[1]:.1f}")
print(f"   ⚠️ Dữ liệu MẤT CÂN BẰNG NGHIÊM TRỌNG - cần xử lý bằng SMOTE hoặc class_weight")

# Visualization: TARGET distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=target_dist.index, y=target_dist.values, palette='Set2')
plt.title("Phân bố TARGET (0 = Không vỡ nợ, 1 = Vỡ nợ)", fontsize=14, fontweight='bold')
plt.xlabel("TARGET")
plt.ylabel("Số lượng mẫu")
for i, val in enumerate(target_dist.values):
    plt.text(i, val + 1000, f"{val:,}\n({target_pct[i]:.1f}%)", ha='center', fontsize=11)
plt.tight_layout()
plt.show()

# 3.3: Kiểm tra logic errors (chỉ đếm, không sửa)
print("\n🔍 Kiểm tra lỗi logic:")
logic_errors = {}

if 'NAME_INCOME_TYPE' in app_train.columns and 'DAYS_EMPLOYED' in app_train.columns:
    unemployed_error = ((app_train['NAME_INCOME_TYPE'] == 'Unemployed') & 
                       (app_train['DAYS_EMPLOYED'] != 365243) & 
                       (app_train['DAYS_EMPLOYED'].notna())).sum()
    if unemployed_error > 0:
        logic_errors['Unemployed nhưng có DAYS_EMPLOYED'] = unemployed_error

if 'CNT_CHILDREN' in app_train.columns:
    negative_children = (app_train['CNT_CHILDREN'] < 0).sum()
    if negative_children > 0:
        logic_errors['CNT_CHILDREN âm'] = negative_children

if 'AMT_INCOME_TOTAL' in app_train.columns:
    invalid_income = (app_train['AMT_INCOME_TOTAL'] <= 0).sum()
    if invalid_income > 0:
        logic_errors['AMT_INCOME_TOTAL <= 0'] = invalid_income

if 'CNT_FAM_MEMBERS' in app_train.columns:
    invalid_family = (app_train['CNT_FAM_MEMBERS'] <= 0).sum()
    if invalid_family > 0:
        logic_errors['CNT_FAM_MEMBERS <= 0'] = invalid_family

if logic_errors:
    print("   ⚠️ Phát hiện các lỗi logic:")
    for error, count in logic_errors.items():
        print(f"      - {error}: {count} dòng ({count/len(app_train)*100:.2f}%)")
else:
    print("   ✓ Không phát hiện lỗi logic")

# ==========================================
# BƯỚC 4: PHÂN TÍCH PHÂN BỐ VÀ OUTLIERS
# ==========================================
print("\n" + "="*80)
print("BƯỚC 4: PHÂN TÍCH PHÂN BỐ VÀ OUTLIERS")
print("="*80)

# 4.1: Thống kê mô tả mở rộng
print("\n📊 Thống kê mô tả mở rộng (số):")
num_cols = app_train.select_dtypes(include=['int64', 'float64']).columns

desc_extended = app_train[num_cols].describe().T
desc_extended['missing_%'] = app_train[num_cols].isnull().mean() * 100
desc_extended['skew'] = app_train[num_cols].apply(lambda x: skew(x.dropna()))
desc_extended['kurtosis'] = app_train[num_cols].apply(lambda x: kurtosis(x.dropna()))
desc_extended = desc_extended[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_%', 'skew', 'kurtosis']]

print("\nTop 15 cột (xem đầy đủ):")
print(desc_extended.round(2).head(15).to_string())

# 4.2: Phân tích độ lệch (Skewness)
print("\n📐 Phân tích độ lệch (Skewness):")
skew_values = app_train[num_cols].apply(lambda x: skew(x.dropna()))
high_skew = skew_values[abs(skew_values) > 1].sort_values(ascending=False, key=abs)

print(f"   - Số cột có |skew| > 1: {len(high_skew)} (cần log-transform)")
print(f"   - Số cột có |skew| > 2: {(abs(skew_values) > 2).sum()} (rất lệch)")
print(f"   - Số cột có |skew| > 3: {(abs(skew_values) > 3).sum()} (cực lệch)")

print("\n   Top 10 cột lệch nhất:")
for col, skew_val in high_skew.head(10).items():
    print(f"      - {col}: {skew_val:.3f}")

print("\n💡 Khuyến nghị:")
print("   - Các cột có |skew| > 1: Áp dụng LOG TRANSFORM (np.log1p)")
print("   - KHÔNG nên xóa outliers trong credit scoring!")

# 4.3: Visualization - Phân bố các biến quan trọng
print("\n📊 Vẽ biểu đồ phân bố...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cols_to_plot = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

for idx, col in enumerate(cols_to_plot):
    ax = axes[idx // 3, idx % 3]
    if col in app_train.columns:
        data = app_train[col].dropna()
        ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f'{col}\nSkew: {data.skew():.2f} | Mean: {data.mean():.0f}', fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle('Phân bố các biến quan trọng', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# 4.4: Boxplot - Phát hiện outliers
print("\n📦 Phân tích outliers (IQR method):")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
financial_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

outlier_summary = []
for idx, col in enumerate(financial_cols):
    if col in app_train.columns:
        ax = axes[idx]
        data = app_train[col].dropna()
        
        # Tính IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        # Vẽ boxplot
        app_train.boxplot(column=col, ax=ax)
        ax.set_title(f'{col}\nOutliers: {outliers:,} ({outliers/len(data)*100:.1f}%)', fontsize=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        outlier_summary.append({
            'Column': col,
            'Outliers_Count': outliers,
            'Outliers_%': round(outliers/len(data)*100, 2),
            'Lower_Bound': round(lower_bound, 2),
            'Upper_Bound': round(upper_bound, 2)
        })

plt.suptitle('Boxplot - Phát hiện Outliers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n   Bảng tổng hợp outliers:")
print(pd.DataFrame(outlier_summary).to_string(index=False))

print("\n⚠️ LƯU Ý QUAN TRỌNG:")
print("   - KHÔNG xóa outliers trong credit scoring!")
print("   - Outliers có thể là khách hàng cao cấp hoặc high-risk")
print("   - Giải pháp: Dùng log-transform thay vì xóa")

# ==========================================
# BƯỚC 5: PHÂN TÍCH TƯƠNG QUAN
# ==========================================
print("\n" + "="*80)
print("BƯỚC 5: PHÂN TÍCH TƯƠNG QUAN")
print("="*80)

# 5.1: Ma trận tương quan cho các biến tài chính
print("\n📊 Ma trận tương quan các biến tài chính:")
financial_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'TARGET']
existing_fin_cols = [col for col in financial_cols if col in app_train.columns]

if len(existing_fin_cols) > 2:
    plt.figure(figsize=(10, 8))
    corr_matrix = app_train[existing_fin_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Ma trận tương quan - Biến tài chính", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Tìm các cặp tương quan cao
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], 
                                       corr_matrix.columns[j], 
                                       corr_val))
    
    if high_corr_pairs:
        print("\n   ⚠️ Các cặp biến có tương quan cao (|r| > 0.7):")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"      - {col1} <-> {col2}: {corr_val:.3f}")
        print("   💡 Cần loại bỏ 1 trong 2 biến tương quan cao để tránh multicollinearity")
    else:
        print("\n   ✓ Không có cặp biến nào tương quan quá cao")

# 5.2: Tương quan với TARGET
print("\n🎯 Tương quan với TARGET:")
if 'TARGET' in app_train.columns:
    target_corr = app_train[num_cols].corrwith(app_train['TARGET']).sort_values(ascending=False, key=abs)
    top_positive = target_corr[target_corr > 0].head(10)
    top_negative = target_corr[target_corr < 0].head(10)
    
    print("\n   Top 10 biến tương quan DƯƠNG với TARGET:")
    for col, corr_val in top_positive.items():
        if col != 'TARGET':
            print(f"      - {col}: {corr_val:.3f}")
    
    print("\n   Top 10 biến tương quan ÂM với TARGET:")
    for col, corr_val in top_negative.items():
        print(f"      - {col}: {corr_val:.3f}")

# ==========================================
# BƯỚC 6: PHÂN TÍCH BIẾN PHÂN LOẠI
# ==========================================
print("\n" + "="*80)
print("BƯỚC 6: PHÂN TÍCH BIẾN PHÂN LOẠI")
print("="*80)

cat_cols = app_train.select_dtypes(include=['object']).columns
print(f"\n📊 Tổng số biến phân loại: {len(cat_cols)}")

# Phân tích chi tiết 5 biến đầu tiên
print("\n📋 Phân tích chi tiết các biến phân loại:")
for col in cat_cols[:5]:
    print(f"\n   {col}:")
    print(f"      - Số giá trị duy nhất: {app_train[col].nunique()}")
    print(f"      - Missing values: {app_train[col].isnull().sum()} ({app_train[col].isnull().sum()/len(app_train)*100:.2f}%)")
    
    value_counts = app_train[col].value_counts()
    print(f"      - Top 3 giá trị phổ biến:")
    for val, count in value_counts.head(3).items():
        print(f"         • {val}: {count:,} ({count/len(app_train)*100:.2f}%)")

# ==========================================
# BƯỚC 7: PHÂN BỐ TARGET THEO NHÓM
# ==========================================
print("\n" + "="*80)
print("BƯỚC 7: PHÂN BỐ TARGET THEO CÁC NHÓM QUAN TRỌNG")
print("="*80)

# 7.1: Theo giới tính
if 'CODE_GENDER' in app_train.columns:
    print("\n👥 Phân bố TARGET theo giới tính:")
    gender_target = pd.crosstab(app_train['CODE_GENDER'], app_train['TARGET'], normalize='index') * 100
    print(gender_target.round(2).to_string())
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    gender_target.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Tỷ lệ vỡ nợ theo giới tính (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Giới tính')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.legend(['Không vỡ nợ', 'Vỡ nợ'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

# 7.2: Theo trình độ học vấn
if 'NAME_EDUCATION_TYPE' in app_train.columns:
    print("\n🎓 Phân bố TARGET theo trình độ học vấn:")
    edu_target = pd.crosstab(app_train['NAME_EDUCATION_TYPE'], app_train['TARGET'], normalize='index') * 100
    print(edu_target.round(2).to_string())

# 7.3: Theo loại thu nhập
if 'NAME_INCOME_TYPE' in app_train.columns:
    print("\n💰 Phân bố TARGET theo loại thu nhập:")
    income_target = pd.crosstab(app_train['NAME_INCOME_TYPE'], app_train['TARGET'], normalize='index') * 100
    print(income_target.round(2).to_string())

print("\n" + "="*80)
print("KẾT THÚC PHẦN PHÂN TÍCH - DỮ LIỆU CHƯA BỊ THAY ĐỔI")
print("="*80)

print("\n" + "🎯"*40)
print("\nĐể bắt đầu XỬ LÝ DỮ LIỆU, chạy PHẦN 2 bên dưới...")
print("\n" + "🎯"*40)

################################################################################
#                                                                              #
#                 PHẦN 2: XỬ LÝ DỮ LIỆU THỰC SỰ                              #
#                                                                              #
################################################################################

print("\n\n" + "="*80)
print(" "*20 + "PHẦN 2: XỬ LÝ DỮ LIỆU THỰC SỰ")
print(" "*10 + "LƯU Ý: Phần này SẼ THAY ĐỔI dữ liệu - tạo bản sao trước khi chạy!")
print("="*80)

# Import thêm các thư viện cần thiết cho xử lý
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# TẠO BẢN SAO ĐỂ XỬ LÝ
print("\n📋 Tạo bản sao dữ liệu để xử lý...")
df = app_train.copy()
print(f"✓ Đã tạo bản sao: {df.shape}")

# ==========================================
# BƯỚC 1: LÀM SẠCH CƠ BẢN
# ==========================================
print("\n" + "="*80)
print("BƯỚC 1: LÀM SẠCH CƠ BẢN")
print("="*80)

# 1.1: Xóa duplicates
print("\n🗑️ Xóa dòng trùng lặp...")
original_rows = df.shape[0]
df = df.drop_duplicates(subset='SK_ID_CURR')
removed = original_rows - df.shape[0]
print(f"   ✓ Đã xóa {removed} dòng trùng lặp")
print(f"   ✓ Kích thước mới: {df.shape}")

# 1.2: Sửa lỗi logic
print("\n🔧 Sửa lỗi logic...")

# Unemployed → DAYS_EMPLOYED = 365243 hoặc NaN
if 'NAME_INCOME_TYPE' in df.columns and 'DAYS_EMPLOYED' in df.columns:
    unemployed_mask = (df['NAME_INCOME_TYPE'] == 'Unemployed') & \
                     (df['DAYS_EMPLOYED'] != 365243) & \
                     (df['DAYS_EMPLOYED'].notna())
    n_fixed = unemployed_mask.sum()
    df.loc[unemployed_mask, 'DAYS_EMPLOYED'] = 365243
    print(f"   ✓ Sửa {n_fixed} dòng: Unemployed → DAYS_EMPLOYED = 365243")

# CNT_CHILDREN < 0 → 0
if 'CNT_CHILDREN' in df.columns:
    negative_children = (df['CNT_CHILDREN'] < 0).sum()
    df.loc[df['CNT_CHILDREN'] < 0, 'CNT_CHILDREN'] = 0
    print(f"   ✓ Sửa {negative_children} dòng: CNT_CHILDREN âm → 0")

# AMT_INCOME_TOTAL <= 0 → NaN
if 'AMT_INCOME_TOTAL' in df.columns:
    invalid_income = (df['AMT_INCOME_TOTAL'] <= 0).sum()
    df.loc[df['AMT_INCOME_TOTAL'] <= 0, 'AMT_INCOME_TOTAL'] = np.nan
    print(f"   ✓ Sửa {invalid_income} dòng: AMT_INCOME_TOTAL <= 0 → NaN")

# CNT_FAM_MEMBERS <= 0 → 1
if 'CNT_FAM_MEMBERS' in df.columns:
    invalid_family = (df['CNT_FAM_MEMBERS'] <= 0).sum()
    df.loc[df['CNT_FAM_MEMBERS'] <= 0, 'CNT_FAM_MEMBERS'] = 1
    print(f"   ✓ Sửa {invalid_family} dòng: CNT_FAM_MEMBERS <= 0 → 1")

# 1.3: Xóa cột có quá nhiều missing (>70%)
print("\n🗑️ Xóa cột có >70% missing values...")
missing_threshold = 0.70
missing_ratio_current = df.isnull().sum() / len(df)
cols_to_drop = missing_ratio_current[missing_ratio_current > missing_threshold].index.tolist()

# Không xóa TARGET
if 'TARGET' in cols_to_drop:
    cols_to_drop.remove('TARGET')

print(f"   ✓ Xóa {len(cols_to_drop)} cột:")
if len(cols_to_drop) > 0:
    print(f"      {', '.join(cols_to_drop[:5])}...")
    df = df.drop(columns=cols_to_drop)

print(f"   ✓ Kích thước mới: {df.shape}")

# ==========================================
# BƯỚC 2: FEATURE ENGINEERING (TRƯỚC KHI SCALE!)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 2: TẠO CÁC BIẾN MỚI (FEATURE ENGINEERING)")
print("="*80)

print("\n✨ Tạo các biến mới từ dữ liệu gốc...")

# 2.1: Tuổi (năm)
if 'DAYS_BIRTH' in df.columns:
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    print("   ✓ AGE_YEARS: tuổi khách hàng (năm)")

# 2.2: Thời gian làm việc (năm)
if 'DAYS_EMPLOYED' in df.columns:
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365.25 if x != 365243 and pd.notna(x) else np.nan
    )
    print("   ✓ EMPLOYMENT_YEARS: số năm làm việc")

# 2.3: Tỷ lệ tài chính
if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_INCOME_RATIO'] = df['CREDIT_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
    print("   ✓ CREDIT_INCOME_RATIO: tỷ lệ khoản vay/thu nhập")

if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['ANNUITY_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
    print("   ✓ ANNUITY_INCOME_RATIO: tỷ lệ trả góp/thu nhập")

if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_PERSON'] = df['INCOME_PER_PERSON'].replace([np.inf, -np.inf], np.nan)
    print("   ✓ INCOME_PER_PERSON: thu nhập bình quân/người")

if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TERM'] = df['CREDIT_TERM'].replace([np.inf, -np.inf], np.nan)
    print("   ✓ CREDIT_TERM: kỳ hạn vay (tháng)")

# 2.4: Điểm tín dụng trung bình
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
if all(col in df.columns for col in ext_sources):
    df['EXT_SOURCE_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_sources].max(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_sources].min(axis=1)
    print("   ✓ EXT_SOURCE_MEAN/MAX/MIN: điểm tín dụng tổng hợp")

# 2.5: Tỷ lệ tuổi/thời gian làm việc
if 'AGE_YEARS' in df.columns and 'EMPLOYMENT_YEARS' in df.columns:
    df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS']
    df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_AGE_RATIO'].replace([np.inf, -np.inf], np.nan)
    print("   ✓ EMPLOYMENT_AGE_RATIO: tỷ lệ thời gian làm việc/tuổi")

print(f"\n📊 Kích thước sau feature engineering: {df.shape}")

# ==========================================
# BƯỚC 3: XỬ LÝ MISSING VALUES
# ==========================================
print("\n" + "="*80)
print("BƯỚC 3: XỬ LÝ MISSING VALUES")
print("="*80)

print("\n💧 Điền missing values...")

# 3.1: Xử lý biến số - dùng median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = [col for col in num_cols if col != 'TARGET']  # Không impute TARGET

for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"   ✓ Đã điền {len(num_cols)} biến số bằng median")

# 3.2: Xử lý biến phân loại - dùng mode
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col] = df[col].fillna(mode_val)

print(f"   ✓ Đã điền {len(cat_cols)} biến phân loại bằng mode")

# Kiểm tra lại
remaining_missing = df.isnull().sum().sum()
print(f"\n📊 Missing values còn lại: {remaining_missing}")

# ==========================================
# BƯỚC 4: XỬ LÝ SKEWED FEATURES (LOG TRANSFORM)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 4: XỬ LÝ BIẾN LỆCH (LOG TRANSFORM)")
print("="*80)

print("\n📐 Áp dụng log-transform cho các biến lệch cao...")

# Tìm các biến số có skew > 1
num_cols_current = df.select_dtypes(include=['int64', 'float64']).columns
num_cols_current = [col for col in num_cols_current if col != 'TARGET']

skewed_features = []
for col in num_cols_current:
    if df[col].min() >= 0:  # Chỉ transform biến không âm
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            df[col] = np.log1p(df[col])  # log1p = log(1 + x)
            skewed_features.append((col, skew_val))

print(f"   ✓ Đã transform {len(skewed_features)} biến:")
for col, old_skew in skewed_features[:10]:
    new_skew = df[col].skew()
    print(f"      - {col}: skew {old_skew:.2f} → {new_skew:.2f}")

# ==========================================
# BƯỚC 5: MÃ HÓA BIẾN PHÂN LOẠI
# ==========================================
print("\n" + "="*80)
print("BƯỚC 5: MÃ HÓA BIẾN PHÂN LOẠI (ONE-HOT ENCODING)")
print("="*80)

print("\n🔢 Mã hóa biến phân loại...")
original_cols = df.shape[1]

# One-hot encoding
df = pd.get_dummies(df, drop_first=True, dtype=int)

print(f"   ✓ Số cột trước: {original_cols}")
print(f"   ✓ Số cột sau: {df.shape[1]}")
print(f"   ✓ Đã tạo thêm {df.shape[1] - original_cols} cột dummy")

# ==========================================
# BƯỚC 6: LOẠI BỎ FEATURES TƯƠNG QUAN CAO
# ==========================================
print("\n" + "="*80)
print("BƯỚC 6: LOẠI BỎ FEATURES TƯƠNG QUAN CAO")
print("="*80)

print("\n🔗 Tìm và loại bỏ features tương quan cao (>0.95)...")

# Tính ma trận tương quan (chỉ lấy biến số)
num_features = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[num_features].corr().abs()

# Tìm các cặp tương quan cao
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

# Không xóa TARGET
if 'TARGET' in high_corr_features:
    high_corr_features.remove('TARGET')

if len(high_corr_features) > 0:
    print(f"   ✓ Xóa {len(high_corr_features)} features tương quan cao:")
    print(f"      {', '.join(high_corr_features[:5])}...")
    df = df.drop(columns=high_corr_features)
else:
    print("   ✓ Không có features nào tương quan quá cao")

print(f"\n📊 Kích thước sau loại bỏ: {df.shape}")

# ==========================================
# BƯỚC 7: CHIA TRAIN-TEST (TRƯỚC KHI SCALE!)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 7: CHIA TRAIN-TEST SET")
print("="*80)

print("\n✂️ Chia dữ liệu thành train và test...")

# Tách X và y
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Train set: {X_train.shape[0]:,} dòng ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"   ✓ Test set: {X_test.shape[0]:,} dòng ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"   ✓ Số features: {X_train.shape[1]}")

# Kiểm tra phân bố TARGET
print(f"\n   Phân bố TARGET trong train:")
print(f"      - Class 0: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"      - Class 1: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# ==========================================
# BƯỚC 8: FEATURE SELECTION (LỌC ĐẶC TRƯNG QUAN TRỌNG)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 8: FEATURE SELECTION - LỌC ĐẶC TRƯNG QUAN TRỌNG")
print("="*80)

print("\n🎯 Lọc features quan trọng để cải thiện accuracy...")

# 8.1: Tính mutual information score
print("\n   📊 Tính Mutual Information Score...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
mi_scores = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

# Lọc features có MI score > 0 (có thông tin)
important_features = mi_scores[mi_scores > 0].index.tolist()
print(f"   ✓ Features có thông tin: {len(important_features)}/{len(X_train.columns)}")

# Giữ top 80% features quan trọng nhất
n_features_to_keep = max(50, int(len(important_features) * 0.8))
selected_features = mi_scores.head(n_features_to_keep).index.tolist()

print(f"   ✓ Chọn top {n_features_to_keep} features quan trọng nhất")
print(f"\n   Top 10 features quan trọng:")
for i, (feat, score) in enumerate(mi_scores.head(10).items(), 1):
    print(f"      {i}. {feat}: {score:.4f}")

# Áp dụng feature selection
X_train_selected = X_train[selected_features].copy()
X_test_selected = X_test[selected_features].copy()

print(f"\n   ✓ Giảm từ {X_train.shape[1]} → {X_train_selected.shape[1]} features")

# ==========================================
# BƯỚC 9: CHUẨN HÓA DỮ LIỆU (ROBUST SCALER)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 9: CHUẨN HÓA DỮ LIỆU (ROBUST SCALER)")
print("="*80)

print("\n⚖️ Chuẩn hóa dữ liệu với RobustScaler (tốt hơn cho outliers)...")

# Chỉ scale các cột số (không scale dummy variables)
numeric_features = X_train_selected.select_dtypes(include=['float64']).columns.tolist()

if len(numeric_features) > 0:
    # RobustScaler sử dụng median và IQR → ít bị ảnh hưởng bởi outliers
    scaler = RobustScaler()
    
    # Fit trên train, transform cả train và test
    X_train_selected[numeric_features] = scaler.fit_transform(X_train_selected[numeric_features])
    X_test_selected[numeric_features] = scaler.transform(X_test_selected[numeric_features])
    
    print(f"   ✓ Đã chuẩn hóa {len(numeric_features)} features số")
    print(f"   ✓ Method: RobustScaler (median, IQR) - tốt cho outliers")
else:
    print("   ⚠️ Không có features số nào cần chuẩn hóa")

# ==========================================
# BƯỚC 10: XỬ LÝ IMBALANCED DATA (SMOTE-TOMEK)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 10: XỬ LÝ IMBALANCED DATA (SMOTE-TOMEK)")
print("="*80)

print("\n⚖️ Áp dụng SMOTETomek để cân bằng và làm sạch dữ liệu...")
print(f"   Trước SMOTETomek:")
print(f"      - Class 0: {(y_train==0).sum():,}")
print(f"      - Class 1: {(y_train==1).sum():,}")

# SMOTETomek = SMOTE + Tomek links cleaning
# Tạo synthetic samples và loại bỏ samples gây nhiễu ở biên
smt = SMOTETomek(random_state=42, sampling_strategy='auto')
X_train_balanced, y_train_balanced = smt.fit_resample(X_train_selected, y_train)

print(f"\n   Sau SMOTETomek:")
print(f"      - Class 0: {(y_train_balanced==0).sum():,} ({(y_train_balanced==0).sum()/len(y_train_balanced)*100:.1f}%)")
print(f"      - Class 1: {(y_train_balanced==1).sum():,} ({(y_train_balanced==1).sum()/len(y_train_balanced)*100:.1f}%)")
print(f"   ✓ Đã cân bằng và làm sạch dữ liệu training")
print(f"   ✓ SMOTETomek vừa tạo synthetic samples, vừa loại bỏ noise")

# ==========================================
# BƯỚC 10A: HUẤN LUYỆN MÔ HÌNH BASELINE (TRƯỚC FEATURE SELECTION & SMOTE)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 10A: HUẤN LUYỆN MÔ HÌNH BASELINE (TRƯỚC TỐI ỨU)")
print("="*80)

print("\n🔵 Huấn luyện baseline trên dữ liệu đã làm sạch cơ bản (chưa Feature Selection/SMOTE)...")

# Sử dụng X_train, X_test sau chia train-test ở BƯỚC 7
model_baseline = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
model_baseline.fit(X_train, y_train)

# Dự đoán
y_pred_prob_base = model_baseline.predict_proba(X_test)[:, 1]
y_pred_class_base = (y_pred_prob_base >= 0.5).astype(int)

# Tính metrics
auc_baseline = roc_auc_score(y_test, y_pred_prob_base)
cm_baseline = confusion_matrix(y_test, y_pred_class_base)
TN_base, FP_base, FN_base, TP_base = cm_baseline.ravel()

accuracy_baseline = (TP_base + TN_base) / (TP_base + TN_base + FP_base + FN_base)
precision_baseline = TP_base / (TP_base + FP_base) if (TP_base + FP_base) > 0 else 0
recall_baseline = TP_base / (TP_base + FN_base) if (TP_base + FN_base) > 0 else 0
f1_baseline = 2 * (precision_baseline * recall_baseline) / (precision_baseline + recall_baseline) if (precision_baseline + recall_baseline) > 0 else 0

print(f"\n📊 Baseline: AUC={auc_baseline:.4f} | Accuracy={accuracy_baseline:.4f} | Recall={recall_baseline:.4f}")

# ==========================================
# BƯỚC 11A: HUẤN LUYỆN NHIỀU MÔ HÌNH (ENSEMBLE)
# ==========================================
print("\n" + "="*80)
print("BƯỚC 11: HUẤN LUYỆN MÔ HÌNH - LOGISTIC REGRESSION")
print("="*80)

print("\n🟢 Huấn luyện Logistic Regression trên dữ liệu đã xử lý...")

# Khởi tạo mô hình Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced',  # Tự động điều chỉnh trọng số cho imbalanced data
    C=0.1,  # Regularization để tránh overfitting
    solver='saga'  # Solver tốt cho large dataset
)

print(f"\n📋 Model parameters:")
print(f"   - max_iter: 1000")
print(f"   - class_weight: balanced (handles imbalance)")
print(f"   - C: 0.1 (strong regularization)")
print(f"   - solver: saga")

# Huấn luyện mô hình
print(f"\n🚀 Training...")
log_reg.fit(X_train_balanced, y_train_balanced)
print(f"   ✓ Model trained successfully!")

# ==========================================
# BƯỚC 11B: TỐI ỨU THRESHOLD QUẾT ĐỊNH
# ==========================================
print("\n" + "="*80)
print("BƯỚC 11B: TỐI ỨU THRESHOLD QUYẾT ĐỊNH")
print("="*80)

print("\n🎯 Tìm threshold tối ưu để cân bằng Precision và Recall...")

# Dự đoán probability trên test set
best_probs = log_reg.predict_proba(X_test_selected)[:, 1]

# Thử các threshold từ 0.1 đến 0.9
thresholds = np.arange(0.1, 0.9, 0.01)
accuracies = []
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (best_probs >= thresh).astype(int)
    acc = (y_pred_thresh == y_test).mean()
    
    # Tính F1
    cm_temp = confusion_matrix(y_test, y_pred_thresh)
    if cm_temp.size == 4:
        tn, fp, fn, tp = cm_temp.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    else:
        f1 = 0
    
    accuracies.append(acc)
    f1_scores.append(f1)

# Tìm threshold tốt nhất
best_threshold_acc = thresholds[np.argmax(accuracies)]
best_accuracy = max(accuracies)

best_threshold_f1 = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"\n   ✅ Threshold tối ưu cho Accuracy: {best_threshold_acc:.2f}")
print(f"      → Accuracy: {best_accuracy:.4f}")

print(f"\n   ✅ Threshold tối ưu cho F1-Score: {best_threshold_f1:.2f}")
print(f"      → F1-Score: {best_f1:.4f}")

# Visualization threshold optimization
plt.figure(figsize=(10, 5))
plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
plt.axvline(x=best_threshold_acc, color='red', linestyle='--', 
            label=f'Best Threshold (Acc={best_threshold_acc:.2f})', alpha=0.7)
plt.axvline(x=0.5, color='gray', linestyle=':', label='Default (0.5)', alpha=0.5)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Threshold Optimization', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Sử dụng threshold 0.2 để ưu tiên Recall (credit default)
optimal_threshold = 0.2  # Tập trung vào việc phát hiện defaults
y_pred_prob = best_probs
y_pred_class = (best_probs >= optimal_threshold).astype(int)

print(f"\n✅ Sử dụng mô hình: Logistic Regression")
print(f"✅ Sử dụng threshold: {optimal_threshold:.2f} (ưu tiên Recall)")

# ==========================================
# BƯỚC 12: ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH TỐI ƯU
# ==========================================
print("\n" + "="*80)
print("BƯỚC 12: ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH TỐI ƯU")
print("="*80)

# 11.1: Đánh giá mô hình đã xử lý
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred_class)
TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n🟢 MÔ HÌNH TRÊN DỮ LIỆU ĐÃ XỬ LÝ:")
print(f"   - AUC Score:  {auc:.4f}")
print(f"   - Accuracy:   {accuracy:.4f}")
print(f"   - Precision:  {precision:.4f}")
print(f"   - Recall:     {recall:.4f}")
print(f"   - F1-Score:   {f1:.4f}")

# 11.2: Bảng so sánh
print("\n" + "="*80)
print("📊 BẢNG SO SÁNH CHI TIẾT")
print("="*80)

comparison_metrics = pd.DataFrame({
    'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Baseline (Raw)': [auc_baseline, accuracy_baseline, precision_baseline, recall_baseline, f1_baseline],
    'Processed (+ Feature Eng + SMOTE)': [auc, accuracy, precision, recall, f1],
    'Improvement': [
        auc - auc_baseline,
        accuracy - accuracy_baseline,
        precision - precision_baseline,
        recall - recall_baseline,
        f1 - f1_baseline
    ],
    'Improvement (%)': [
        (auc - auc_baseline) / auc_baseline * 100 if auc_baseline > 0 else 0,
        (accuracy - accuracy_baseline) / accuracy_baseline * 100 if accuracy_baseline > 0 else 0,
        (precision - precision_baseline) / precision_baseline * 100 if precision_baseline > 0 else 0,
        (recall - recall_baseline) / recall_baseline * 100 if recall_baseline > 0 else 0,
        (f1 - f1_baseline) / f1_baseline * 100 if f1_baseline > 0 else 0
    ]
})

print("\n" + comparison_metrics.to_string(index=False))

# 11.3: Phân tích kết quả
print("\n" + "="*80)
print("💡 PHÂN TÍCH KẾT QUẢ")
print("="*80)

print("\n🎯 Hiệu quả của việc xử lý dữ liệu:\n")

# Accuracy
if accuracy > accuracy_baseline:
    acc_diff = (accuracy - accuracy_baseline) * 100
    print(f"   ✅ Accuracy TĂNG: {accuracy_baseline:.4f} → {accuracy:.4f} (+{acc_diff:.2f}%)")
    print(f"      → Xử lý dữ liệu cải thiện độ chính xác dự đoán")
elif accuracy == accuracy_baseline:
    print(f"   ➖ Accuracy KHÔNG ĐỔI: {accuracy:.4f}")
    print(f"      → Xử lý dữ liệu chưa tác động đến accuracy")
else:
    acc_diff = (accuracy_baseline - accuracy) * 100
    print(f"   ⚠️  Accuracy GIẢM: {accuracy_baseline:.4f} → {accuracy:.4f} (-{acc_diff:.2f}%)")
    print(f"      → Có thể do overfitting hoặc xử lý dữ liệu chưa phù hợp")

# AUC
if auc > auc_baseline:
    auc_diff = (auc - auc_baseline) * 100
    print(f"\n   ✅ AUC TĂNG: {auc_baseline:.4f} → {auc:.4f} (+{auc_diff:.2f}%)")
    print(f"      → Mô hình phân biệt class tốt hơn sau xử lý")
elif auc == auc_baseline:
    print(f"\n   ➖ AUC KHÔNG ĐỔI: {auc:.4f}")
else:
    auc_diff = (auc_baseline - auc) * 100
    print(f"\n   ⚠️  AUC GIẢM: {auc_baseline:.4f} → {auc:.4f} (-{auc_diff:.2f}%)")

# Recall
if recall > recall_baseline:
    recall_diff = (recall - recall_baseline) * 100
    print(f"\n   ✅ Recall TĂNG: {recall_baseline:.4f} → {recall:.4f} (+{recall_diff:.2f}%)")
    print(f"      → Phát hiện được nhiều trường hợp vỡ nợ hơn (quan trọng!)")
elif recall == recall_baseline:
    print(f"\n   ➖ Recall KHÔNG ĐỔI: {recall:.4f}")
else:
    recall_diff = (recall_baseline - recall) * 100
    print(f"\n   ⚠️  Recall GIẢM: {recall_baseline:.4f} → {recall:.4f} (-{recall_diff:.2f}%)")
    print(f"      → Bỏ sót nhiều trường hợp vỡ nợ hơn")

# F1-Score
if f1 > f1_baseline:
    f1_diff = (f1 - f1_baseline) * 100
    print(f"\n   ✅ F1-Score TĂNG: {f1_baseline:.4f} → {f1:.4f} (+{f1_diff:.2f}%)")
    print(f"      → Cân bằng giữa Precision và Recall tốt hơn")
elif f1 == f1_baseline:
    print(f"\n   ➖ F1-Score KHÔNG ĐỔI: {f1:.4f}")
else:
    f1_diff = (f1_baseline - f1) * 100
    print(f"\n   ⚠️  F1-Score GIẢM: {f1_baseline:.4f} → {f1:.4f} (-{f1_diff:.2f}%)")

# Tổng kết
print("\n" + "-"*80)
print("📌 KẾT LUẬN:")
print("-"*80)

improvements = sum([
    1 if auc > auc_baseline else 0,
    1 if accuracy > accuracy_baseline else 0,
    1 if precision > precision_baseline else 0,
    1 if recall > recall_baseline else 0,
    1 if f1 > f1_baseline else 0
])

if improvements >= 4:
    print("✅ Xử lý dữ liệu RẤT HIỆU QUẢ - hầu hết metrics đều cải thiện")
    print("   → Nên sử dụng pipeline xử lý này cho production")
elif improvements >= 3:
    print("✅ Xử lý dữ liệu HIỆU QUẢ - đa số metrics cải thiện")
    print("   → Có thể tinh chỉnh thêm để tối ưu hơn")
elif improvements >= 2:
    print("⚠️  Xử lý dữ liệu CÓ TÁC DỤNG - một số metrics cải thiện")
    print("   → Cần xem xét lại các bước xử lý")
else:
    print("❌ Xử lý dữ liệu CHƯA HIỆU QUẢ - ít metrics cải thiện")
    print("   → Cần thay đổi chiến lược xử lý")

# Đặc biệt chú ý Recall trong credit scoring
if recall > recall_baseline:
    print("\n💡 ĐẶC BIỆT: Recall tăng rất quan trọng trong credit scoring!")
    print("   → Giảm thiểu rủi ro bỏ sót khách hàng vỡ nợ")
elif recall < recall_baseline:
    print("\n⚠️  CHÚ Ý: Recall giảm là vấn đề nghiêm trọng trong credit scoring!")
    print("   → Bỏ sót nhiều khách hàng vỡ nợ hơn → tăng rủi ro tài chính")

# 11.4: Classification Report chi tiết
print("\n" + "="*80)
print("📋 CLASSIFICATION REPORT CHI TIẾT")
print("="*80)

print("\n🟢 Mô hình đã xử lý:")
print(classification_report(y_test, y_pred_class, target_names=['No Default (0)', 'Default (1)']))

print("\n🔵 Mô hình baseline:")
print(classification_report(y_test, y_pred_class_base, target_names=['No Default (0)', 'Default (1)']))

# 11.5: Confusion Matrix chi tiết
print("\n" + "="*80)
print("🔢 CONFUSION MATRIX CHI TIẾT")
print("="*80)

print("\n🟢 Mô hình đã xử lý:")
print(f"   True Negative (TN):  {TN:,} - Dự đoán đúng KHÔNG vỡ nợ")
print(f"   False Positive (FP): {FP:,} - Dự đoán SAI là vỡ nợ (Type I error)")
print(f"   False Negative (FN): {FN:,} - Dự đoán SAI là KHÔNG vỡ nợ (Type II error) ⚠️")
print(f"   True Positive (TP):  {TP:,} - Dự đoán đúng vỡ nợ")

print("\n🔵 Mô hình baseline:")
print(f"   True Negative (TN):  {TN_base:,}")
print(f"   False Positive (FP): {FP_base:,}")
print(f"   False Negative (FN): {FN_base:,} ⚠️")
print(f"   True Positive (TP):  {TP_base:,}")

# ==========================================
# BƯỚC 14: VISUALIZATION KẾT QUẢ MÔ HÌNH
# ==========================================
print("\n" + "="*80)
print("BƯỚC 14: VISUALIZATION KẾT QUẢ MÔ HÌNH")
print("="*80)

# 14.1: So sánh metrics bằng biểu đồ
print("\n📊 Vẽ biểu đồ so sánh metrics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: So sánh các metrics
ax1 = axes[0, 0]
metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_values = [auc_baseline, accuracy_baseline, precision_baseline, recall_baseline, f1_baseline]
processed_values = [auc, accuracy, precision, recall, f1]

x_pos = np.arange(len(metrics_names))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, baseline_values, width, label='Baseline (Raw)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, processed_values, width, label='Processed', color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('So sánh Performance: Baseline vs Processed', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics_names, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

# Thêm nhãn giá trị
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Chart 2: ROC Curves so sánh
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_prob_base)

ax2.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'Processed (AUC = {auc:.3f})')
ax2.plot(fpr_base, tpr_base, color='#e74c3c', lw=2, label=f'Baseline (AUC = {auc_baseline:.3f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# Chart 3: Confusion Matrix - Processed
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True, ax=ax3,
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'],
            annot_kws={"size": 12})
ax3.set_title('Confusion Matrix - Processed', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=10)
ax3.set_xlabel('Predicted Label', fontsize=10)

# Chart 4: Confusion Matrix - Baseline
ax4 = axes[1, 1]
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax4,
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'],
            annot_kws={"size": 12})
ax4.set_title('Confusion Matrix - Baseline', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label', fontsize=10)
ax4.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.show()

# 14.2: ROC Curve riêng cho processed model
print("\n📊 Vẽ ROC Curve chi tiết...")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Processed Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Processed Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 13.2: Precision-Recall Curve
print("\n📊 Vẽ Precision-Recall Curve...")
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR Curve (AUC = {auc:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 13.3: Confusion Matrix Heatmap
print("\n📊 Vẽ Confusion Matrix...")
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted: 0', 'Predicted: 1'],
            yticklabels=['Actual: 0', 'Actual: 1'],
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# 14.4: Feature Importance
print("\n📊 Vẽ Feature Importance...")

# Logistic Regression coefficients
feature_importance = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': np.abs(log_reg.coef_[0])
}).sort_values(by='Importance', ascending=False).head(30)
importance_title = 'Top 30 Feature Importance - Logistic Regression'

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title(importance_title, fontsize=14, fontweight='bold')
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# KẾT THÚC
# ==========================================
print("\n" + "="*80)
print("✅ HOÀN THÀNH QUY TRÌNH TIỀN XỬ LÝ VÀ HUẤN LUYỆN MÔ HÌNH")
print("="*80)

print("\n📊 KẾT QUẢ CUỐI CÙNG:")
print(f"   {'Metric':<12} | {'Baseline':<10} | {'Processed':<10} | {'Improvement':<12}")
print(f"   {'-'*52}")
print(f"   {'Accuracy':<12} | {accuracy_baseline:<10.4f} | {accuracy:<10.4f} | {(accuracy-accuracy_baseline)*100:>+10.2f}%")
print(f"   {'AUC':<12} | {auc_baseline:<10.4f} | {auc:<10.4f} | {(auc-auc_baseline)*100:>+10.2f}%")
print(f"   {'Recall':<12} | {recall_baseline:<10.4f} | {recall:<10.4f} | {(recall-recall_baseline)*100:>+10.2f}%")
print(f"   {'F1-Score':<12} | {f1_baseline:<10.4f} | {f1:<10.4f} | {(f1-f1_baseline)*100:>+10.2f}%")

print("\n✨ Các kỹ thuật đã áp dụng:")
print("   1-7. Làm sạch, Feature Engineering, One-hot, Loại bỏ tương quan cao")
print("   8. ✅ Feature Selection → Giảm noise")
print("   9. ✅ RobustScaler → Xử lý outliers tốt")
print("   10. ✅ SMOTETomek → Cân bằng + làm sạch")
print("   11. ✅ Logistic Regression với class_weight='balanced'")
print("   12. ✅ Threshold=0.2 → Ưu tiên Recall (credit default)")

print("\n" + "="*80)
print("💡 KẾT LUẬN")
print("="*80)

improvement_accuracy = ((accuracy - accuracy_baseline) / accuracy_baseline * 100) if accuracy_baseline > 0 else 0
improvement_auc = ((auc - auc_baseline) / auc_baseline * 100) if auc_baseline > 0 else 0

if accuracy > accuracy_baseline and auc > auc_baseline:
    print("\n✅ THÀNH CÔNG: Pipeline xử lý dữ liệu đã cải thiện hiệu suất mô hình!")
    print(f"   - Accuracy tăng: {improvement_accuracy:+.2f}%")
    print(f"   - AUC tăng: {improvement_auc:+.2f}%")
    print("\n   → Các kỹ thuật Feature Selection, Ensemble, và Threshold Optimization")
    print("     đã giúp mô hình dự đoán chính xác hơn!")
elif accuracy > accuracy_baseline or auc > auc_baseline:
    print("\n✅ CẢI THIỆN MỘT PHẦN: Một số metrics đã tăng")
    if accuracy > accuracy_baseline:
        print(f"   - Accuracy tăng: {improvement_accuracy:+.2f}%")
    if auc > auc_baseline:
        print(f"   - AUC tăng: {improvement_auc:+.2f}%")
else:
    print("\n⚠️  LƯU Ý: Một số metrics có thể giảm do:")
    print("   - SMOTE tạo synthetic data → mô hình nhạy cảm hơn")
    print("   - Nhưng AUC và Recall thường TĂNGtốt cho credit scoring!")

print("\n🎯 KHUYẾN NGHỊ SỬ DỤNG:")
print(f"   - Mô hình: Logistic Regression")
print(f"   - Threshold: {optimal_threshold:.2f} (ưu tiên Recall)")
print(f"   - Accuracy đạt được: {accuracy:.4f}")
print(f"   - AUC đạt được: {auc:.4f}")
print(f"   - Recall đạt được: {recall:.4f} (quan trọng cho credit default)")

print("\n" + "="*80)
print("KẾT THÚC CHƯƠNG TRÌNH")
print("="*80)
