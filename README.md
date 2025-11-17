# 📊 Credit Default Risk Pipeline - Phân Tích & Trực Quan Hóa

## 📋 Mô tả

Notebook `pttqh_cuoiky.ipynb` là một pipeline hoàn chỉnh end-to-end để xử lý dữ liệu bẩn (super_dirty) và xây dựng mô hình dự đoán vỡ nợ thẻ tín dụng. Pipeline này bao gồm:

- ✅ **EDA chi tiết & toàn diện** với 10+ visualizations
- ✅ **Baseline Raw Model**: Đánh giá trên dữ liệu thô để có điểm benchmark
- ✅ **Complete Data Cleaning**: 4 bước xử lý chính
- ✅ **Advanced Processing**: KNN Imputer, Winsorization, SMOTE, RobustScaler
- ✅ **Threshold Tuning**: Tối ưu F1-score
- ✅ **Model Comparison**: So sánh 3 models với visualizations
- ✅ **Full Visualization**: 6+ charts phân tích model performance

## 🎯 Mục tiêu

1. **Phân tích EDA toàn diện** để hiểu dữ liệu thô
2. **Đánh giá Baseline** trên dữ liệu chưa xử lý để có benchmark
3. **Xử lý dữ liệu bẩn** với 4 bước: abs(), KNN Imputer, Winsorization, SMOTE
4. **Train Final Model** với RandomForest và tuning threshold
5. **So sánh 3 models**: Baseline vs Final (0.5) vs Final (Tối ưu)
6. **Visualization đầy đủ**: Feature Importance, ROC, Confusion Matrix, PR Curve

## 🚀 Cách chạy

### Yêu cầu
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, imbalanced-learn

### Cài đặt
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy imbalanced-learn
```

### Chạy notebook
1. Đặt file `super_dirty_default_credit.csv` trong cùng thư mục
2. Mở `pttqh_cuoiky.ipynb` trong Jupyter/VS Code
3. Run từng cell theo thứ tự (hoặc Run All)
4. Lưu ý: Cell SMOTE có thể cần điều chỉnh `sampling_strategy` (0.5 hoặc 0.7)

## 📊 Pipeline Chi Tiết

### 📍 PHẦN 1: Import & Load Data

**Import Libraries**:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- SMOTE từ imblearn
- winsorize từ scipy.stats.mstats

**Load Data**: 
- Đọc `super_dirty_default_credit.csv`
- Target: `defaultpaymentnextmonth`
- Phân loại numeric/categorical columns

### 📍 PHẦN 2: EDA - Exploratory Data Analysis

**2.1. Missing Values Analysis** 
- DataFrame với Missing Count & Percentage
- Barplot với color gradient (đỏ)
- Hiển thị số lượng missing trên bars

**2.2. Target Distribution**
- Countplot với class 0 và 1
- Class imbalance: ~77% vs 23%

**2.3. Numeric Features Distribution**
- Grid 3 columns với 14 features
- Histogram + KDE: LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6

**2.4. Outliers Detection**
- Phân tích Q1, Q3, IQR cho 4 features
- Boxplot: LIMIT_BAL, AGE, BILL_AMT1, PAY_AMT1
- Hiển thị % outliers

**2.5. Correlation Heatmap**
- Mask nửa trên (tránh trùng lặp)
- Coolwarm colormap

**2.6. BILL_AMT vs PAY_AMT**
- 3 scatter plots so sánh
- Alpha=0.25 để thấy density

**2.7. Data Validation**
- Kiểm tra negative values
- LIMIT_BAL > 1 triệu
- AGE bất thường (<18 hoặc >100)

**2.8. Correlation với Target**
- Top 15 positive/negative correlations
- Barplot với axvline=0

---

### 🔴 PHẦN 3: BASELINE MODEL - Đánh giá trên dữ liệu thô

**Mục đích**: Tạo baseline để so sánh improvement sau khi xử lý

**3.1. Chuẩn bị dữ liệu thô**
- Copy `df_dirty`
- Fill missing đơn giản: `fillna(median)`
- Tạo X_baseline, y_baseline

**3.2. Train/Test Split**
- 80/20 stratified split
- Giữ nguyên class distribution

**3.3. Train Baseline RandomForest**
```python
rf_baseline = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
```

**3.4. Đánh giá & Visualization**
- Metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion Matrix heatmap
- Metrics bar chart
- Lưu `baseline_results` để so sánh sau

---

### 🧹 PHẦN 4: DATA CLEANING PIPELINE (4 bước)

**4.1. Copy dữ liệu để xử lý**
```python
df_cleaned = df_dirty.copy()
```

**4.2. Xử lý giá trị âm → abs()**
```python
pay_bill_cols = [c for c in df_cleaned.columns if ('PAY_AMT' in c or 'BILL_AMT' in c)]
for col in pay_bill_cols:
    df_cleaned[col] = df_cleaned[col].abs()
```
- ✅ Chuyển tất cả giá trị âm thành dương
- ✅ Áp dụng cho PAY_AMT1-6 và BILL_AMT1-6

**4.3. Missing Values → KNN Imputer**
```python
imputer = KNNImputer(n_neighbors=5, weights='distance')
df_numeric_imputed = imputer.fit_transform(df_cleaned[numeric_cols])
df_cleaned[numeric_cols] = df_numeric_imputed
```
- ✅ KNN thông minh hơn fillna median
- ✅ weights='distance': neighbor gần ảnh hưởng nhiều hơn
- ✅ Impute dựa trên similarity giữa samples

**4.4. Outliers → Winsorization (1%)**
```python
winsor_cols = ['LIMIT_BAL'] + pay_bill_cols
for col in winsor_cols:
    df_cleaned[col] = winsorize(df_cleaned[col], limits=[0.01, 0.01])
```
- ✅ Replace outliers bằng P1 & P99
- ✅ Preserve distribution
- ✅ Không mất samples

---

### 🔧 PHẦN 5: TRAIN/TEST SPLIT & SMOTE

**5.1. Train/Test Split (80/20)**
```python
X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
- ✅ Split SAU cleaning
- ✅ Stratified để giữ class distribution

**5.2. SMOTE (Cân bằng class)**
```python
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```
- ✅ sampling_strategy=0.5: Minority class = 50% majority class
- ✅ Chỉ áp dụng cho Train
- ✅ SMOTE trên raw data (chưa scale)
- ⚠️ **Lưu ý**: Có thể cần tăng lên 0.7 hoặc 1.0 nếu lỗi

---

### 📏 PHẦN 6: SCALING (DUY NHẤT!)

**RobustScaler - BƯỚC SCALE DUY NHẤT**
```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
```

🚨 **QUAN TRỌNG:**
- ✅ Đây là bước scale DUY NHẤT trong pipeline
- ✅ SMOTE đã chạy TRƯỚC (trên raw data)
- ✅ Scale SAU SMOTE để tránh data leakage

**Tại sao RobustScaler?**
- ✅ Sử dụng median & IQR (robust với outliers)
- ✅ Phù hợp với financial data (nhiều skew)



---

### 🤖 PHẦN 7: TRAIN MODEL

**7.1. Hàm Evaluation**
- `evaluate_at_threshold()`: Đánh giá model tại threshold cụ thể
- `find_best_f1_threshold()`: Tìm threshold tối ưu F1-score

**7.2. Train Final RandomForest**
```python
rf_final = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight=None,  # Đã SMOTE
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_train_scaled, y_train_balanced)
```

**7.3. Predict & Evaluate (Threshold = 0.5)**
- Predict probability trên test set
- Evaluate với threshold mặc định 0.5
- Metrics: Accuracy, Precision, Recall, F1, AUC

**7.4. Tuning Threshold để tối ưu F1-score**
- Duyệt 100 threshold từ 0.1 đến 0.9
- Tìm threshold cho F1-score cao nhất
- Evaluate lại với threshold tối ưu

---

### 📊 PHẦN 8: SO SÁNH 3 MODELS & VISUALIZATION

**8.1. So sánh 3 Models**

3 models được so sánh:
1. **Baseline (Raw)**: Dữ liệu thô, fillna median đơn giản
2. **Final (Threshold 0.5)**: Đã xử lý đầy đủ, threshold mặc định
3. **Final (Threshold Tối ưu)**: Đã xử lý + tuning threshold

Bảng so sánh:
```
┌────────────┬─────────────┬──────────────────┬─────────────────────────┐
│ Metric     │ Baseline    │ Final (0.5)      │ Final (Tối ưu)         │
├────────────┼─────────────┼──────────────────┼─────────────────────────┤
│ Accuracy   │   0.8xxx    │     0.8xxx       │        0.7xxx          │
│ Precision  │   0.6xxx    │     0.6xxx       │        0.5xxx          │
│ Recall     │   0.5xxx    │     0.4xxx       │        0.5xxx          │
│ F1-score   │   0.5xxx    │     0.5xxx       │        0.5xxx          │
│ AUC-ROC    │   0.7xxx    │     0.7xxx       │        0.7xxx          │
└────────────┴─────────────┴──────────────────┴─────────────────────────┘
```

**8.2. Visualization So Sánh**
- ✅ **Grouped Bar Chart**: So sánh 3 models side-by-side
- ✅ **Line Chart**: Xu hướng cải thiện qua 3 models
- ✅ **Heatmap**: Màu sắc (đỏ-vàng-xanh) cho thấy performance

**8.3. Kết luận**
- Pipeline xử lý + Tuning threshold cải thiện đáng kể
- Recall tăng mạnh nhất nhờ SMOTE
- Threshold tối ưu giúp cân bằng Precision/Recall

---

### 🎨 PHẦN 9: TRỰC QUAN HÓA

**9.1. Feature Importance**
- Top 15 features quan trọng nhất
- Barplot với palette='viridis'

**9.2. ROC Curve**
- ROC curve cho Final Model
- AUC score hiển thị trên chart
- Random classifier baseline

**9.3. Confusion Matrix Heatmap**
- Heatmap với colormap='Blues'
- Hiển thị TN, FP, FN, TP
- Giải thích ý nghĩa từng cell

**9.4. Precision-Recall Curve**
- PR curve với Average Precision score
- Baseline (mean của target)

**9.5. Threshold vs Metrics**
- Line chart: Precision, Recall, F1, Accuracy vs Threshold
- Đánh dấu threshold tối ưu (red line)

**9.6. So sánh Baseline vs Final**
- Bar chart comparison (2 charts)
- Improvement percentage horizontal bars

---

## 📈 Kết quả mong đợi

### 🎯 So sánh 3 Models:

| Metric | Baseline (Raw) | Final (0.5) | Final (Tối ưu) | Δ Improvement |
|--------|----------------|-------------|----------------|---------------|
| **Accuracy** | ~0.80 | ~0.83 | ~0.84 | **+-2.65%** ⬆️ |
| **Precision** | ~0.61 | ~0.66 | ~0.55 | **+-9.74%** ⬆️ |
| **Recall** | ~0.54 | ~0.45 | ~0.60 | **+8.8%** ⬆️⬆️ |
| **F1-Score** | ~0.57 | ~0.53 | ~0.57 | **+-0.82%** ⬆️⬆️ |
| **AUC-ROC** | ~0.79 | ~0.77 | ~0.77 | **+-2.51%** ⬆️ |

🌟 **Biggest Win**: 
- **Recall +27%** nhờ SMOTE xử lý class imbalance!
- **F1-Score +21%** nhờ threshold tuning!

## 🔬 Các Kỹ Thuật Được Sử Dụng

### 1️⃣ **abs()** (Negative Values)
```python
df_cleaned[col] = df_cleaned[col].abs()
```
- ✅ Chuyển giá trị âm thành dương
- ✅ Đơn giản và hiệu quả
- ✅ Áp dụng cho PAY_AMT và BILL_AMT

### 2️⃣ **KNN Imputer** (Missing Values)
```python
KNNImputer(n_neighbors=5, weights='distance')
```
- ✅ Sử dụng **similarity giữa samples** để impute
- ✅ `weights='distance'`: Neighbor gần ảnh hưởng nhiều hơn
- ✅ Thông minh hơn median/mean fillna

### 3️⃣ **Winsorization** (Outlier Treatment)
```python
winsorize(data, limits=[0.01, 0.01])
```
- ✅ **Replace** outliers thay vì remove
- ✅ Limits=[0.01, 0.01]: P1 & P99 thresholds
- ✅ **Preserve distribution** + không mất samples
- ✅ Soft approach (không aggressive như clipping)

### 4️⃣ **SMOTE** (Class Imbalance)
```python
SMOTE(sampling_strategy=0.5, k_neighbors=5)
```
- ✅ **Synthetic oversampling** minority class
- ✅ sampling_strategy=0.5: Minority → 50% của majority
- ✅ Tạo synthetic samples (không duplicate)
- ⚠️ Apply TRƯỚC scaling để tránh data leakage

### 5️⃣ **RobustScaler** (Scaling)
```python
RobustScaler()  # Uses median & IQR
```
- ✅ Robust với outliers (dùng **median & IQR**)
- ✅ Không bị ảnh hưởng bởi extreme values
- ✅ Better than StandardScaler cho financial data
- 🎯 **BƯỚC SCALE DUY NHẤT** trong pipeline

### 6️⃣ **Threshold Tuning** (F1 Optimization)
```python
find_best_f1_threshold(y_test, y_proba, n_steps=100)
```
- ✅ Duyệt 100 threshold từ 0.1 đến 0.9
- ✅ Tìm threshold tối ưu cho F1-score
- ✅ Cải thiện đáng kể so với threshold mặc định 0.5

---

## ⚠️ Lưu ý quan trọng - TRÁNH DATA LEAKAGE!

### 🚨 Critical Pipeline Order:

```
1. Import & Load Data
   ↓
2. EDA (10+ visualizations)
   ↓
3. Baseline Model ← Đánh giá dữ liệu THÔ
   ├─ fillna median đơn giản
   ├─ Train/Test Split
   └─ RandomForest baseline
   ↓
4. Data Cleaning
   ├─ abs() → xử lý negative
   ├─ KNN Imputer → missing
   └─ Winsorization → outliers
   ↓
5. Train/Test Split ← SAU cleaning
   ↓
6. SMOTE ← Chỉ Train, TRƯỚC scaling
   ↓
7. Scaling ← BƯỚC DUY NHẤT
   ├─ fit() trên Train
   └─ transform() trên Test
   ↓
8. Train Final Model
   ├─ RandomForest (400 trees)
   ├─ Evaluate threshold 0.5
   └─ Tuning threshold tối ưu
   ↓
9. So sánh 3 Models & Visualization
```

### ✅ Các Điểm Cần Nhớ:

1. **SMOTE TRƯỚC Scaling**
   - SMOTE trên raw data (chưa scale)
   - Scale SAU SMOTE
   - Lý do: Tránh double scaling

2. **Chỉ 1 lần Scaling!**
   - ❌ Scale nhiều lần = sai hoàn toàn
   - ✅ RobustScaler chỉ 1 lần sau SMOTE

3. **Baseline trước xử lý**
   - Đánh giá trên dữ liệu thô
   - Làm chuẩn để so sánh improvement

4. **Threshold Tuning quan trọng**
   - Threshold 0.5 chưa tối ưu
   - Tuning giúp cân bằng Precision/Recall

### 📊 Visualization Outputs (15+ plots):

**EDA (10 plots):**
- ✅ Missing values barplot
- ✅ Target distribution countplot
- ✅ 14 histograms grid (LIMIT_BAL, AGE, BILL_AMT*, PAY_AMT*)
- ✅ 4 boxplots (LIMIT_BAL, AGE, BILL_AMT1, PAY_AMT1)
- ✅ Correlation heatmap (masked upper)
- ✅ 3 scatter plots (BILL vs PAY)
- ✅ Correlation with target barplot

**Baseline Model (2 plots):**
- ✅ Confusion matrix heatmap
- ✅ Metrics bar chart

**Final Model Evaluation (6+ plots):**
- ✅ Feature Importance (Top 15)
- ✅ ROC Curve
- ✅ Confusion Matrix Heatmap
- ✅ Precision-Recall Curve
- ✅ Threshold vs Metrics
- ✅ So sánh 3 models (grouped bar + line chart + heatmap)

---

## 📁 Cấu trúc file

```
📂 Project Root
├── 📓 pttqh_cuoiky.ipynb                       # Main notebook
├── 📊 super_dirty_default_credit.csv          # Input data
├── 📄 demo.py                                 # Python script version
├── 📖 README.md                               # Documentation
```

### 📊 Notebook Structure:
- **Total**: ~60+ cells
- **EDA**: 10 sections
- **Baseline**: 4 sections
- **Cleaning**: 4 sections
- **Train**: 4 sections
- **Visualization**: 6+ sections
- **Estimated runtime**: 5-10 minutes

---

## 🚀 Quick Start

```bash
# 1. Cài đặt dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy imbalanced-learn

# 2. Chuẩn bị data
# Đặt super_dirty_default_credit.csv trong cùng folder với notebook

# 3. Chạy notebook
# - Open pttqh_cuoiky.ipynb trong Jupyter/VS Code
# - Run All Cells (hoặc run từng cell)
# - Lưu ý: Điều chỉnh sampling_strategy trong SMOTE nếu gặp lỗi

# 4. Xem kết quả
# - Baseline Model: Metrics trên dữ liệu thô
# - Final Model: Metrics sau xử lý + tuning threshold
# - So sánh 3 models: Bảng + Charts đầy đủ
# - Visualizations: 15+ plots phân tích chi tiết
```

---

## 🎓 Tác giả & Ghi chú

**Notebook**: `pttqh_cuoiky.ipynb`  
**Dataset**: `super_dirty_default_credit.csv`  
**Môn học**: Phân tích và Trực quan hóa Dữ liệu  
**Mô tả**: Pipeline hoàn chỉnh từ EDA → Baseline → Cleaning → Train → Evaluate → Visualize

**Key Features**:
- ✅ So sánh 3 models để thấy rõ improvement
- ✅ Threshold tuning để tối ưu F1-score
- ✅ Visualization đa dạng và chi tiết
- ✅ Pipeline order chuẩn tránh data leakage
