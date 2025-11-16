# 📊 Credit Default Prediction: Dirty Data Cleaning Pipeline

## 📋 Mô tả

Notebook `credit_default_dirty_full_pipeline.ipynb` là một pipeline hoàn chỉnh để xử lý dữ liệu siêu bẩn và xây dựng mô hình dự đoán vỡ nợ thẻ tín dụng. Pipeline này bao gồm:

- **EDA chi tiết** trên dữ liệu bẩn (missing 25%, outliers, noise, corruption)
- **Baseline Raw**: Đánh giá model ngay sau EDA (trên dữ liệu bẩn)
- **Xử lý đầy đủ corruption**: Missing, outliers, label flipping, shuffle, noise, negative values
- **Feature Engineering + Scaling + Feature Selection**
- **So sánh 3 models**: Raw Baseline vs Clean Baseline vs Final Model

## 🎯 Mục tiêu

- Khôi phục dữ liệu bẩn về trạng thái gần ban đầu
- Đánh giá impact của từng bước preprocessing
- Xây dựng mô hình tối ưu với performance cao

## 🚀 Cách chạy

### Yêu cầu
- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Cài đặt
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Chạy notebook
1. Đặt file `super_dirty_default_credit.csv` trong cùng thư mục
2. Mở notebook trong Jupyter/VS Code
3. Run từ đầu đến cuối (restart kernel trước khi chạy)

## 📊 Các bước chính

### 1. EDA (Exploratory Data Analysis)
- Phân tích missing values, outliers, correlations
- Phân bố target và features chính
- Phát hiện các vấn đề dữ liệu bẩn

### 2. Baseline Raw
- Train RandomForest trên dữ liệu bẩn (chỉ fill missing cơ bản)
- Đánh giá performance baseline trước xử lý

### 3. Data Cleaning Pipeline
- **Negative values**: Abs cho PAY_AMT, BILL_AMT
- **Missing values**: Median cho numeric, mode cho categorical
- **Outliers**: IQR clipping + percentile 99 cho LIMIT_BAL/AGE
- **Shuffle**: Rolling median smoothing
- **Noise**: Median filter
- **Label flipping**: Model-based detection và correction

### 4. Feature Engineering
- AVG_BILL, AVG_PAY, UTILIZATION, PAY_STAB
- Áp dụng riêng biệt trên Train/Test (tránh data leakage)

### 5. Scaling & Feature Selection
- RobustScaler (fit trên Train)
- RandomForest feature importance (chỉ trên Train)

### 6. Model Training & Comparison
- **Baseline Clean**: RF trên dữ liệu sạch cơ bản
- **Final Model**: RF với full pipeline (FE + FS)
- So sánh metrics và ROC curves

## 📈 Kết quả mong đợi

- **Raw Baseline**: ~0.75-0.80 AUC (dữ liệu bẩn)
- **Clean Baseline**: ~0.82-0.85 AUC (sau preprocessing cơ bản)
- **Final Model**: ~0.85-0.88 AUC (full pipeline)

## 📁 Cấu trúc file

```
├── credit_default_dirty_full_pipeline.ipynb  # Main notebook
├── super_dirty_default_credit.csv           # Input data (bẩn)
└── README.md                                # This file
```
