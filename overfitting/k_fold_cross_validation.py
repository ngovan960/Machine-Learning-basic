import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# =========================================================
# 1. ĐỌC DỮ LIỆU
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(SCRIPT_DIR, "data_1.csv")
)


# =========================================================
# 2. KIỂM TRA DỮ LIỆU
# =========================================================

print("5 dòng đầu:")
print(df.head())

print("\nKích thước dữ liệu:")
print(df.shape)

print("\nTên các cột:")
print(df.columns)

print("\nKiểu dữ liệu:")
print(df.dtypes)


# =========================================================
# 3. TÁCH X VÀ y
# =========================================================

X = df.drop(
    columns=["gia_nha_trieu"]
)

y = df["gia_nha_trieu"]


# =========================================================
# 4. TÁCH 20% TEST RIÊNG
# =========================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n================================")
print("CHIA DỮ LIỆU")
print("================================")

print("Tổng số dữ liệu:", len(X))

print(
    "Train + Validation:",
    len(X_train_val)
)

print(
    "Test riêng:",
    len(X_test)
)


# =========================================================
# 5. TẠO K-FOLD
# =========================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# 6. CHẠY K-FOLD
# =========================================================

fold_results = []


for fold, (train_index, val_index) in enumerate(
    kf.split(X_train_val),
    start=1
):

    # -----------------------------------------
    # Lấy dữ liệu Train
    # -----------------------------------------

    X_train = X_train_val.iloc[train_index]

    y_train = y_train_val.iloc[train_index]


    # -----------------------------------------
    # Lấy dữ liệu Validation
    # -----------------------------------------

    X_val = X_train_val.iloc[val_index]

    y_val = y_train_val.iloc[val_index]


    # -----------------------------------------
    # Tạo Linear Regression
    # -----------------------------------------

    model = LinearRegression()


    # -----------------------------------------
    # Train
    # -----------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------
    # Dự đoán Train
    # -----------------------------------------

    y_train_pred = model.predict(
        X_train
    )


    # -----------------------------------------
    # Dự đoán Validation
    # -----------------------------------------

    y_val_pred = model.predict(
        X_val
    )


    # -----------------------------------------
    # MAE
    # -----------------------------------------

    train_mae = mean_absolute_error(
        y_train,
        y_train_pred
    )

    val_mae = mean_absolute_error(
        y_val,
        y_val_pred
    )


    # -----------------------------------------
    # R²
    # -----------------------------------------

    train_r2 = r2_score(
        y_train,
        y_train_pred
    )

    val_r2 = r2_score(
        y_val,
        y_val_pred
    )


    # -----------------------------------------
    # Lưu kết quả
    # -----------------------------------------

    fold_results.append({

        "fold": fold,

        "train_mae": train_mae,

        "val_mae": val_mae,

        "train_r2": train_r2,

        "val_r2": val_r2,

        "train_index": train_index,

        "val_index": val_index

    })


    # -----------------------------------------
    # In kết quả Fold
    # -----------------------------------------

    print(
        f"\n========== FOLD {fold} =========="
    )

    print(
        "Train:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_val)
    )

    print(
        "Train MAE:",
        train_mae
    )

    print(
        "Validation MAE:",
        val_mae
    )

    print(
        "Train R²:",
        train_r2
    )

    print(
        "Validation R²:",
        val_r2
    )


# =========================================================
# 7. CHỌN FOLD TỐT NHẤT
# =========================================================

best_fold = max(
    fold_results,
    key=lambda x: x["val_r2"]
)


print("\n================================")
print("FOLD TỐT NHẤT")
print("================================")

print(
    "Fold:",
    best_fold["fold"]
)

print(
    "Validation R²:",
    best_fold["val_r2"]
)

print(
    "Validation MAE:",
    best_fold["val_mae"]
)


# =========================================================
# 8. LẤY ĐÚNG TRAIN DATA CỦA FOLD TỐT NHẤT
# =========================================================

best_train_index = best_fold["train_index"]

best_val_index = best_fold["val_index"]


X_train_best = X_train_val.iloc[
    best_train_index
]

y_train_best = y_train_val.iloc[
    best_train_index
]


X_val_best = X_train_val.iloc[
    best_val_index
]

y_val_best = y_train_val.iloc[
    best_val_index
]


print("\n================================")
print("DỮ LIỆU FOLD TỐT NHẤT")
print("================================")

print(
    "Train:",
    len(X_train_best)
)

print(
    "Validation:",
    len(X_val_best)
)

print(
    "Test riêng:",
    len(X_test)
)


# =========================================================
# 9. TRAIN MODEL BẰNG CHÍNH TRAIN DATA CỦA FOLD TỐT NHẤT
# =========================================================

final_model = LinearRegression()


final_model.fit(
    X_train_best,
    y_train_best
)


# =========================================================
# 10. DỰ ĐOÁN TRÊN TEST RIÊNG 20%
# =========================================================

y_test_pred = final_model.predict(
    X_test
)


# =========================================================
# 11. ĐÁNH GIÁ TEST
# =========================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# =========================================================
# 12. KẾT QUẢ TEST CUỐI CÙNG
# =========================================================

print("\n================================")
print("TEST CUỐI CÙNG")
print("================================")

print(
    "Số mẫu Train:",
    len(X_train_best)
)

print(
    "Số mẫu Test:",
    len(X_test)
)

print(
    "Test MAE:",
    test_mae
)

print(
    "Test R²:",
    test_r2
)


# =========================================================
# 13. SO SÁNH TRAIN VÀ TEST
# =========================================================

y_train_best_pred = final_model.predict(
    X_train_best
)


final_train_mae = mean_absolute_error(
    y_train_best,
    y_train_best_pred
)

final_train_r2 = r2_score(
    y_train_best,
    y_train_best_pred
)


print("\n================================")
print("TRAIN VS TEST")
print("================================")

print(
    "Train MAE:",
    final_train_mae
)

print(
    "Test MAE:",
    test_mae
)

print(
    "Train R²:",
    final_train_r2
)

print(
    "Test R²:",
    test_r2
)

print(
    "Chênh lệch R²:",
    final_train_r2 - test_r2
)

print(
    "Chênh lệch MAE:",
    test_mae - final_train_mae
)


# =========================================================
# 14. HIỂN THỊ GIÁ TRỊ THẬT VÀ DỰ ĐOÁN
# =========================================================

print("\n================================")
print("GIÁ TRỊ TEST")
print("================================")

for i in range(len(y_test)):

    print(
        f"Mẫu {i + 1}: "
        f"Thực tế = {y_test.iloc[i]:.2f} | "
        f"Dự đoán = {y_test_pred[i]:.2f}"
    )