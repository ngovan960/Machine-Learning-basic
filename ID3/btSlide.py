import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# =========================
# 1. Tạo dữ liệu
# =========================

data = {
    "age": [
        "<=30", "<=30", "31...40", ">40", ">40",
        ">40", "31...40", "<=30", "<=30", ">40",
        "<=30", "31...40", "31...40", ">40"
    ],

    "income": [
        "high", "high", "high", "medium", "low",
        "low", "low", "medium", "low", "medium",
        "medium", "medium", "high", "medium"
    ],

    "student": [
        "no", "no", "no", "no", "yes",
        "yes", "yes", "no", "yes", "yes",
        "yes", "no", "yes", "no"
    ],

    "credit_rating": [
        "fair", "excellent", "fair", "fair", "fair",
        "excellent", "excellent", "fair", "fair", "fair",
        "excellent", "excellent", "fair", "excellent"
    ],

    "buys_computer": [
        "no", "no", "yes", "yes", "yes",
        "no", "yes", "no", "yes", "yes",
        "yes", "yes", "yes", "no"
    ]
}

df = pd.DataFrame(data)

print("Dữ liệu:")
print(df)


# =========================
# 2. Tách X và y
# =========================

X = df.drop("buys_computer", axis=1)
y = df["buys_computer"]


# =========================
# 3. Chuyển dữ liệu chữ
#    thành số
# =========================

X = pd.get_dummies(X)

print("\nDữ liệu sau khi mã hóa:")
print(X)


# =========================
# 4. Xây dựng cây ID3
# =========================

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X, y)


# =========================
# 5. Dự đoán
# =========================

y_pred = model.predict(X)

print("\nNhãn thực tế:")
print(y.values)

print("\nNhãn dự đoán:")
print(y_pred)

print("\nQuy tắc của cây ID3:")
print(export_text(model, feature_names=list(X.columns), decimals=2))


# =========================
# 6. Vẽ cây quyết định
# =========================

plt.figure(figsize=(24, 14))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
    precision=2,
    fontsize=11
)

plt.title("Cây quyết định ID3 - Dự đoán mua máy tính", fontsize=18, pad=20)
plt.tight_layout()
plt.savefig("id3_tree.png", dpi=200, bbox_inches="tight")
print("\nĐã lưu hình cây rõ nét tại: id3_tree.png")
plt.show()
