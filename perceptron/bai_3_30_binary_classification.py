import sys
import io
import os
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm đường dẫn thư mục hiện tại vào sys.path để tránh lỗi import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification


# =====================================================================
# LỚP PERCEPTRON TỰ XÂY DỰNG THEO CHUẨN MACHINE LEARNING CƠ BẢN (MLCB)
# https://machinelearningcoban.com/2017/01/21/perceptron/
# =====================================================================

class Perceptron:
    """
    Thuật toán Perceptron cho bài toán phân loại nhị phân (Binary Classification).
    Được tối ưu theo phong cách và công thức của Machine Learning Cơ Bản.
    """
    def __init__(self, eta=1.0, max_iter=1000, random_state=None, learning_rate=None, fit_intercept=True):
        self.eta = learning_rate if learning_rate is not None else eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.w = None              # Vector trọng số mở rộng (d+1, 1): [w0, w1, ..., wd]^T
        self.w_history = []        # Lưu lại lịch sử cập nhật
        self.mis_points = []       # Lưu chỉ số các điểm bị phân lớp sai
        self.classes_ = None

    @property
    def weights(self):
        """Trả về vector trọng số đặc trưng (w1, ..., wd)"""
        return self.w[1:].flatten() if self.w is not None else None

    @property
    def bias(self):
        """Trả về hệ số bias w0"""
        return self.w[0, 0] if self.w is not None else 0.0

    def fit(self, X, y):
        """
        Huấn luyện mô hình Perceptron trên tập dữ liệu (X, y).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Lưu lại 2 nhãn gốc và ánh xạ về {-1, +1}
        y_raw = np.asarray(y).flatten()
        self.classes_ = np.unique(y_raw)
        if len(self.classes_) != 2:
            raise ValueError(f"Perceptron yêu cầu bài toán 2 lớp, tìm thấy {len(self.classes_)} lớp: {self.classes_}")

        y_mlcb = np.where(y_raw == self.classes_[1], 1.0, -1.0).reshape(1, -1)

        # Đưa ma trận X về dạng chuẩn (d, N) theo MLCB
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] >= X_arr.shape[1] and X_arr.shape[0] == len(y_raw):
            X_arr = X_arr.T

        # Mở rộng ma trận Xbar (d+1, N) với hàng đầu toàn 1 nếu fit_intercept=True
        N = X_arr.shape[1]
        if self.fit_intercept:
            Xbar = np.concatenate((np.ones((1, N)), X_arr), axis=0)
        else:
            Xbar = X_arr

        d_dim = Xbar.shape[0]

        # Khởi tạo vector trọng số ban đầu w0 = 0
        w_curr = np.zeros((d_dim, 1))
        self.w_history = [w_curr]
        self.mis_points = []

        # Quá trình lặp cập nhật Perceptron (PLA)
        for epoch in range(self.max_iter):
            mix_id = np.random.permutation(N)
            for i in range(N):
                idx = mix_id[i]
                xi = Xbar[:, idx].reshape(-1, 1)
                yi = y_mlcb[0, idx]

                # Dự đoán: sgn(w^T * x_i)
                y_pred = np.sign(np.dot(w_curr.T, xi))[0, 0]
                if y_pred == 0:
                    y_pred = 1.0

                # Nếu phân lớp sai: cập nhật w_new = w + eta * y_i * x_i
                if y_pred != yi:
                    self.mis_points.append(idx)
                    w_curr = w_curr + self.eta * yi * xi
                    self.w_history.append(w_curr)

            # Kiểm tra hội tụ (tất cả các điểm được phân lớp đúng)
            all_preds = np.sign(np.dot(w_curr.T, Xbar))
            all_preds[all_preds == 0] = 1.0
            if np.array_equal(all_preds, y_mlcb):
                break

        self.w = w_curr
        return self

    def predict(self, X):
        """
        Dự báo nhãn cho tập dữ liệu mới X.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        
        # Nếu X là (N, d) mà mô hình kỳ vọng (d, N), thực hiện chuyển vị
        expected_features = (self.w.shape[0] - 1) if self.fit_intercept else self.w.shape[0]
        if X_arr.shape[1] == expected_features:
            X_arr = X_arr.T

        N = X_arr.shape[1]
        if self.fit_intercept:
            Xbar = np.concatenate((np.ones((1, N)), X_arr), axis=0)
        else:
            Xbar = X_arr

        # Tính toán dự đoán: sgn(w^T * Xbar)
        raw_pred = np.sign(np.dot(self.w.T, Xbar)).flatten()
        raw_pred[raw_pred == 0] = 1.0

        # Chuyển đổi ngược lại nhãn ban đầu của dữ liệu
        return np.where(raw_pred == 1.0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        """Tính Accuracy"""
        return np.mean(self.predict(X) == np.asarray(y).flatten())


# =====================================================================
# BÀI TOÁN PHÁT HIỆN GIAO DỊCH GIAN LẬN (CREDIT CARD FRAUD DETECTION)
# =====================================================================

def create_fraud_dataset(n_samples=1200, random_state=42):
    """
    Tạo tập dữ liệu mô phỏng bài toán Phát hiện Giao dịch Gian lận (Fraud Detection).
    Gồm 5 thuộc tính đặc trưng:
      - X0: Số tiền giao dịch (Transaction Amount)
      - X1: Khoảng cách vị trí so với giao dịch trước (Distance from Previous Tx)
      - X2: Độ lệch thời gian / giờ giao dịch bất thường (Time Anomaly Score)
      - X3: Điểm tin cậy của thiết bị/IP (Device Trust Score)
      - X4: Tần suất giao dịch trong 24h qua (Tx Frequency 24h)
    Nhãn:
      - 0: Giao dịch hợp lệ (Legitimate)
      - 1: Giao dịch gian lận (Fraudulent)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],  # 85% hợp lệ, 15% gian lận (đặc thù thực tế)
        flip_y=0.01,
        random_state=random_state
    )
    feature_names = [
        "Transaction_Amount",
        "Distance_From_Prev_Tx",
        "Time_Anomaly_Score",
        "Device_Trust_Score",
        "Tx_Frequency_24h"
    ]
    return X, y, feature_names


def main():
    print("=" * 70)
    print("BÀI TẬP 3.30: PHÂN LỚP NHỊ PHÂN PHÁT HIỆN GIAO DỊCH GIAN LẬN BẰNG PERCEPTRON")
    print("=" * 70)

    # 1. Chuẩn bị tập dữ liệu
    X, y, feature_names = create_fraud_dataset(n_samples=1200, random_state=42)
    print(f"Tổng số mẫu: {X.shape[0]}")
    print(f"Số đặc trưng: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"Tỉ lệ nhãn thực tế: Hợp lệ (0) = {np.sum(y == 0)}, Gian lận (1) = {np.sum(y == 1)}")
    print("-" * 70)

    # 2. Chia tập Train (80%) và Test (20%) có phân tầng (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Số mẫu tập Train: {X_train.shape[0]}")
    print(f"Số mẫu tập Test : {X_test.shape[0]}")

    # 3. Chuẩn hóa đặc trưng (Feature Scaling bằng StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Tìm kiếm tốc độ học (Learning rate eta) tối ưu cho Perceptron
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_f1 = -1
    best_lr = None
    best_model = None

    print("\n--- QUÁ TRÌNH TỐI ƯU SIÊU THAM SỐ (LEARNING RATE) ---")
    for lr in learning_rates:
        model = Perceptron(eta=lr, max_iter=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        print(f"Learning rate = {lr:<6} | Accuracy = {acc:.4f} | F1-Score = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_lr = lr
            best_model = model

    # 5. Đánh giá chi tiết mô hình tối ưu trên tập Test
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON TỐI ƯU (Learning Rate eta = {best_lr})")
    print("=" * 70)

    y_test_pred = best_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    print(f"1. Accuracy  (Độ chính xác toàn thể) : {acc * 100:.2f}%")
    print(f"2. Precision (Độ chuẩn xác gian lận) : {prec * 100:.2f}%")
    print(f"3. Recall    (Độ nhạy / Bao quát)    : {rec * 100:.2f}%")
    print(f"4. F1-Score  (Trung bình điều hòa)   : {f1 * 100:.2f}%")
    print("-" * 70)

    # Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print(f"                     Dự đoán Hợp lệ (0)   Dự đoán Gian lận (1)")
    print(f"Thực tế Hợp lệ (0):          TN = {tn:<8}     FP = {fp}")
    print(f"Thực tế Gian lận (1):         FN = {fn:<8}     TP = {tp}")
    print("-" * 70)

    print("\nBÁO CÁO CHI TIẾT (CLASSIFICATION REPORT):")
    print(classification_report(y_test, y_test_pred, target_names=["Hợp lệ (0)", "Gian lận (1)"]))

    print("TRỌNG SỐ HỌC ĐƯỢC CỦA MÔ HÌNH:")
    for name, w in zip(feature_names, best_model.weights):
        print(f"  - {name:<25}: {w:+.4f}")
    print(f"  - Bias (w0)                 : {best_model.bias:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
