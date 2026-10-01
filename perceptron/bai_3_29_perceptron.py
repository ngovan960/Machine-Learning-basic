import sys
import io
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =====================================================================
# PHẦN 1: CÁC HÀM CƠ BẢN THEO CHUẨN MACHINE LEARNING CƠ BẢN (MLCB)
# https://machinelearningcoban.com/2017/01/21/perceptron/
# =====================================================================

def h(w, x):
    """
    Hàm dự đoán nhãn cho điểm dữ liệu x với vector trọng số w.
    sgn(w^T * x): trả về 1 nếu w^T * x >= 0, ngược lại trả về -1.
    """
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    """
    Kiểm tra xem tất cả các điểm trong tập dữ liệu đã được phân loại đúng chưa.
    Trả về True nếu h(w, X) == y, ngược lại trả về False.
    """
    return np.array_equal(h(w, X), y)


def perceptron_mlcb(X, y, w_init, max_iter=1000):
    """
    Thuật toán Perceptron Learning Algorithm (PLA) theo Machine Learning Cơ Bản.
    
    Tham số:
    - X: Ma trận dữ liệu đã mở rộng bias (d+1, N)
    - y: Vector hàng chứa nhãn {-1, 1} (1, N)
    - w_init: Vector trọng số khởi tạo (d+1, 1)
    
    Trả về:
    - w: Danh sách các vector trọng số qua từng lần cập nhật
    - mis_points: Danh sách các chỉ số điểm bị phân lớp sai được cập nhật
    """
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    
    for epoch in range(max_iter):
        # Xáo trộn thứ tự các điểm dữ liệu ngẫu nhiên
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(-1, 1)
            yi = y[0, mix_id[i]]
            # Nếu dự đoán sai: sgn(w^T * xi) != yi
            if h(w[-1], xi)[0, 0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi   # Công thức cập nhật: w_new = w + y_i * x_i
                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
            
    return w, mis_points


# =====================================================================
# PHẦN 2: LỚP (CLASS) PERCEPTRON CÓ FIT & PREDICT (CHUẨN BÀI 3.29 & MLCB)
# =====================================================================

class Perceptron:
    """
    Lớp Perceptron kế thừa tư tưởng và công thức từ Machine Learning Cơ Bản:
    - Nhãn y thuộc {-1, +1}
    - Hàm kích hoạt: sgn(w^T * x_bar)
    - Công thức cập nhật: w = w + eta * y_i * x_bar_i khi dự đoán sai
    """
    def __init__(self, eta=1.0, max_iter=1000, random_state=None, learning_rate=None, fit_intercept=True):
        self.eta = learning_rate if learning_rate is not None else eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.w = None              # Vector trọng số mở rộng (d+1, 1): [w0, w1, ..., wd]^T
        self.w_history = []        # Lưu lại lịch sử cập nhật trọng số
        self.mis_points = []       # Lưu các điểm phân lớp sai

    @property
    def weights(self):
        """Trả về vector trọng số đặc trưng (w1, ..., wd)"""
        return self.w[1:].flatten() if self.w is not None else None

    @property
    def bias(self):
        """Trả về hệ số bias w0"""
        return self.w[0, 0] if self.w is not None else 0.0

    def _add_bias(self, X):
        """
        Thêm hàng bias (toàn số 1) vào ma trận dữ liệu theo chuẩn MLCB:
        Xbar có dạng (d+1, N) với hàng đầu tiên là các số 1.
        """
        # Nếu X có dạng (N, d) chuẩn Scikit-learn, chuyển vị về (d, N)
        if X.shape[0] != 1 and X.shape[1] != 1 and X.shape[0] > X.shape[1]:
            X = X.T
        N = X.shape[1]
        Xbar = np.concatenate((np.ones((1, N)), X), axis=0)
        return Xbar

    def fit(self, X, y):
        """
        Huấn luyện mô hình Perceptron.
        
        Tham số:
        - X: Ma trận dữ liệu (N, d) hoặc (d, N)
        - y: Vector nhãn (N,) hoặc (1, N). Hỗ trợ nhãn {0, 1} hoặc {-1, 1}.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Đảm bảo y ở dạng vector hàng (1, N) với giá trị {-1, 1}
        y = np.asarray(y).flatten()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron chỉ áp dụng cho bài toán 2 lớp (Binary Classification).")

        # Chuyển đổi nhãn về {-1, +1} theo MLCB
        y_mlcb = np.where(y == self.classes_[1], 1.0, -1.0).reshape(1, -1)

        # Xây dựng Xbar (d+1, N)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] >= X_arr.shape[1] and X_arr.shape[0] == len(y):
            X_arr = X_arr.T  # Đưa về dạng (d, N)

        Xbar = np.concatenate((np.ones((1, X_arr.shape[1])), X_arr), axis=0)
        d_plus_1, N = Xbar.shape

        # Khởi tạo w = 0 hoặc w ngẫu nhiên
        w_curr = np.zeros((d_plus_1, 1))
        self.w_history = [w_curr]
        self.mis_points = []

        for epoch in range(self.max_iter):
            mix_id = np.random.permutation(N)
            has_error = False

            for i in range(N):
                idx = mix_id[i]
                xi = Xbar[:, idx].reshape(-1, 1)
                yi = y_mlcb[0, idx]

                # Dự đoán: sgn(w^T * xi)
                y_pred = np.sign(np.dot(w_curr.T, xi))[0, 0]
                if y_pred == 0:
                    y_pred = 1.0

                # Nếu phân lớp sai: cập nhật w
                if y_pred != yi:
                    self.mis_points.append(idx)
                    w_curr = w_curr + self.eta * yi * xi
                    self.w_history.append(w_curr)
                    has_error = True

            # Kiểm tra hội tụ trên toàn bộ tập dữ liệu
            all_preds = np.sign(np.dot(w_curr.T, Xbar))
            all_preds[all_preds == 0] = 1.0
            if np.array_equal(all_preds, y_mlcb):
                # print(f"Hội tụ tại epoch {epoch + 1}")
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
            
        # Nếu X là (N, d), chuyển về (d, N)
        if X_arr.shape[1] == (self.w.shape[0] - 1):
            X_arr = X_arr.T

        N = X_arr.shape[1]
        Xbar = np.concatenate((np.ones((1, N)), X_arr), axis=0)

        # Tính sgn(w^T * Xbar)
        raw_pred = np.sign(np.dot(self.w.T, Xbar)).flatten()
        raw_pred[raw_pred == 0] = 1.0

        # Ánh xạ về lại nhãn gốc của dữ liệu
        y_pred = np.where(raw_pred == 1.0, self.classes_[1], self.classes_[0])
        return y_pred

    def score(self, X, y):
        """Tính Accuracy"""
        y_pred = self.predict(X)
        y_true = np.asarray(y).flatten()
        return np.mean(y_pred == y_true)


# =====================================================================
# DEMO KIỂM THỬ THEO VÍ DỤ CỦA MACHINE LEARNING CƠ BẢN
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("DEMO PERCEPTRON THEO MACHINE LEARNING CƠ BẢN (mlcb)")
    print("=" * 65)

    # 1. Tạo dữ liệu 2 cụm Gaussian 2D (như trong bài viết của MLCB)
    np.random.seed(2)
    means = [[2, 2], [4, 2]]
    cov = [[.3, .2], [.2, .3]]
    N = 10
    X0 = np.random.multivariate_normal(means[0], cov, N).T  # Nhãn +1
    X1 = np.random.multivariate_normal(means[1], cov, N).T  # Nhãn -1

    X = np.concatenate((X0, X1), axis=1) # (2, 20)
    y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1) # (1, 20)
    Xbar = np.concatenate((np.ones((1, 2*N)), X), axis=0) # (3, 20)

    # 2. Huấn luyện bằng hàm perceptron_mlcb truyền thống
    w_init = np.random.randn(3, 1)
    w_list, mis_points = perceptron_mlcb(Xbar, y, w_init)
    print(f"1. Hàm perceptron_mlcb():")
    print(f"   - Số lần cập nhật trọng số: {len(w_list) - 1}")
    print(f"   - Vector trọng số tối ưu w:\n{w_list[-1].flatten()}")

    # 3. Huấn luyện bằng lớp Perceptron (OOP)
    clf = Perceptron(eta=1.0, max_iter=100, random_state=42)
    clf.fit(X.T, y.flatten())
    print(f"\n2. Lớp Perceptron (OOP) fit & predict:")
    print(f"   - Trọng số w tìm được: {clf.w.flatten()}")
    print(f"   - Độ chính xác Accuracy: {clf.score(X.T, y.flatten()) * 100:.2f}%")
    print("=" * 65)
