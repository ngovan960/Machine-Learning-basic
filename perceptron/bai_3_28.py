import sys
import io
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def h(w, x):
    """Hàm dự đoán theo Machine Learning Cơ Bản: sgn(w^T * x)"""
    return np.sign(np.dot(w.T, x))


def main():
    print("=" * 60)
    print("BÀI TẬP 3.28: PERCEPTRON (KIỂM TRA VÀ CẬP NHẬT TRỌNG SỐ)")
    print("=" * 60)

    # Cho trước: w = [-2, 1, 0]^T, x = [2, 3, 1]^T, nhãn y = 1
    w = np.array([-2, 1, 0], dtype=float).reshape(-1, 1)
    x = np.array([2, 3, 1], dtype=float).reshape(-1, 1)
    y_true = 1
    eta = 1.0  # Tốc độ học (mặc định = 1)

    print(f"Vector trọng số ban đầu w =\n{w}")
    print(f"\nVector mẫu dữ liệu x (đã có bias) =\n{x}")
    print(f"\nNhãn thực tế: y = {y_true}")
    print("-" * 60)

    # 1. Kiểm tra mẫu có bị phân lớp sai hay không
    score = np.dot(w.T, x)[0, 0]
    y_pred = int(h(w, x)[0, 0])
    if y_pred == 0:
        y_pred = 1

    print("1. Kiểm tra mẫu có bị phân lớp sai hay không:")
    print(f"   - w^T x = (-2*2) + (1*3) + (0*1) = -4 + 3 + 0 = {score:.1f}")
    print(f"   - Nhãn dự đoán: y_pred = sgn(w^T x) = sgn({score:.1f}) = {y_pred}")
    print(f"   - Kiểm tra điều kiện: y * (w^T x) = ({y_true}) * ({score:.1f}) = {y_true * score:.1f} <= 0")
    print(f"   => KẾT LUẬN: Mẫu BỊ PHÂN LỚP SAI (vì y_pred = {y_pred} != y = {y_true}).")

    # 2. Nếu sai, thực hiện một bước cập nhật Perceptron
    # Theo Machine Learning Cơ Bản: w_new = w + eta * y * x
    w_new = w + eta * y_true * x
    print(f"\n2. Thực hiện một bước cập nhật Perceptron (với eta = {eta}):")
    print(f"   Công thức cập nhật (MLCB): w_new = w + eta * y * x")
    print(f"   w_new = [-2, 1, 0]^T + 1 * [2, 3, 1]^T")
    print(f"         = [-2+2, 1+3, 0+1]^T")
    print(f"   => w_new =\n{w_new}")

    # 3. Tính lại giá trị w^T x sau cập nhật
    new_score = np.dot(w_new.T, x)[0, 0]
    new_y_pred = int(h(w_new, x)[0, 0])
    if new_y_pred == 0:
        new_y_pred = 1

    print(f"\n3. Tính lại giá trị w^T x sau cập nhật:")
    print(f"   w_new^T x = (0*2) + (4*3) + (1*1) = 0 + 12 + 1 = {new_score:.1f}")
    print(f"   Nhãn dự đoán mới: y_pred_new = sgn({new_score:.1f}) = {new_y_pred}")
    print(f"   Kiểm tra lại: y * (w_new^T x) = ({y_true}) * ({new_score:.1f}) = {y_true * new_score:.1f} > 0")
    print(f"   => KẾT LUẬN: Điểm dữ liệu ĐÃ ĐƯỢC PHÂN LỚP ĐÚNG sau cập nhật.")
    print("=" * 60)


if __name__ == "__main__":
    main()
