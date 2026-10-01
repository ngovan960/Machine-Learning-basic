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
    print("BÀI TẬP 3.27: PHƯƠNG PHÁP PERCEPTRON (DỰ ĐOÁN NHÃN)")
    print("=" * 60)
    
    # Cho trước: w = [1, 2, -10]^T, x = [3, 4, 1]^T, nhãn thực tế y = -1
    w = np.array([1, 2, -10], dtype=float).reshape(-1, 1)
    x = np.array([3, 4, 1], dtype=float).reshape(-1, 1)
    y_true = -1

    print(f"Vector trọng số w =\n{w}")
    print(f"\nVector mẫu dữ liệu x (đã có bias) =\n{x}")
    print(f"\nNhãn thực tế: y = {y_true}")
    print("-" * 60)

    # 1. Tính w^T x
    score = np.dot(w.T, x)[0, 0]
    print(f"1. Tính w^T x:")
    print(f"   w^T x = (1*3) + (2*4) + (-10*1) = 3 + 8 - 10 = {score:.1f}")

    # 2. Xác định nhãn dự đoán của điểm dữ liệu
    # sgn(w^T x) = 1 nếu >= 0, ngược lại -1
    y_pred = int(h(w, x)[0, 0])
    if y_pred == 0:
        y_pred = 1
    print(f"\n2. Xác định nhãn dự đoán của điểm dữ liệu:")
    print(f"   y_pred = sgn(w^T x) = sgn({score:.1f}) = {y_pred:+d}")

    # 3. Kiểm tra xem điểm dữ liệu có bị phân lớp sai không
    print(f"\n3. Kiểm tra với nhãn thực tế y = {y_true}:")
    print(f"   - Nhãn dự đoán: y_pred = {y_pred}")
    print(f"   - Nhãn thực tế : y = {y_true}")
    print(f"   - Kiểm tra điều kiện: y * (w^T x) = ({y_true}) * ({score:.1f}) = {y_true * score:.1f} <= 0")
    print(f"   => KẾT LUẬN: Điểm dữ liệu BỊ PHÂN LỚP SAI (Misclassified).")
    print("=" * 60)


if __name__ == "__main__":
    main()
