import sys
import io
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =====================================================================
# GRADIENT DESCENT CHO HÀM 1 BIẾN THEO MACHINE LEARNING CƠ BẢN (MLCB)
# https://machinelearningcoban.com/2017/01/12/gradientdescent/
# =====================================================================

# 1. Định nghĩa hàm đạo hàm grad(x) = f'(x)
def grad(x):
    """Đạo hàm bậc nhất f'(x) = 2x - 4"""
    return 2*x - 4


# 2. Định nghĩa hàm chi phí / hàm mục tiêu cost(x) = f(x)
def cost(x):
    """Hàm mục tiêu f(x) = x^2 - 4x + 5"""
    return x**2 - 4*x + 5


# 3. Thuật toán Gradient Descent 1 chiều (myGD1) chuẩn theo MLCB
def myGD1(eta, x0, num_steps=4):
    """
    Thuật toán Gradient Descent cho hàm 1 biến theo phong cách MLCB.
    
    Tham số:
    - eta: Tốc độ học (learning rate)
    - x0: Điểm khởi tạo ban đầu
    - num_steps: Số bước cập nhật (ở đây yêu cầu 4 bước)
    
    Trả về:
    - x: Danh sách các giá trị x qua từng bước cập nhật [x(0), x(1), ...]
    - it: Số bước lặp thực hiện
    """
    x = [x0]
    for it in range(num_steps):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)
        # Điều kiện dừng nếu gradient đủ nhỏ (tiêu chuẩn MLCB)
        if abs(grad(x_new)) < 1e-3:
            break
    return (x, it + 1)


def giai_chi_tiet_bai_3_26():
    eta = 0.2
    x0 = 5.0
    num_steps = 4

    print("=" * 65)
    print("BÀI TẬP 3.26: GRADIENT DESCENT (THEO CHUẨN MACHINE LEARNING CƠ BẢN)")
    print("=" * 65)
    print("1. Đạo hàm f'(x) = 2x - 4")
    print(f"2. Điểm khởi tạo: x0 = {x0}, Learning rate eta = {eta}\n")

    # Chạy thuật toán myGD1 theo MLCB
    (x_history, it) = myGD1(eta, x0, num_steps=100)

    print("3. BẢNG CHI TIẾT TỪNG BƯỚC CẬP NHẬT:")
    print(f"{'Bước (t)':<10}{'x(t)':<15}{'grad f\'(x(t))':<18}{'cost f(x(t))':<15}")
    print("-" * 65)
    for t in range(len(x_history)):
        xt = x_history[t]
        dft = grad(xt)
        ft = cost(xt)
        print(f"{t:<10}{xt:<15.6f}{dft:<18.6f}{ft:<15.6f}")
    print("=" * 65)

    print("\n4. NHẬN XÉT SỰ HỘI TỤ (THEO MLCB):")
    print("- Nghiệm tối ưu lý thuyết: f'(x*) = 2x - 4 = 0 => x* = 2, f(x*) = 1.")
    print(f"- Sau {len(x_history)-1} bước cập nhật:")
    print(f"  + x(t) giảm dần từ {x_history[0]:.4f} -> {x_history[1]:.4f} -> {x_history[2]:.4f} -> {x_history[3]:.4f} -> {x_history[4]:.4f} (tiến gần về x* = 2).")
    print(f"  + Giá trị hàm cost(x) giảm đơn điệu từ {cost(x_history[0]):.4f} -> {cost(x_history[4]):.4f} (tiến sát cực tiểu toàn cục f(x*) = 1).")
    print(f"  + Độ lớn gradient |grad(x)| giảm dần từ {abs(grad(x_history[0])):.4f} -> {abs(grad(x_history[4])):.4f} giúp bước nhảy thu nhỏ dần.")
    print("=> Thuật toán hội tụ đơn điệu, ổn định và chính xác.")


if __name__ == "__main__":
    giai_chi_tiet_bai_3_26()
