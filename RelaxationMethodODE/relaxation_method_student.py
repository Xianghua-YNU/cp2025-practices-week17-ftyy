"""
学生模板：松弛迭代法解常微分方程
文件：relaxation_method_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    实现松弛迭代法求解常微分方程 d²x/dt² = -g
    边界条件：x(0) = x(10) = 0（抛体运动问题）

    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差

    返回:
        tuple: (时间数组, 解数组)

    物理背景: 质量为1kg的球从高度x=0抛出，10秒后回到x=0
    数值方法: 松弛迭代法，迭代公式 x(t) = 0.5*h²*(-g) + 0.5*[x(t+h)+x(t-h)]
    """
    # 初始化时间数组
    t = np.arange(0, 10 + h, h)
    n = t.size

    # 初始化解数组，边界条件已满足：x[0] = x[-1] = 0
    x = np.zeros(n)

    delta = 1.0
    iter_count = 0

    while delta > tol and iter_count < max_iter:
        x_new = x.copy()
        # 对内部点应用松弛迭代公式
        x_new[1:-1] = 0.5 * (h * h * (-g) + x[2:] + x[:-2])
        delta = np.max(np.abs(x_new - x))
        x = x_new
        iter_count += 1

    return t, x

if __name__ == "__main__":
    # 测试参数
    h = 10 / 100  # 时间步长
    g = 9.8       # 重力加速度

    # 调用求解函数
    t, x = solve_ode(h, g)

    # 绘制结果（中文支持）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

    plt.plot(t, x)
    plt.xlabel('时间 (秒)')
    plt.ylabel('高度 (米)')
    plt.title('抛体运动轨迹（松弛迭代法）')
    plt.grid()
    plt.show()
