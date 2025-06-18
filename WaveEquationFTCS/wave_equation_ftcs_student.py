"""
学生模板：波动方程FTCS解
文件：wave_equation_ftcs_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。
    """
    # 高斯型初始速度分布，中心在d，宽度为sigma
    return C * np.exp(-((x - d * L) ** 2) / (2 * sigma ** 2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    """
    # 1. 获取参数
    a = parameters['a']
    L = parameters['L']
    d = parameters['d']
    C = parameters['C']
    sigma = parameters['sigma']
    dx = parameters['dx']
    dt = parameters['dt']
    total_time = parameters['total_time']

    # 2. 初始化空间和时间网格
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, total_time + dt, dt)
    Nx = x.size
    Nt = t.size

    # 3. 创建解数组
    u = np.zeros((Nx, Nt))

    # 4. 稳定性条件
    c = (a * dt / dx) ** 2
    if c >= 1:
        print("警告：稳定性条件不满足，c = {:.3f} >= 1，结果可能不稳定！".format(c))

    # 5. 初始条件
    # u(x, 0) = 0 已经在初始化时设置
    # u_t(x, 0) = psi(x)
    psi = u_t(x, C, d, sigma, L)
    # 6. 第一个时间步
    # u(x, 1) = u(x, 0) + dt * psi(x) + 0.5 * c * (u(x+dx,0) - 2u(x,0) + u(x-dx,0))
    u[1:-1, 1] = u[1:-1, 0] + dt * psi[1:-1] + 0.5 * c * (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0])
    # 边界条件
    u[0, 1] = 0
    u[-1, 1] = 0

    # 7. FTCS主算法
    for n in range(1, Nt - 1):
        u[1:-1, n + 1] = (2 * (1 - c) * u[1:-1, n] - u[1:-1, n - 1] +
                          c * (u[2:, n] + u[:-2, n]))
        # 边界条件
        u[0, n + 1] = 0
        u[-1, n + 1] = 0

    return u, x, t

if __name__ == "__main__":
    # Demonstration and testing
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(u_sol.min() * 1.1, u_sol.max() * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("1D Wave Equation (FTCS)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement")

    def update(frame):
        line.set_data(x_sol, u_sol[:, frame])
        return line,

    ani = FuncAnimation(fig, update, frames=t_sol.size, interval=1, blit=True)
    plt.show()
