import numpy as np


class Simplex(object):
    # 单纯形表
    def __init__(self, A, B, N, b, c, C_B, C_N, B_x, N_x):
        self.A = A  # 系数矩阵
        self.c = c
        self.C_B = C_B
        self.b = b
        self.N = N
        self.B = B
        self.C_N = C_N
        self.B_x = B_x
        self.N_x = N_x


def simplex_run(a):
    B_inverse = np.linalg.inv(a.B)
    coefs = (a.C_N.T - np.matmul(np.matmul(a.C_B.T, B_inverse), a.N)).flatten()
    # print(coefs)
    if coefs.max() <= 0:  #最优解条件，检验数都非正
        is_optim = True
        solution = np.matmul(B_inverse, b)
        print("找到最优解。")
        print('基变量为 {}'.format(a.B_x))
        print('非基变量为 {}'.format(a.N_x))
        print("最优解为 {}".format(solution.flatten()))
        print("最大值为 {} \n".format(np.matmul(a.C_B.T, solution).flatten()[0]))
        return is_optim, a
    else:
        in_base = np.argmax(coefs)
        N_i = a.N[:, in_base].reshape(-1, 1)  # 入基变量 in_base， 出基变量 out_base
        y = np.matmul(B_inverse, N_i)
        x_B = np.matmul(B_inverse, a.b)
        out_base = find_out_base(y, x_B)
        temp = a.N_x[in_base]
        a.N_x[in_base] = a.B_x[out_base]
        a.B_x[out_base] = temp
        is_optim = False
        print("没有达到最优解")
        print("入基变量 x{}".format(temp))
        print("出基变量 x{}".format(a.N_x[in_base]))
        print("基变量为 {}".format(sorted(a.B_x)))
        print("非基变量为 {} \n".format(sorted(a.N_x)))
        return is_optim, a


def find_out_base(y, b):
    index = []
    min_value = []
    for i, value in enumerate(y):
        if value <= 0:
            continue
        else:
            index.append(i)
            min_value.append(b[i] / float(value))
    return index[np.argmin(min_value)]


if __name__ == "__main__":
    c = np.array([6, 14, 13, 0, 0]).reshape(-1, 1)
    C_B = np.array([0, 0]).reshape(-1, 1)
    C_N = np.array([6, 14, 13]).reshape(-1, 1)
    A = np.array([[1,4,2,1,0], [1,2,4,0,1]])
    N = np.array([[1,4,2], [1,2,4]])
    B = np.array([[1,0], [0,1]])
    b = np.array([48, 60])
    B_x = np.array([3, 4])
    N_x = np.array([0, 1, 2])
    a = Simplex(A, B, N, b, c, C_B, C_N, B_x, N_x)
    steps = 0
    while True:
        steps += 1
        print('steps is {}'.format(steps))
        is_optim, a = simplex_run(a)
        if is_optim:
            break
        else:
            a.B = a.A[:, a.B_x]
            a.N = a.A[:, a.N_x]
            a.C_B = a.c[a.B_x, :]
            a.C_N = a.c[a.N_x, :]

