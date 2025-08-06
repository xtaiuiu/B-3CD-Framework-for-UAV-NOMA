from scipy.optimize import root_scalar

# 定义一个示例函数
def f(x):
    return x**4 + x**3 - 2

# 使用 root_scalar 求根
result = root_scalar(f, bracket=[0, 2], method='brentq', options={'disp': True})

# 打印结果
print("Root:", result.root)
print("Converged:", result.converged)
print("Iterations:", result.iterations)
print("Function evaluations:", result.function_calls)