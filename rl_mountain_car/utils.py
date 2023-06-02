import matplotlib.pyplot as plt
import numpy as np
from IPython import display


def render_env(env, title):
    plt.imshow(env.render())
    plt.title(title)
    display.clear_output(wait=True)
    display.display(plt.gcf())


def energy(state):
    """報酬計算に使用する(https://qiita.com/payanotty/items/07fb38a44cc3bd13e4dd 参考)。"""
    x = state[0]  # 位置(横方向)
    g = 0.0025  # 重力定数
    v = state[1]  # 速度

    c = 1 / (g * np.sin(3 * 0.5) + 0.5 * 0.07 * 0.07)  # 正規化定数

    return c * (g * np.sin(3 * x) + 0.5 * v * v)
