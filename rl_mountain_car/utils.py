import matplotlib.pyplot as plt
from IPython import display


def render_env(env, title):
    plt.imshow(env.render())
    plt.title(title)
    display.clear_output(wait=True)
    display.display(plt.gcf())


def calc_reward(
    current_state: tuple[float, float],
    next_state: tuple[float, float],
    num_trial: int,
    done: bool,
) -> float:
    """MountainCarのデフォルト報酬が厳しすぎるので新たに定義した報酬関数
    Args:
        current_state: 現在の状態(位置, 速度)
        next_state: 次の状態(位置, 速度)
        num_trial: 試行回数
        done: ゴールにたどり着いたか
    Returns:
        報酬
    """
    reward = -1
    # 200回以内にdoneにたどり着いた場合は報酬に250を加算
    if done and (num_trial < 200):
        reward += 250
    else:
        # それ以外は、移動距離に比例した報酬を加算
        reward = 5 * abs(next_state[0] - current_state[0]) + 3 * abs(current_state[1])
    return reward
