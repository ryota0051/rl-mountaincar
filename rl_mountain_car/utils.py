import matplotlib.pyplot as plt
from IPython import display


def render_env(env, title):
    plt.imshow(env.render())
    plt.title(title)
    display.clear_output(wait=True)
    display.display(plt.gcf())
