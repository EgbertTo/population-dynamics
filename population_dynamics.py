import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from scipy.integrate import odeint
import seaborn as sns


def dP_dt(
    t: int,
    P: float,
    r: float,
    alpha: float,
    beta: float,
    K: float,
) -> float:
    return r * ((K - P) / (K - (1 - beta) * P)) * P**alpha


def P_t(
    r: float,
    alpha: float,
    beta: float,
    K: float = 1_000_000,
    P0: float = 1_000,
    t_max: float = 180,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, t_max)
    return t, odeint(dP_dt, P0, t, args=(r, alpha, beta, K))[:, 0]


# noinspection PyPep8Naming
def main():
    r0, alpha0, beta0 = 1.0, 1.0, 1.0
    K0 = 1_000
    t_min, t_max = 0, 180

    # Create the figure and the (initial) line
    fig, ax = plt.subplots(figsize=(16, 9))
    line, = plt.plot(*P_t(r0, alpha0, beta0, K0, t_min, t_max), label='P(t)')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$P(t)$')
    ax.margins(x=0)

    # Clear some space for the sliders to the right and add margin
    plt.subplots_adjust(left=0.075, top=0.95, bottom=0.1, right=9/16)

    # Construct the sliders to control the parameters
    slider_r = widgets.Slider(
        ax=plt.axes([0.65, 0.1, 0.0225, 0.63]),
        label='r',
        valmin=0.01,
        valmax=6,
        valinit=r0,
        orientation='vertical',
    )
    slider_alpha = widgets.Slider(
        ax=plt.axes([0.7, 0.1, 0.0225, 0.63]),
        label='alpha',
        valmin=0,
        valmax=2,
        valinit=alpha0,
        orientation='vertical',
    )
    slider_beta = widgets.Slider(
        ax=plt.axes([0.75, 0.1, 0.0225, 0.63]),
        label='beta',
        valmin=0,
        valmax=2,
        valinit=beta0,
        orientation='vertical',
    )
    slider_K = widgets.Slider(
        ax=plt.axes([0.8, 0.1, 0.0225, 0.63]),
        label='K',
        valmin=100,
        valmax=10_000,
        valinit=1_000,
        orientation='vertical',
    )
    slider_P0 = widgets.Slider(
        ax=plt.axes([0.85, 0.1, 0.0225, 0.63]),
        label='P0',
        valmin=0,
        valmax=1_000_000,
        valinit=1_000,
        orientation='vertical',
    )
    slider_t = widgets.Slider(
        ax=plt.axes([0.91, 0.1, 0.0225, 0.63]),
        label='t',
        valmin=0,
        valmax=10_000,
        valinit=1_000,
        orientation='vertical',
    )

    # Update the line on slider changes
    def update_line(_):
        t, y = P_t(
            slider_r.val,
            slider_alpha.val,
            slider_beta.val,
            slider_K.val,
            slider_P0.val,
            slider_t.val,
        )

        line.set_xdata(t)
        ax.set_xlim(0, t.max())

        line.set_ydata(y)
        ax.set_ylim(0, y.max())

    slider_r.on_changed(update_line)
    slider_alpha.on_changed(update_line)
    slider_beta.on_changed(update_line)
    slider_K.on_changed(update_line)
    slider_P0.on_changed(update_line)
    slider_t.on_changed(update_line)

    # Display
    plt.show()


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    sns.set_theme()

    main()
