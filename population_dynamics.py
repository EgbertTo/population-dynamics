import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from scipy.integrate import odeint
import seaborn as sns

# parameter name -> (initial value, (min, max))
PARAMS = {
    'r': (1, (0, 6)),
    'alpha': (1, (0, 2)),
    'beta': (1, (0, 2)),
    'K': (1_000, (100, 10_000)),
    'P0': (0, (0, 100_000)),
    't_max': (180, (1, 1_000)),
}


def dP_dt(
    t: int,
    P: float,
    r: float,
    alpha: float,
    beta: float,
    K: float,
) -> float:
    """
    d/dt P(t)
    The differential equation of the logistic population curve.
    """
    return r * ((K - P) / (K - (1 - beta) * P)) * P**alpha


def P_t(
    r: float,
    alpha: float,
    beta: float,
    K: float = 1_000_000,
    P0: float = 1_000,
    t_max: float = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    P(t)
    Numeric approximation of the solution of d/dt P(t).
    """
    t = np.arange(0, t_max)
    return t, odeint(dP_dt, P0, t, args=(r, alpha, beta, K))[:, 0]


# noinspection PyPep8Naming
def main():
    # Create the figure and the (initial) line
    fig, ax = plt.subplots(figsize=(16, 9))
    line, = plt.plot(*P_t(*(v for v, _ in PARAMS.values())), label='P(t)')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$P(t)$')
    ax.margins(x=0)

    # Clear some space for the sliders to the right and add margin
    plt.subplots_adjust(left=0.075, top=0.95, bottom=0.1, right=9/16)

    # Construct the sliders to control the parameters
    sliders = {}
    slider_position_x = 0.65
    for name, (value0, (a, b)) in PARAMS.items():
        sliders[name] = widgets.Slider(
            ax=plt.axes([slider_position_x, 0.1, 0.0225, 0.63]),
            label=name,
            valmin=a,
            valmax=b,
            valinit=value0,
            orientation='vertical',
        )
        slider_position_x += 0.05  # 5% horizontal padding between the sliders

    # Update the line on slider changes
    def update_line(_):
        param_vals = {_name: _slider.val for _name, _slider in sliders.items()}
        t, y = P_t(**param_vals)

        line.set_xdata(t)
        ax.set_xlim(0, t.max())

        line.set_ydata(y)
        ax.set_ylim(0, y.max() * 1.05)  # 5% margin

    for slider in sliders.values():
        slider.on_changed(update_line)

    # Display
    plt.show()


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    sns.set_theme()

    main()
