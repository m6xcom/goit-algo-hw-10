import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi


def f(x):
    return x ** 2


a = 0  # Нижня межа
b = 2  # Верхня межа
ANALYTIC = (b ** 3 - a ** 3) / 3  # Первісна x^3 / 3, тобто 8/3


def monte_carlo_hit_or_miss(func, a: float, b: float, n: int, rng: np.random.Generator):
    """Метод «влучань»: частка точок під кривою, помножена на площу прямокутника.

    Повертає оцінку інтеграла та самі точки (для графіка).
    """
    x_grid = np.linspace(a, b, 1000)
    y_max = func(x_grid).max()  # висота прямокутника [a, b] x [0, y_max]

    x = rng.uniform(a, b, n)
    y = rng.uniform(0, y_max, n)
    under = y <= func(x)

    area = (b - a) * y_max
    return area * under.mean(), x, y, under


def monte_carlo_mean(func, a: float, b: float, n: int, rng: np.random.Generator) -> float:
    """Метод середнього значення: (b - a) * середнє f у випадкових точках."""
    x = rng.uniform(a, b, n)
    return (b - a) * func(x).mean()


def plot(x, y, under) -> None:
    """Графік з умови, доповнений випадковими точками методу Монте-Карло."""
    xs = np.linspace(-0.5, 2.5, 400)
    ys = f(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, "r", linewidth=2)

    ix = np.linspace(a, b)
    ax.fill_between(ix, f(ix), color="gray", alpha=0.3)

    ax.scatter(x[under], y[under], s=2, color="tab:blue", label="під кривою")
    ax.scatter(x[~under], y[~under], s=2, color="tab:orange", label="над кривою")

    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([0, max(ys) + 0.1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(x=a, color="gray", linestyle="--")
    ax.axvline(x=b, color="gray", linestyle="--")
    ax.set_title(f"Графік інтегрування f(x) = x^2 від {a} до {b}")
    ax.legend(loc="upper left")
    ax.grid()
    fig.savefig("monte_carlo.png", dpi=120, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="не відкривати вікно з графіком")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    quad_result, quad_error = spi.quad(f, a, b)
    print(f"Аналітично:  {ANALYTIC:.10f}")
    print(f"quad:        {quad_result:.10f} (оцінка похибки {quad_error:.1e})")

    print("\nОдин запуск для різної кількості точок N:")
    header = f"{'N':>10} | {'Влучання':>10} | {'Похибка':>8} | {'Середнє':>10} | {'Похибка':>8}"
    print(header)
    print("-" * len(header))
    for n in (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        hm, *_ = monte_carlo_hit_or_miss(f, a, b, n, rng)
        mean = monte_carlo_mean(f, a, b, n, rng)
        print(
            f"{n:>10} | {hm:>10.6f} | {abs(hm - quad_result) / quad_result:>7.3%} | "
            f"{mean:>10.6f} | {abs(mean - quad_result) / quad_result:>7.3%}"
        )

    runs = 100
    print(f"\nСередня відносна похибка за {runs} запусків:")
    header = f"{'N':>10} | {'Влучання':>9} | {'Середнє':>9}"
    print(header)
    print("-" * len(header))
    for n in (100, 1_000, 10_000, 100_000):
        hm_err = np.mean([abs(monte_carlo_hit_or_miss(f, a, b, n, rng)[0] - quad_result) for _ in range(runs)])
        mean_err = np.mean([abs(monte_carlo_mean(f, a, b, n, rng) - quad_result) for _ in range(runs)])
        print(f"{n:>10} | {hm_err / quad_result:>8.3%} | {mean_err / quad_result:>8.3%}")

    _, x, y, under = monte_carlo_hit_or_miss(f, a, b, 2_000, rng)
    plot(x, y, under)
    print("\nГрафік збережено у monte_carlo.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
