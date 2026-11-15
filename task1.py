import timeit

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """Жадібний алгоритм: щоразу бере найбільший номінал, що вміщується в залишок.

    Складність: O(k log k) на сортування + O(k) на прохід, де k — кількість номіналів.
    Від величини суми час не залежить, бо кількість монет рахується діленням.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    result = {}
    remainder = amount
    for coin in sorted(coins, reverse=True):
        count, remainder = divmod(remainder, coin)
        if count:
            result[coin] = count

    if remainder:
        raise ValueError(f"Суму {amount} неможливо видати номіналами {coins}")
    return result


def find_min_coins(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """Динамічне програмування: мінімальна кількість монет для суми amount.

    min_coins[s] — мінімальна кількість монет для суми s,
    last_coin[s] — монета, якою закінчується оптимальний набір для суми s.
    Складність: час O(n * k), пам'ять O(n), де n — сума, k — кількість номіналів.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    inf = float("inf")
    min_coins = [0] + [inf] * amount
    last_coin = [0] * (amount + 1)

    for s in range(1, amount + 1):
        for coin in coins:
            if coin <= s and min_coins[s - coin] + 1 < min_coins[s]:
                min_coins[s] = min_coins[s - coin] + 1
                last_coin[s] = coin

    if min_coins[amount] == inf:
        raise ValueError(f"Суму {amount} неможливо видати номіналами {coins}")

    # Відновлення набору монет за масивом last_coin
    result = {}
    s = amount
    while s > 0:
        coin = last_coin[s]
        result[coin] = result.get(coin, 0) + 1
        s -= coin
    return dict(sorted(result.items()))


def measure(func, amount: int, coins: list[int] = COINS) -> float:
    """Середній час одного виклику функції в секундах."""
    timer = timeit.Timer(lambda: func(amount, coins))
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=3, number=number)) / number


def main() -> None:
    amount = 113
    print(f"Сума: {amount}, номінали: {COINS}")
    print("Жадібний алгоритм:       ", find_coins_greedy(amount))
    print("Динамічне програмування: ", find_min_coins(amount))

    print("\nПорівняння часу виконання:")
    header = f"{'Сума':>10} | {'Жадібний, с':>13} | {'ДП, с':>13} | {'ДП / жадібний':>14} | Монет (Ж / ДП)"
    print(header)
    print("-" * len(header))
    for amount in (113, 1_000, 10_000, 100_000, 1_000_000):
        t_greedy = measure(find_coins_greedy, amount)
        t_dp = measure(find_min_coins, amount)
        n_greedy = sum(find_coins_greedy(amount).values())
        n_dp = sum(find_min_coins(amount).values())
        print(
            f"{amount:>10} | {t_greedy:>13.2e} | {t_dp:>13.2e} | "
            f"{t_dp / t_greedy:>13.0f}x | {n_greedy} / {n_dp}"
        )

    # Набір, для якого жадібний алгоритм не дає оптимуму
    coins, amount = [4, 3, 1], 6
    print(f"\nНестандартні номінали {coins}, сума {amount}:")
    print("Жадібний алгоритм:       ", find_coins_greedy(amount, coins))
    print("Динамічне програмування: ", find_min_coins(amount, coins))


if __name__ == "__main__":
    main()
