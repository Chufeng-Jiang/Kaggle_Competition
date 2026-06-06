import re
import numpy as np


def solve_roman(prompt: str) -> str:
    m = re.search(r"write the number (\d+)", prompt.lower())
    if not m:
        return ""

    n = int(m.group(1))

    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]

    ans = ""
    for val, sym in table:
        while n >= val:
            ans += sym
            n -= val

    return ans


def solve_gravity(prompt: str) -> str:
    examples = re.findall(
        r"For t = ([\d.]+)s, distance = ([\d.]+) m",
        prompt
    )

    target = re.search(
        r"for t = ([\d.]+)s",
        prompt.split("Now")[-1]
    )

    if not examples or not target:
        return ""

    gs = []
    for t, d in examples:
        t = float(t)
        d = float(d)
        g = 2 * d / (t ** 2)
        gs.append(g)

    g_hat = np.mean(gs)
    t_new = float(target.group(1))
    pred = 0.5 * g_hat * t_new ** 2

    return f"{pred:.2f}"


def solve_unit(prompt: str) -> str:
    examples = re.findall(
        r"([\d.]+) m becomes ([\d.]+)",
        prompt
    )

    target = re.search(
        r"following measurement: ([\d.]+) m",
        prompt
    )

    if not examples or not target:
        return ""

    ratios = []
    for x, y in examples:
        ratios.append(float(y) / float(x))

    ratio = np.mean(ratios)
    x_new = float(target.group(1))
    pred = x_new * ratio

    if abs(pred - round(pred)) < 0.01:
        return str(int(round(pred)))

    return f"{pred:.2f}"