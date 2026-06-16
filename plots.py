import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PlotCurve:
    data: List[Tuple[list, list]]
    x_label: str
    y_label: str
    line_label: list[str]
    title: str
    save_address: str

def plot_curves(curve: PlotCurve):
    plt.figure()
    for (x_vals, y_vals), label in zip(curve.data, curve.line_label):
        plt.plot(x_vals, y_vals, label=label)

    plt.xlabel(curve.x_label)
    plt.ylabel(curve.y_label)
    plt.title(curve.title)
    plt.legend()
    os.makedirs(os.path.dirname(curve.save_address) or ".", exist_ok=True)
    plt.savefig(curve.save_address)
    plt.show()