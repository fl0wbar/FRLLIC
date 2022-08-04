import numpy as np


_WIDTH = 0.8


# TODO: rename to something with bar plot


def plot_histogram(datas, plt):
    for i, data in enumerate(datas):
        rel_i = i - len(datas) / 2
        w = _WIDTH/len(datas)
        _plot_histogram(data, plt, w, rel_i * w)
    plt.legend()


def _plot_histogram(data, plt, width, offset):
    name, values = data
    plt.bar(np.arange(len(values)) + offset, values,
            width=width,
            label=name, align='edge')


def _test():
    import matplotlib.pyplot as plt

    f = plt.figure()
    datas = [('gt', [1000, 10, 33, 500, 600, 700]),
             ('outs', [900, 20, 0, 0, 100, 1000]),
             ('ups', 0.5 * np.array([900, 20, 0, 0, 100, 1000])),
             ]
    plot_histogram(datas, plt)
    f.show()


if __name__ == '__main__':
    _test()
