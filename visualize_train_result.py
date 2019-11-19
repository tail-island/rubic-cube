import matplotlib.pyplot    as plot
import matplotlib.animation as animation
import numpy                as np

from funcy   import *
from pathlib import *


def main():
    data = {}

    while True:
        try:
            model_name, y_true_string, y_pred_string = input().split('\t')

            y_true = float(y_true_string)
            y_pred = float(y_pred_string)

            if model_name not in data:
                data[model_name] = {}

            if y_true not in data[model_name]:
                data[model_name][y_true] = []

            data[model_name][y_true].append(y_pred)

        except EOFError:
            break;

    figure, ax = plot.subplots()
    artists    = []

    for model_name in sorted(data.keys()):
        x = []
        y = []

        for y_true, y_preds in data[model_name].items():
            x.extend((y_true,) * len(y_preds))
            y.extend(y_preds)

        plot.xlim(0, 32)
        plot.ylim(0, 32)
        plot.grid(True)

        artists.append((ax.scatter(x, y, c='#0000ff', alpha=0.01),))

    artist_animation = animation.ArtistAnimation(figure, artists, interval=1000)
    artist_animation.save('./temp/train-result.gif', writer='imagemagick')

    plot.show()


if __name__ == '__main__':
    main()
