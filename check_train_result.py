import numpy      as np
import tensorflow as tf

from funcy   import *
from game    import *
from pathlib import *


def main():
    path = Path('./model')

    for model_path in path.glob('cost-1024x4-1000x1000*.h5'):
        model = tf.keras.models.load_model(model_path)

        seed(0)

        for y_true in range(1, 32):
            for y_pred in model.predict(np.array(tuple(repeatedly(lambda: get_x(get_random_state(y_true)[0]), 1000))), 1000).flatten():
                print(f'{model_path.stem}\t{y_true}\t{y_pred}')

        tf.keras.backend.clear_session()

    seed()


if __name__ == '__main__':
    main()
