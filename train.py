import tensorflow as tf

from funcy   import *
from game    import *
from pathlib import *


def computational_graph():
    def add():
        return tf.keras.layers.Add()

    def batch_normalization():
        return tf.keras.layers.BatchNormalization()

    def conv(filter_size, kernel_size=3):
        return tf.keras.layers.Conv2D(filter_size, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal')

    def dense(unit_size):
        return tf.keras.layers.Dense(unit_size, use_bias=False, kernel_initializer='he_normal')

    def global_average_pooling():
        return tf.keras.layers.GlobalAveragePooling2D()

    def relu():
        return tf.keras.layers.ReLU()

    ####

    def residual_block(width):
        return rcompose(ljuxt(rcompose(batch_normalization(),
                                       conv(width),
                                       batch_normalization(),
                                       relu(),
                                       conv(width),
                                       batch_normalization()),
                              identity),
                        add())

    W = 1024
    H =    4

    return rcompose(conv(W, 1),
                    rcompose(*repeatedly(partial(residual_block, W), H)),
                    global_average_pooling(),
                    dense(1),
                    relu())  # マイナスの値が出ると面倒な気がするので、ReLUしてみました。


def main():
    def create_model():
        result = tf.keras.Model(*juxt(identity, computational_graph())(tf.keras.Input(shape=(3, 3, 6 * 6))))

        result.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        result.summary()

        return result

    def create_generator(batch_size):
        while True:
            xs = []
            ys = []

            for i in range(batch_size):
                step = randrange(1, 32)

                xs.append(get_x(get_random_state(step)[0]))
                ys.append(step)

            yield np.array(xs), np.array(ys)

    model_path = Path('./model/cost.h5')

    model = create_model() if not model_path.exists() else tf.keras.models.load_model(model_path)
    model.fit_generator(create_generator(1000), steps_per_epoch=1000, epochs=100)

    model_path.parent.mkdir(exist_ok=True)
    tf.keras.models.save_model(model, 'model/cost.h5')

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
