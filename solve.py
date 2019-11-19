import batch_weighted_a_star
import beam_search
import tensorflow as tf

from game   import *
from random import *
from time   import *


def main():
    model = tf.keras.models.load_model('model/cost.h5')

    seed(0)

    for _ in range(10):
        state, question = get_random_state(32)

        starting_time = time()
        answer = batch_weighted_a_star.get_answer(state, model, 100, 0.2)  # DeepCubeAのWebサイトは、n=100でl=0.2らしい。
        # answer = beam_search.get_answer(state, model, 100)               # l=0.2だと古いのはほぼ捨てられるので、ビーム・サーチとあまり変わりません。


        print(f'{len(answer)} steps, {time() - starting_time:6.3f} seconds')
        print(' '.join(map(lambda action: action if len(action) == 2 else action + ' ', question)))
        print(' '.join(map(lambda action: action if len(action) == 2 else action + ' ', answer  )))

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
