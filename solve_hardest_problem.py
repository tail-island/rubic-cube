import batch_weighted_a_star
import tensorflow as tf

from game import *
from time import *


def main():
    model = tf.keras.models.load_model('model/cost.h5')

    question = "U U F U U R' L F F U F' B' R L U U R U D' R L' D R' L' D D".split(' ')   # 26手問題
    # question = "U U F U U R' L F F U F' B' R L U U R U D' R L' D R' L' D'".split(' ')  # 25手問題
    # question = "U F U U R' L F F U F' B' R L U U L U D' R' L D R' L' U U".split(' ')   # 25手問題

    state = GOAL_STATE

    for action in question:
        state = get_next_state(state, action)

    starting_time = time()

    answer = batch_weighted_a_star.get_answer(state, model, 10000, 0.6)  # 論文だと、最適解を出す場合はn=10000でl=0.6が良いらしい。

    print(f'{len(answer)} steps, {time() - starting_time:6.3f} seconds')
    print(' '.join(map(lambda action: action if len(action) == 2 else action + ' ', question)))
    print(' '.join(map(lambda action: action if len(action) == 2 else action + ' ', answer  )))

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
