import numpy as np

from funcy    import *
from operator import *
from random   import *


def _create_actions():
    result = {}

    #  00 01 02   08 09 10
    #  07 FF 03   15 RR 11
    #  06 05 04   14 13 12
    #
    #             16 17 18   24 25 26
    #             23 DD 19   31 BB 27
    #             22 21 20   30 29 28
    #
    #                        32 33 34  40 41 42
    #                        39 LL 35  47 UU 43
    #                        38 37 36  46 45 44

    action_targets = (( 0,  1,  2,  3,  4,  5,  6,  7, 46, 45, 44,  8, 15, 14, 16, 23, 22, 38, 37, 36),
                      ( 8,  9, 10, 11, 12, 13, 14, 15, 44, 43, 42, 26, 25, 24, 18, 17, 16,  4,  3,  2),
                      (16, 17, 18, 19, 20, 21, 22, 23, 14, 13, 12, 24, 31, 30, 32, 39, 38,  6,  5,  4),
                      (24, 25, 26, 27, 28, 29, 30, 31, 12, 11, 10, 42, 41, 40, 34, 33, 32, 20, 19, 18),
                      (32, 33, 34, 35, 36, 37, 38, 39, 30, 29, 28, 40, 47, 46,  0,  7,  6, 22, 21, 20),
                      (40, 41, 42, 43, 44, 45, 46, 47, 28, 27, 26, 10,  9,  8,  2,  1,  0, 36, 35, 34))

    #      08 09 10
    #
    #  19  00 01 02  11
    #  18  07    03  12
    #  17  06 05 04  13
    #
    #      16 15 14

    rotate = [6, 7, 0, 1, 2, 3, 4, 5, 17, 18, 19, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    for action_name, action_target in zip(['F', 'R', 'D', 'B', 'L', 'U'], action_targets):
        result[action_name      ] = list(action_target), list(np.array(action_target)[rotate]                )
        result[action_name + "'"] = list(action_target), list(np.array(action_target)[rotate][rotate][rotate])

    return result


ACTIONS    = _create_actions()
GOAL_STATE = (0,) * 8 + (1,) * 8 + (2,) * 8 + (3,) * 8 + (4,) * 8 + (5,) * 8


def get_rev_action(action):
    return action[0] if action[-1] == "'" else action + "'"


def get_random_state(step):  # TODO: リファクタリングする。random_actionsとアクションを実行した結果の状態を取得する関数に分ける。
    def random_actions():
        result = []

        while len(result) < step:
            action = choice(tuple(ACTIONS.keys()))

            if not result or action != get_rev_action(last(result)):
                result.append(action)

        return tuple(result)

    state   = GOAL_STATE
    actions = random_actions()

    for action in actions:
        state = get_next_state(state, action)

    return state, actions


def get_next_state(state, action):
    np_state = np.array(state)

    np_state[ACTIONS[action][0]] = np_state[ACTIONS[action][1]]

    return tuple(np_state)


def render_string(state):
    ns = np.array(state + (6,))[[-1, -1, -1, 40, 41, 42, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, 47, -1, 43, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, 46, 45, 44, -1, -1, -1, -1, -1, -1,
                                 34, 35, 36,  0,  1,  2,  8,  9, 10, 26, 27, 28,
                                 33, -1, 37,  7, -1,  3, 15, -1, 11, 25, -1, 29,
                                 32, 39, 38,  6,  5,  4, 14, 13, 12, 24, 31, 30,
                                 -1, -1, -1, 22, 23, 16, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, 21, -1, 17, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, 20, 19, 18, -1, -1, -1, -1, -1, -1]]

    cs = np.array(('F ', 'R ', 'D ', 'B ', 'L ', 'U ', '  '))[ns]

    return '\n'.join(map(lambda line: ''.join(line), partition(12, cs)))


def get_x(state):
    result = np.array(((0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0)),
                      dtype=np.float)

    np_state       = np.array(state)
    target_indexes = np.concatenate(tuple(map(partial(add, np.array((0, 1, 2, 5, 8, 7, 6, 3))), take(6, iterate(partial(add, 9), 0)))))

    for i in range(6):
        result[i][target_indexes] = (np_state == i)

    return np.transpose(np.reshape(result, (6 * 6, 3, 3)), (1, 2, 0))
