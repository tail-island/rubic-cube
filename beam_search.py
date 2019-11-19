from game  import *
from heapq import *


def get_answer(initial_state, cost_model, n):
    def get_next_state_and_next_answers():
        for _ in range(min(n, len(queue))):
            _, state, answer = heappop(queue)

            for action in ACTIONS.keys():
                next_state  = get_next_state(state, action)
                next_answer = answer + (action,)

                if next_state not in visited_states or visited_states[next_state] > len(next_answer):
                    visited_states[next_state] = len(next_answer)

                    yield next_state, next_answer

    queue = [(0, initial_state, ())]
    visited_states = {initial_state: 0}

    while queue:
        next_queue = []

        next_states, next_answers = zip(*get_next_state_and_next_answers())

        for next_state, next_answer in zip(next_states, next_answers):
            if next_state == GOAL_STATE:
                return next_answer

        cost_to_goals = cost_model.predict(np.array(tuple(map(get_x, next_states))), batch_size=10000).flatten()

        for next_state, next_answer, cost_to_goal in zip(next_states, next_answers, cost_to_goals):
            heappush(next_queue, (cost_to_goal, next_state, next_answer))

        queue = next_queue

    return ()
