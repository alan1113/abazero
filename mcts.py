import numpy as np
import copy
from config import CONFIG


def softmax(x)->np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children: dict[int, 'TreeNode'] = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for i in range(len(action_priors)):
            if action_priors[i] > 0:
                if i not in self._children:
                    self._children[i] = TreeNode(self, action_priors[i])

    def select(self, c_puct:float)->tuple[int,  'TreeNode']:
        if not self._children:
            print('No child')
            return None
        return max(
            self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def get_value(self, c_puct:float) -> float:
        self._u = (
            c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        )
        return self._Q + self._u

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state: 'game.Board') -> None:
        node = self._root  # from root search again
        while True:
            if node.is_leaf():  # find leaf
                break
            else:
                action, node = node.select(self._c_puct)  # continue search
                # print(type(action))
                # print(action)
                state.do_move(action)  # move for policy
        # print(state.state_deque[-1])

        action_probs, left_value = self._policy(state)
        # print(action_probs)
        winner = state.has_a_winner()
        if not winner:
            node.expand(action_probs)
        else:
            left_value = 1.0 if winner == state.get_current_player_color else -1.0

        node.update_recursive(-left_value)

    def get_move_probs(self, state, temp=1e-3) -> tuple[np.ndarray, np.ndarray]:
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            # print(n)
        items = list(self._root._children.items())
        acts = np.array([act for act, _ in items])
        visits = np.array([node._n_visits for _, node in items])
        # print(visits)

        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move: int):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return "MCTS {}".format(self.player)

    def get_action(self, board, temp=1e-3, return_prob=0) -> tuple[int, np.ndarray]:
        # move_probs = np.zeros(486)
        acts, probs = self.mcts.get_move_probs(board, temp)
        # move_probs[list(acts)] = probs
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75 * probs
                + (
                    0.25
                    * np.random.dirichlet(CONFIG["dirichlet"] * np.ones(len(probs)))
                )
            )

            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        return move, probs
        # if return_prob:
        #     return move, probs
        # else:
        #     return move
