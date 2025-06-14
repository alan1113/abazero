import numpy as np
from collections import deque
import copy

from net import PolicyValueNet
from config import CONFIG
import mcts

# abalone
state_list_init = np.array(
    [
        [1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 1, 1, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 2, 2, 2, 0, 0],
        [-1, -1, -1, 2, 2, 2, 2, 2, 2],
        [-1, -1, -1, -1, 2, 2, 2, 2, 2],
    ]
)

move_dir = {
    0: (1, 0),  # right
    1: (-1, 0),  # left
    2: (0, -1),  # upper_right
    3: (-1, -1),  # upper_left
    4: (1, 1),  # bottom_right
    5: (0, 1),  # bottom_left
}

state_deque_init = deque(maxlen=2)
for _ in range(2):
    state_deque_init.append(copy.deepcopy(state_list_init))


def change_state(state_list, move, player):
    board = copy.deepcopy(state_list)
    y, x, direction = int(move / 54), int(move / 6) % 9, move % 6
    cx, cy = x, y
    if player == board[y][x]:
        psum, esum = 0, 0
        mx, my = move_dir[direction]
        while 0 <= cx < 9 and 0 <= cy < 9 and board[cy][cx] == player:
            cx += mx
            cy += my
            psum += 1
        while 0 <= cx < 9 and 0 <= cy < 9 and board[cy][cx] == player % 2 + 1:
            cx += mx
            cy += my
            esum += 1
        ex, ey = x + mx * (psum + esum), y + my * (psum + esum)
        if (
                psum > esum
                and ex < 0 or ex >= 9  # 添加Y轴边界检查
                or ey < 0 or ey >= 9  # 添加X轴边界检查
                or board[ey][ex] != player % 2 + 1
        ):
            px, py = x + mx * psum, y + my * psum
            ex, ey = x + mx * (psum + esum), y + my * (psum + esum)
            board[y][x] = 0
            if 0 <= px < 9 and 0 <= py < 9 and board[py][px] != -1:
                board[py][px] = player
            if 0 <= ex < 9 and 0 <= ey < 9 and board[ey][ex] != -1 and esum != 0:
                board[ey][ex] = player % 2 + 1
    return board


def get_legal_moves(state_deque, current_player_color) -> np.ndarray:
    board = state_deque[-1]
    legal = np.zeros(board.shape + (6,), dtype=int)
    for y in range(9):
        for x in range(9):
            if current_player_color == board[y][x]:
                for i, direction in move_dir.items():
                    # print(board[y][x],x,y,direction)
                    cx, cy = x, y
                    psum, esum = 0, 0
                    mx, my = direction
                    while (
                            0 <= cx < 9
                            and 0 <= cy < 9
                            and board[cy][cx] == current_player_color
                    ):
                        cx += mx
                        cy += my
                        psum += 1

                    while (
                            0 <= cx < 9
                            and 0 <= cy < 9
                            and board[cy][cx] == current_player_color % 2 + 1
                    ):
                        cx += mx
                        cy += my
                        esum += 1
                    final_y = y + my * (psum + esum)
                    final_x = x + mx * (psum + esum)
                    if (
                            psum > esum
                            and final_y < 0 or final_y >= 9  # 添加Y轴边界检查
                            or final_x < 0 or final_x >= 9  # 添加X轴边界检查
                            or board[final_y][final_x] != current_player_color % 2 + 1
                            != current_player_color % 2 + 1
                    ):
                        legal[y][x][i] = 1

    return legal.flatten()  # 1 dim ndarray


class Board(object):

    def __init__(self):
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = 0
        self.state_deque = copy.deepcopy(state_deque_init)

    # 初始化棋盘的方法
    def init_board(self, start_player=1):  # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字典
        # 永远是红方先移动
        self.start_player = start_player

        # 当前手玩家，也就是先手玩家
        self.current_player_color = start_player
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill = np.zeros(2)
        self.limit = 6
        self.game_start = False
        self.action_count = 0  # 游戏动作计数器
        self.winner = 0

    def availables(self) -> np.ndarray:
        return get_legal_moves(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        _current_state = np.zeros([1, 6, 9, 9], dtype=np.float32)
        _current_state[0][0] = (self.state_deque[-1] == 1).astype(np.float32)
        _current_state[0][1] = (self.state_deque[-1] == 2).astype(np.float32)
        _current_state[0][2] = (self.state_deque[-2] == 1).astype(np.float32)
        _current_state[0][3] = (self.state_deque[-2] == 2).astype(np.float32)
        if self.current_player_color == 1:
            _current_state[0][-2] = np.ones([9, 9])
        else:
            _current_state[0][-1] = np.ones([9, 9])
        return _current_state

    # 根据move对棋盘状态做出改变
    def do_move(self, move) -> None:
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        self.change_state(move, self.current_player_color)
        self.current_player_color = self.current_player_color % 2 + 1

    # 是否产生赢家
    def has_a_winner(self) -> int:
        if self.kill[0] == self.limit:
            self.winner = 1
        elif self.kill[1] == self.limit:
            self.winner = 2
        return self.winner

    def get_current_player_color(self):
        return self.current_player_color

    def change_state(self, move, player) -> None:
        board = copy.deepcopy(self.state_deque[-1])
        y, x, direction = int(move / 54), int(move / 6) % 9, move % 6
        cx, cy = x, y
        if player == board[y][x]:
            psum, esum = 0, 0
            mx, my = move_dir[direction]
            while 0 <= cx < 9 and 0 <= cy < 9 and board[cy][cx] == player:
                cx += mx
                cy += my
                psum += 1
            while 0 <= cx < 9 and 0 <= cy < 9 and board[cy][cx] == player % 2 + 1:
                cx += mx
                cy += my
                esum += 1
            ex, ey = x + mx * (psum + esum), y + my * (psum + esum)
            if (
                    psum > esum
                    and ex < 0 or ex >= 9  # 添加Y轴边界检查
                    or ey < 0 or ey >= 9  # 添加X轴边界检查
                    or board[ey][ex] != player
            ):
                px, py = x + mx * psum, y + my * psum
                # ex, ey = x + mx * (psum + esum), y + my * (psum + esum)
                board[y][x] = 0
                if 0 <= px < 9 and 0 <= py < 9 and board[py][px] != -1:
                    board[py][px] = player
                    if 0 <= ex < 9 and 0 <= ey < 9 and board[ey][ex] != -1 and esum != 0:
                        board[ey][ex] = player % 2 + 1
                    elif esum != 0:
                        self.kill[self.current_player_color % 2 - 1] += 1
                else:
                    self.kill[self.current_player_color - 1] += 1

        self.state_deque.append(board)


class Game(object):

    def __init__(self, board):
        self.board: Board = board

    # 可视化
    def graphic(self, board, player1_color, player2_color):
        print("player1 take: ", player1_color)
        print("player2 take: ", player2_color)
        print(board.state_deque[-1])

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=True, temp=1e-3):
        self.board.init_board()  # 初始化棋盘, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            _count += 1
            move, move_probs = player.get_action(
                self.board, temp=temp, return_prob=1
            )


            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.get_current_player_color())

            # 执行一步落子
            self.board.do_move(move)
            print(_count)
            print(self.board.state_deque[-1])
            print(int(move / 54), int(move / 6) % 9, move % 6)
            print('-----')
            winner = self.board.has_a_winner()
            if winner:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                winner_z[np.array(current_players) == winner] = 1.0
                winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print("Game end. Tie")

                return winner, zip(states, mcts_probs, winner_z)


if __name__ == "__main__":
    # 測試：玩家 1 從 (2, 4) 向 bottom_right 移動
    model_path = CONFIG["pytorch_model_path"]
    player = mcts.MCTSPlayer(PolicyValueNet(model_file=model_path).policy_value_fn, n_playout=100,
                             is_selfplay=1)
    board = Board()
    game = Game(board)
    game.start_self_play(player)
