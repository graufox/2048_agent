from funcs import board_2_array

import numpy as np
import pandas as pd
from icecream import ic
from numpy.random import rand, randint


def slide_column_down(column, return_score=True):
    """
    Slide a column downwards, assuming index 0 is at the top.
    """

    squished_column = [value for value in column if value > 0]

    new_column = []
    score = 0
    idx = len(squished_column) - 1
    while idx >= 0:
        value = squished_column[idx]
        if idx >= 1:
            value_above = squished_column[idx - 1]
            if value == value_above:
                new_column.append(2 * value)
                score += 2 * value
                idx -= 1
            else:
                new_column.append(value)
        else:
            new_column.append(value)
        idx -= 1
    new_column = [0] * (len(column) - len(new_column)) + new_column[::-1]
    if return_score:
        return new_column, score
    else:
        return new_column


class Game:
    """
    2048

    python class for the game 2048
    """

    def __init__(self, board_size=4, board_depth=17):
        self.board_size = board_size
        self.board_depth = board_depth
        self.board = np.zeros((board_size, board_size, board_size), dtype=np.int32)
        self.score = 0
        self.num_moves = 0
        self.action_space = [0, 1, 2, 3]
        self.last_board = self.board
        self.refresh_board()

    def refresh_board(self):
        """resets the board and places a tile randomly"""
        # clear the board
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        # put a 2 in a random place on the board
        i, j = randint(self.board_size), randint(self.board_size)
        self.add_random_tile()

    def slide_down(self):
        """slides the tiles downward, combining as necessary"""
        # in each column, move the pieces downward
        new_board = np.array(self.board)

        # for each column:
        slide_reward = 0
        for j in range(self.board_size):
            col = new_board[:, j]
            new_col, reward = slide_column_down(col, return_score=True)
            # update the column
            new_board[:, j] = new_col
            slide_reward += reward

        # update board and move count
        self.board = new_board
        self.num_moves += 1
        self.score += slide_reward
        return slide_reward

    def add_random_tile(self):
        """add new tile (usually 2) to a random blank spot"""
        blank_row_idxs, blank_col_idxs = np.where(self.board <= 0)
        num_blanks = len(blank_row_idxs)
        selected_position = randint(num_blanks)
        selected_row_idx = blank_row_idxs[selected_position]
        selected_col_idx = blank_col_idxs[selected_position]
        tile_value = 2 if (rand() < 0.9) else 4
        self.board[selected_row_idx, selected_col_idx] = tile_value

    def slide_up(self):
        """slides the tiles upward, combining as necessary"""
        self.board = self.board[::-1]
        slide_reward = self.slide_down()
        self.board = self.board[::-1]
        return slide_reward

    def slide_right(self):
        """slides the tiles to the right, combining as necessary"""
        self.board = np.rot90(self.board, -1)
        slide_reward = self.slide_down()
        self.board = np.rot90(self.board, 1)
        return slide_reward

    def slide_left(self):
        """slides the tiles to the left, combining as necessary"""
        self.board = np.rot90(self.board, 1)
        slide_reward = self.slide_up()
        self.board = np.rot90(self.board, -1)
        return slide_reward

    def available_moves(self):
        """returns all available moves at the current state"""

        def check_slide_up(board):
            for col_idx in range(self.board_size):
                col = np.array(board[:, col_idx]).astype(int)
                repeat_idxs = pd.Series(col).duplicated().values
                if (col[repeat_idxs] > 0).any():
                    print(col[repeat_idxs])
                    return True
            return False

        moves = np.zeros((4,), dtype=bool)
        for k in range(4):
            moves[k] = check_slide_up(np.rot90(self.board, k))
        return np.array([moves], dtype=np.float32)

    def is_done(self):
        """check if the game is done"""
        done = False
        board_full = (self.board > 0).all()
        if board_full:
            diffs_x = np.diff(self.board)
            diffs_y = np.diff(self.board.transpose())
            print(diffs_x, diffs_y)
            fully_mixed = (np.abs(diffs_x) > 0).all() and (np.abs(diffs_y) > 0).all()
            done = board_full and fully_mixed
        return done

    def reset(self):
        """for openai gym"""
        self.refresh_board()
        self.score = 0
        self.num_moves = 0
        return board_2_array(self.board, self.board_size, self.board_depth)

    def step(self, action):
        """for openai gym"""

        old_score = self.score
        # decode action
        if action == 0:
            slide_reward = self.slide_up()
        elif action == 1:
            slide_reward = self.slide_right()
        elif action == 2:
            slide_reward = self.slide_down()
        elif action == 3:
            slide_reward = self.slide_left()

        reward = np.log1p(slide_reward)
        done = self.is_done()
        if not done:
            self.last_board = self.board
            self.add_random_tile()

        return (
            board_2_array(self.board, self.board_size, self.board_depth),
            reward,
            self.is_done(),
            {},
        )


if __name__ == "__main__":
    g = Game(board_size=4)
    g.board = np.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]], dtype=np.int32
    )
    print(g.board)

    g.slide_down()
    print("||||||")
    print(g.board)

    g.slide_up()
    print("||||||")
    print(g.board)

    g.slide_left()
    print("||||||")
    print(g.board)

    g.slide_right()
    print("||||||")
    print(g.board)

    g.board = np.array(
        [[4, 8, 2, 32], [8, 2, 4, 8], [16, 4, 8, 32], [2, 8, 16, 2]], dtype=np.int32
    )
    print(g.board)
    g.slide_down()
