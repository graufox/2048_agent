from funcs import board_2_array

import numpy as np
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
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
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
        self.board[i, j] = 2

    def slide_down(self, testing=False):
        """slides the tiles downward, combining as necessary"""
        # in each column, move the pieces downward
        new_board = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        # for each column:
        for j in range(self.board_size):
            col = self.board[:, j]
            new_col, reward = slide_column_down(col, return_score=True)
            # update the column
            new_board[:, j] = new_col
            if not testing:
                self.score += reward

        # if we have the same board, don't increase
        #   the number of steps taken or add a new tile
        if (self.board != new_board).any():

            # only perform update if we're not in test mode
            if not testing:
                # update board and move count
                self.board = new_board
                self.num_moves += 1
                # add a tile at random
                self.add_random_tile()

            return 1

        # otherwise make no changes
        else:
            # make no changes if we have the same board
            return 0

    def add_random_tile(self):
        """add new tile (usually 2) to a random blank spot"""
        blank_row_idxs, blank_col_idxs = np.where(self.board == 0)
        num_blanks = len(blank_row_idxs)
        if num_blanks > 0:
            selected_position = randint(num_blanks)
            selected_row_idx = blank_row_idxs[selected_position]
            selected_col_idx = blank_col_idxs[selected_position]
            tile_value = (2 if (rand() < 0.9) else 4)
            self.board[selected_row_idx, selected_col_idx] = tile_value

    def slide_up(self, testing=False):
        """slides the tiles upward, combining as necessary"""
        self.board = self.board[::-1]
        available = self.slide_down(testing=testing)
        self.board = self.board[::-1]
        return available

    def slide_right(self, testing=False):
        """slides the tiles to the right, combining as necessary"""
        self.board = self.board.transpose()
        available = self.slide_down(testing=testing)
        self.board = self.board.transpose()
        return available

    def slide_left(self, testing=False):
        """slides the tiles to the left, combining as necessary"""
        self.board = self.board.transpose()
        available = self.slide_up(testing=testing)
        self.board = self.board.transpose()
        return available

    def available_moves(self):
        """returns all available moves at the current state"""
        moves = []
        moves += [self.slide_up(testing=True)]
        moves += [self.slide_right(testing=True)]
        moves += [self.slide_down(testing=True)]
        moves += [self.slide_left(testing=True)]
        return np.array([moves], dtype=np.float32)

    def is_done(self):
        """check if the game is done"""
        board_full = (self.board > 0).all()
        diffs_x = np.diff(self.board)
        diffs_y = np.diff(self.board.transpose())
        fully_mixed = (np.abs(diffs_x) > 0).all() and (np.abs(diffs_y) > 0).all()
        done = board_full and fully_mixed
        return done

    def reset(self):
        """for openai gym"""
        self.refresh_board()
        self.score = 0
        self.num_moves = 0
        return board_2_array(self.board, self.board_size)

    def step(self, action):
        """for openai gym"""

        old_score = self.score
        # decode action
        if action == 0:
            self.slide_up()
        elif action == 1:
            self.slide_right()
        elif action == 2:
            self.slide_down()
        elif action == 3:
            self.slide_left()

        reward = self.score - old_score
        if not self.is_done():
            self.last_board = self.board

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
