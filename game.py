import numpy as np
from numpy.random import rand, randint

from funcs import board_2_array


class Game:
    """
    2048

    python class for the game 2048
    """

    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.num_moves = 0

        self.action_space = [0, 1, 2, 3]
        self.last_board = self.board

        self.refresh_board()

    def refresh_board(self):
        """resets the board and places a tile randomly"""
        # clear the board
        self.board = np.zeros((4, 4), dtype=np.int32)
        # put a 2 in a random place on the board
        i, j = randint(4), randint(4)
        self.board[i, j] = 2

    def slide_down(self, testing=False):
        """slides the tiles downward, combining as necessary"""
        # in each column, move the pieces downward
        new_board = np.zeros(self.board.shape, dtype=np.int32)

        # for each column:
        for j in range(4):

            # extract the non-zero elements in order
            col = self.board[:, j]
            squished_col = col[col > 0].tolist()
            # print(squished_col)
            # calculate the number of blank spots
            num_blanks = 4 - len(squished_col)

            # find matches, starting from the end
            new_col_reversed = []
            while len(squished_col) > 1:
                val = squished_col[-1]
                val_above = squished_col[-2]
                if val == val_above:
                    new_col_reversed.append(2 * val)
                    self.score += 2 * val
                    squished_col.pop()
                else:
                    new_col_reversed.append(val)
                squished_col.pop()
            new_col_reversed += squished_col[::-1]
            new_col_reversed += [0] * (4 - len(new_col_reversed))
            new_col = new_col_reversed[::-1]

            # update the column
            new_board[:, j] = new_col

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
        total_blanks = (self.board == 0).sum()
        if total_blanks > 0:
            blank_ids = []
            for i in range(4):
                for j in range(4):
                    if self.board[i, j] == 0:
                        blank_ids += [(i, j)]
            pos = blank_ids[randint(len(blank_ids))]
            self.board[pos] = 2

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
        return board_2_array(self.board)

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
        # if reward < 0.1: # if no score increase, dock points
        #     reward -= 2
        # if (self.board == self.last_board).all():
        #     reward -= 64
        # if self.is_done():
        #     reward -= 2**10
        if not self.is_done():
            self.last_board = self.board

        return board_2_array(self.board), reward, self.is_done(), {}


if __name__ == "__main__":

    g = Game()
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
