import numpy as np


def ema(data, a):
    """
    exponential moving average of `data`
        data   =  [1-D numpy array] the data to smooth
        a      =  [float between 0 and 1] the smoothness factor
    """
    assert len(data.shape) == 1
    smooth_data = np.zeros(data.shape)
    if len(data) > 0:
        smooth_data[0] = data[0]
        for j in range(1, len(data)):
            smooth_data[j] = (1 - a) * smooth_data[j - 1] + a * data[j]
    return smooth_data


def board_2_array(board, board_size=4, board_depth=17):
    """
    outputs a 4 x 4 x 16 array with one-hot encodings for the tiles
        board  = 4 x 4 numpy array of tile values associated to board
    """
    assert len(board.shape) == 2
    channels = []
    for i in range(1, board_depth):
        channels += [np.ones(shape=(board_size, board_size)) * (2**i) == board]
    return np.stack(channels, axis=-1)


def sample_action(Qvals, moves):
    """
    Samples an action proportional to Qvals and available moves
    """
    temp = (Qvals[0] - Qvals[0].min()) * moves[0]
    if temp.max() > 100:
        p = temp[0] / temp[0].sum()
    else:
        p = np.exp(temp)[0] * moves[0]
    if p.sum() < 1e-5:
        p = p > 0
    p = p / p.sum()
    return [np.random.choice([0, 1, 2, 3], p=p)]
