
def shift_func(shift_channels=3):
    if shift_channels == 3:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    elif shift_channels == 7:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1]]
    elif shift_channels == 9:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9]]
    elif shift_channels == 15:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9],
                # long range direct hood
                [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4]]
    elif shift_channels == 17:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9],
                # long range direct hood
                [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                # inplane diagonal dam edges
                [0, -27, 0], [0, 0, -27]]
    elif shift_channels == 23:
        shift = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9],
                # long range direct hood
                [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                # inplane diagonal dam edges
                [0, -27, 0], [0, 0, -27],
                # new
                [0, -27, -27], [0, 27, -27], [0, -27, -9], [0, -9, -27], [0, 9, -27], [0, 27, -9]]
    else:
        raise NotImplementedError
    return shift
