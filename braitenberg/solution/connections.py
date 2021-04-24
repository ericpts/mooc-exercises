from typing import Tuple

import numpy as np

middle = 320
upper_half_horiz = 100

lower_half_horiz = 150
lower_half_width = 250

HIGH = 1
MED = 0.5
LOW = 0.1


def linspace(*args, **kwargs):
    return np.expand_dims(np.linspace(*args, **kwargs), axis=-1)


def set_upper_half(m, sign: int, horiz: int):
    m[:horiz, :middle] = (
        linspace(sign * HIGH, sign * LOW, num=horiz) @ linspace(MED, HIGH, num=middle).T
    )
    m[:horiz, middle:] = (
        linspace(-sign * HIGH, -sign * LOW, num=horiz)
        @ linspace(HIGH, MED, num=middle).T
    )


def set_lower_half(m, sign: int, horiz: int, width: int):
    (size_x, size_y) = m.shape
    assert middle == size_y / 2
    m[horiz:size_x, middle - width : middle] = (
        linspace(sign * LOW, sign * HIGH, num=size_x - horiz)
        @ linspace(MED, HIGH, num=width).T
    )

    m[horiz:size_x, middle : middle + width] = (
        linspace(-sign * LOW, -sign * HIGH, num=size_x - horiz)
        @ linspace(HIGH, MED, num=width).T
    )


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(
        shape=shape, dtype="float32"
    )  # write your function instead of this one
    set_upper_half(res, sign=-1, horiz=upper_half_horiz)
    set_lower_half(res, sign=1, horiz=lower_half_horiz, width=lower_half_width)
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(
        shape=shape, dtype="float32"
    )  # write your function instead of this one
    set_upper_half(res, sign=1, horiz=upper_half_horiz)
    set_lower_half(res, sign=-1, horiz=lower_half_horiz, width=lower_half_width)
    return res
