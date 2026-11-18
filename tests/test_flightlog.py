import numpy as np

from splatsim.flightlog import matrix_from_quat_wxyz, quat_wxyz_from_matrix


def test_quaternion_round_trip():
    # Both branches of the conversion: a small rotation and two half turns.
    for quat in [(0.9239, 0.0, 0.3827, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.5, 0.5, 0.5)]:
        back = quat_wxyz_from_matrix(matrix_from_quat_wxyz(quat))
        # q and -q are the same rotation.
        assert abs(abs(np.dot(back, quat)) - 1) < 1e-4
