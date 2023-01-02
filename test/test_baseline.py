"""Tests for arithmetic.py"""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from timeflux.core.io import Port
from timeflux_neurofeedback_inverse_gamepad.nodes.baseline import Baseline


def test_baseline():
    node = Baseline()
    node.i = Port()
    node.i.data = pd.DataFrame([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    node.update()
    expected = pd.DataFrame([[1.0, 0.5, 0.25]], index=[2])
    #    print(node.o.data)
    #    print(expected)
    assert_frame_equal(node.o.data, expected)
