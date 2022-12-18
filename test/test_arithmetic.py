"""Tests for arithmetic.py"""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from timeflux.core.io import Port
from timeflux_example.nodes.arithmetic import Add, MatrixAdd

def test_add():
    node = Add(1)
    node.i = Port()
    node.i.data = pd.DataFrame([[1, 1], [1, 1]])
    node.update()
    expected = pd.DataFrame([[2, 2], [2, 2]])
    assert_frame_equal(node.o.data, expected)

def test_matrix():
    node = MatrixAdd()
    node.i_m1 = Port()
    node.i_m2 = Port()
    node.i_m1.data = pd.DataFrame([[1, 1], [1, 1]])
    node.i_m2.data = pd.DataFrame([[2, 2], [2, 2]])
    node.update()
    expected = pd.DataFrame([[3, 3], [3, 3]])
    assert_frame_equal(node.o.data, expected)
