import numpy as np
import pytest

import part18_wrapup as p18


def test_part18_graph_is_connected_and_edges_valid():
    assert all(a in p18.NODES and b in p18.NODES for a, b, _ in p18.EDGES) and p18.connected()


def test_part18_covers_parts_2_to_17_except_1_and_10():
    assert {p for p, _ in p18.NODES.values()} == set(range(2, 18)) - {10}
