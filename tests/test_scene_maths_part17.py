import numpy as np
import pytest
import part17_agentic as p17


def test_part17_only_model_tokens_trained():
    m = p17.loss_mask(); assert m.sum() == 12 and len(m) == 25


def test_part17_mask_zero_on_tool_output():
    m = p17.loss_mask([("model", 2), ("tool", 3)]); assert list(m) == [1, 1, 0, 0, 0]
