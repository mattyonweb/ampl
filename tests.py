from typing import *

from m import *

def assert_delta(x, expected):
    assert expected - 0.01 <= x <= expected + 0.01, f"Expected: {expected}; obtained {x}"
    
def test_env_asr():
    atk = 10
    rel = 10
    sus = 1.0

    assert_delta(env_asr(0, Trigger(0, 1), atk, sus, rel), 0)
    assert_delta(env_asr(1, Trigger(0, 0), atk, sus, rel), 0.1)
    assert_delta(env_asr(2, Trigger(0, 0), atk, sus, rel), 0.2)
    
    assert_delta(env_asr(9, Trigger(0, 0), atk, sus, rel), 0.9)
    assert_delta(env_asr(10, Trigger(0, 0), atk, sus, rel), 1)
    assert_delta(env_asr(11, Trigger(0, 0), atk, sus, rel), 0.9)
    assert_delta(env_asr(12, Trigger(0, 0), atk, sus, rel), 0.8)
    assert_delta(env_asr(19, Trigger(0, 0), atk, sus, rel), 0.1)
    assert_delta(env_asr(20, Trigger(0, 0), atk, sus, rel), 0)
    
    
