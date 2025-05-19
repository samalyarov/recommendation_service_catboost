import pytest
from main import get_exp_group, Config

def test_get_exp_group_control():
    """Test that user is assigned to control group correctly."""
    # Using a known user_id that should be in control group
    user_id = 1
    result = get_exp_group(user_id)
    assert result == "control"

def test_get_exp_group_test():
    """Test that user is assigned to test group correctly."""
    # Using a known user_id that should be in test group
    user_id = 2
    result = get_exp_group(user_id)
    assert result == "test"

def test_get_exp_group_negative_id():
    """Test that negative user_id is handled correctly."""
    user_id = -1
    result = get_exp_group(user_id)
    assert result in ["control", "test"]  # Should still return a valid group

def test_get_exp_group_consistency():
    """Test that same user_id always gets same group."""
    user_id = 123
    result1 = get_exp_group(user_id)
    result2 = get_exp_group(user_id)
    assert result1 == result2

def test_config_defaults():
    """Test that Config class has correct default values."""
    assert Config.SPLIT_PERCENTAGE == 50
    assert Config.CHUNKSIZE == 100000
    assert Config.DEFAULT_RECOMMENDATION_LIMIT == 5
    assert isinstance(Config.SALT, str) 