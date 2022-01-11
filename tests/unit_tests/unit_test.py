"""
Module for unit testing.
"""

from os.path import join

import pytest

from lib.scripts.global_variables import (UNIT_TEST_MODEL_PATH, UNIT_TEST_SCENARIOS_PATH,
                                          UNIT_TEST_MODEL_CHECKPOINT, UNIT_TEST_SCENARIOS_CHECKPOINT)
from lib.scripts.reader import read_json, load_model


def collect_data() -> list:
    """Collects all test cases.

    Returns:
        List of tuples with input and output test cases.
    """
    test_scenarios = []
    json_scenarios = read_json(join(UNIT_TEST_SCENARIOS_PATH, UNIT_TEST_SCENARIOS_CHECKPOINT))

    for json_scenario in json_scenarios:
        test_scenarios.append((json_scenario['input'], json_scenario['output']))

    return test_scenarios


@pytest.mark.parametrize('test_input, test_output', collect_data())
def test_classifier(test_input, test_output) -> None:
    """Tests classifier for basic test cases.

    Args:
        test_input:  Input test case.
        test_output: Expected classifier output.

    """
    clf = load_model(join(UNIT_TEST_MODEL_PATH, UNIT_TEST_MODEL_CHECKPOINT))
    clf_response = clf.predict([test_input])[0]

    error_string = f"Classifier issued wrong answer:\n" \
                   f"{clf_response}:\n" \
                   f"While expecting:\n" \
                   f"{test_output}"

    assert clf_response == test_output, error_string
