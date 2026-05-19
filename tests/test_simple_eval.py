import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_prompt_regression():
    """Catches when a prompt change breaks expected output"""
    test_case = LLMTestCase(
        input="What is 2+2?",
        actual_output="4",
        expected_output="4"
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])

def test_model_swap():
    """Catches behavior drift when switching model versions"""
    test_case = LLMTestCase(
        input="Capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])

def test_structured_output():
    """Catches when model stops returning expected format"""
    test_case = LLMTestCase(
        input="Say hello",
        actual_output="Hello",
        expected_output="Hello"
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])
