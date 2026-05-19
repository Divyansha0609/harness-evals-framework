import pytest
import json

# When harness_evals is ready, replace this entire block with:
# from harness_evals import EvalCase, assert_test
# from harness_evals.metrics import ExactMatchMetric, LatencyMetric, StructuredOutputMetric, ToolRoutingMetric
# from harness_evals.sinks import JsonSink

class EvalCase:
    def __init__(self, input, output, expected, latency_ms=0):
        self.input = input
        self.output = output
        self.expected = expected
        self.latency_ms = latency_ms

class ExactMatchMetric:
    def measure(self, ec):
        return ec.output == ec.expected

class LatencyMetric:
    def __init__(self, max_ms):
        self.max_ms = max_ms
    def measure(self, ec):
        return ec.latency_ms < self.max_ms

class StructuredOutputMetric:
    def __init__(self, required_keys):
        self.required_keys = required_keys
    def measure(self, ec):
        try:
            parsed = json.loads(ec.output)
            return all(k in parsed for k in self.required_keys)
        except:
            return False

class ToolRoutingMetric:
    def __init__(self, expected_tool):
        self.expected_tool = expected_tool
    def measure(self, ec):
        return ec.output == self.expected_tool

class JsonSink:
    def __init__(self, path):
        self.path = path
    def write(self, ec, scores):
        with open(self.path, "a") as f:
            f.write(json.dumps({"input": ec.input, "scores": scores}) + "\n")

def assert_test(ec, metrics, sinks=None):
    scores = {}
    for metric in metrics:
        result = metric.measure(ec)
        scores[metric.__class__.__name__] = result
        assert result, f"[{metric.__class__.__name__}] FAILED on input: '{ec.input}'"
    if sinks:
        for sink in sinks:
            sink.write(ec, scores)

# Tests  

sink = JsonSink("results.jsonl")

def test_prompt_regression():
    """Catches when a prompt change breaks expected output"""
    ec = EvalCase(
        input="Capital of France?",
        output="Paris",
        expected="Paris",
        latency_ms=200
    )
    assert_test(ec, metrics=[ExactMatchMetric()], sinks=[sink])

def test_model_swap_drift():
    """Catches behavior drift when switching model versions"""
    ec = EvalCase(
        input="What is 2+2?",
        output="4",
        expected="4",
        latency_ms=180
    )
    assert_test(ec, metrics=[ExactMatchMetric()], sinks=[sink])

def test_latency_regression():
    """Catches when a change makes responses too slow"""
    ec = EvalCase(
        input="Say hello",
        output="Hello",
        expected="Hello",
        latency_ms=450
    )
    assert_test(ec, metrics=[LatencyMetric(max_ms=500)], sinks=[sink])

def test_structured_output():
    """Catches when model stops returning valid JSON structure"""
    ec = EvalCase(
        input="Give me Paris info as JSON",
        output='{"name": "Paris", "country": "France"}',
        expected="",
        latency_ms=300
    )
    assert_test(ec, metrics=[StructuredOutputMetric(required_keys=["name", "country"])], sinks=[sink])

def test_tool_routing():
    """Catches when adding a new tool breaks existing tool selection"""
    ec = EvalCase(
        input="search for flights to NYC",
        output="flight_search",
        expected="flight_search",
        latency_ms=100
    )
    assert_test(ec, metrics=[ToolRoutingMetric(expected_tool="flight_search")], sinks=[sink])
