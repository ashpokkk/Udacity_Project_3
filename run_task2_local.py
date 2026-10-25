"""
Local-only Task 2 test runner.
Does NOT edit config.py or tests/test_agent.py and does NOT contact CloudFormation.
Run from project root: uv run --active python run_task2_local.py
"""
import os
import runpy
import sys
import types

# Match the names the official test checks; actual AWS model invocation is not performed
# by the structural Task 2 tests.
config_stub = types.ModuleType("config")
config_stub.AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
config_stub.PROJECT_NAME = os.environ.get("PROJECT_NAME", "udacity-agentcore")
config_stub.ORCHESTRATOR_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
config_stub.WORKER_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
config_stub.AGENT_LOG_GROUP = "/aws/bedrock/agentcore/udacity-agentcore"
config_stub.ORDERS_TABLE = "local-test-orders"
config_stub.CUSTOMERS_TABLE = "local-test-customers"
config_stub.WORKFLOW_STATE_TABLE = "local-test-workflow-state"
config_stub.MEMORY_NAMESPACE = "local-test-memory-namespace"
config_stub.POLICY_BUCKET = "local-test-policy-bucket"
config_stub.VECTOR_STORE_BUCKET = "local-test-vector-bucket"
config_stub.AGENTCORE_ROLE_ARN = "arn:aws:iam::000000000000:role/local-test-role"
config_stub.RETURNS_KB_ID = "local-test-returns-kb"
config_stub.SHIPPING_KB_ID = "local-test-shipping-kb"
config_stub.WARRANTY_KB_ID = "local-test-warranty-kb"
config_stub.GUARDRAIL_NAME = "local-test-guardrail"
config_stub.GUARDRAIL_ID = ""
config_stub.GUARDRAIL_VERSION = "DRAFT"
config_stub.AGENTCORE_RUNTIME_ARN = ""
sys.modules["config"] = config_stub

# Run the supplied official tests unchanged, but only Task 2.
sys.argv = [os.path.join("tests", "test_agent.py"), "task2"]
runpy.run_path(sys.argv[0], run_name="__main__")
