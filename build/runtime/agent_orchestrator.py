"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results
from agent_observability import apply_observability_config

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """Build an Inventory Agent that retrieves order and customer facts."""
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=0.1)
    system_prompt = (
        "You are NovaMart's Inventory Agent. Retrieve facts from the available tools; "
        "do not decide refund eligibility or invent missing data. Clearly report when "
        "an order or customer cannot be found."
    )

    @tool
    def check_order_status(order_id: str, customer_id: str) -> dict:
        """Look up an order and verify it belongs to the specified customer."""
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.get_item(Key={"customer_id": customer_id, "order_id": order_id})
        order = response.get("Item")
        if not order:
            return {"found": False, "order_id": order_id, "message": "Order not found."}
        if str(order.get("customer_id", "")) != str(customer_id):
            return {"found": False, "order_id": order_id,
                    "message": "Order was not found for this customer."}
        return {"found": True, "order": order}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """Retrieve a customer's tier and account details from DynamoDB."""
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        response = table.get_item(Key={"customer_id": customer_id})
        customer = response.get("Item")
        if not customer:
            return {"found": False, "customer_id": customer_id,
                    "message": "Customer not found."}
        return {"found": True, "customer": customer,
                "tier": customer.get("tier", "Standard")}

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """List orders belonging to a customer, including status, date, and amount."""
        from boto3.dynamodb.conditions import Attr
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.scan(FilterExpression=Attr("customer_id").eq(customer_id))
        orders = response.get("Items", [])
        while response.get("LastEvaluatedKey"):
            response = table.scan(
                FilterExpression=Attr("customer_id").eq(customer_id),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            orders.extend(response.get("Items", []))
        return {"customer_id": customer_id, "orders": orders, "count": len(orders)}

    return Agent(model=model, system_prompt=system_prompt,
                 tools=[check_order_status, get_customer_tier, list_customer_orders])


def build_refund_agent() -> Agent:
    """Build a Refund Agent that checks return eligibility and records return requests."""
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=0.1)
    system_prompt = (
        "You are NovaMart's Refund Agent. ALWAYS call get_inventory_context first and use "
        "the inventory findings in WorkflowState before making a decision. Verify order "
        "ownership and the order date; Standard customers have a 30-day return window and "
        "Premium customers have a 60-day window. Never claim a refund was initiated "
        "unless the tool confirms it. Explain missing or invalid information."
    )

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """Read inventory facts from the session's shared WorkflowState."""
        state = _read_workflow_state(session_id)
        if not state:
            return {"found": False, "session_id": session_id,
                    "inventory_agent": {}, "message": "Workflow session not found."}
        return {"found": True, "session_id": session_id,
                "inventory_agent": state.get("inventory_agent", {})}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """Validate order ownership and return-window eligibility, then record a return request."""
        from datetime import datetime, timezone
        from boto3.dynamodb.conditions import Attr
        orders_table = dynamodb.Table(config.ORDERS_TABLE)
        response = orders_table.get_item(Key={"customer_id": customer_id, "order_id": order_id})
        order = response.get("Item")
        if not order or str(order.get("customer_id", "")) != str(customer_id):
            return {"approved": False, "message": "Order not found for this customer."}

        customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)
        customer = customers_table.get_item(Key={"customer_id": customer_id}).get("Item", {})
        tier = str(customer.get("tier", "Standard")).lower()
        window_days = 60 if tier == "premium" else 30
        raw_date = order.get("order_date") or order.get("created_at")
        if not raw_date:
            return {
                "approved": False,
                "order_id": order_id,
                "return_window_days": window_days,
                "message": "Cannot verify return eligibility because the order date is missing.",
            }
        try:
            order_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            if order_date.tzinfo is None:
                order_date = order_date.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - order_date).days
        except (TypeError, ValueError):
            logger.warning("Could not parse order date for %s", order_id)
            return {
                "approved": False,
                "order_id": order_id,
                "return_window_days": window_days,
                "message": "Cannot verify return eligibility because the order date is invalid.",
            }
        if age_days < 0:
            return {
                "approved": False,
                "order_id": order_id,
                "return_window_days": window_days,
                "order_age_days": age_days,
                "message": "Cannot verify return eligibility because the order date is in the future.",
            }
        if age_days > window_days:
            return {"approved": False, "order_id": order_id,
                    "customer_tier": customer.get("tier", "Standard"),
                    "return_window_days": window_days, "order_age_days": age_days,
                    "message": f"Order is outside the {window_days}-day return window."}

        reference = f"RET-{uuid.uuid4().hex[:10].upper()}"
        try:
            orders_table.update_item(
                Key={"customer_id": customer_id, "order_id": order_id},
                UpdateExpression="SET #status = :status, return_reference = :ref, return_reason = :reason",
                ConditionExpression="customer_id = :customer_id",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": "RETURN_REQUESTED", ":ref": reference,
                    ":reason": reason, ":customer_id": customer_id,
                },
            )
        except Exception as exc:
            logger.exception("Failed to record return request")
            return {"approved": False, "order_id": order_id,
                    "message": f"Could not record the return request: {exc}"}
        return {"approved": True, "order_id": order_id, "return_reference": reference,
                "return_window_days": window_days,
                "instructions": "Keep the return reference for your records.",
                "message": "Return request recorded successfully."}

    return Agent(model=model, system_prompt=system_prompt,
                 tools=[get_inventory_context, initiate_refund])


def build_policy_agent() -> Agent:
    """Build a coordinator that queries Returns, Shipping, and Warranty KBs in parallel."""
    retriever_model = BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=0.0)

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        return str(retrieve_from_knowledge_base(config.RETURNS_KB_ID, query))

    returns_retriever = Agent(
        model=retriever_model,
        system_prompt="Retrieve and report only relevant Returns Policy passages. Do not invent policy.",
        tools=[retrieve_returns_policy],
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        return str(retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query))

    shipping_retriever = Agent(
        model=retriever_model,
        system_prompt="Retrieve and report only relevant Shipping Policy passages. Do not invent policy.",
        tools=[retrieve_shipping_policy],
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        return str(retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query))

    warranty_retriever = Agent(
        model=retriever_model,
        system_prompt="Retrieve and report only relevant Warranty Policy passages. Do not invent policy.",
        tools=[retrieve_warranty_policy],
    )

    @tool
    def search_all_policies(query: str) -> str:
        """Query all three policy knowledge bases in parallel and combine their results."""
        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }
        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, question: str) -> tuple:
            try:
                result = agent(question)
                return domain, str(result)
            except Exception as exc:
                logger.exception("%s policy retrieval failed", domain)
                return domain, f"Retrieval failed: {exc}"

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain, result_text = future.result()
                results[domain] = result_text

        trace.kb_done(len(retrievers))
        for domain in ("Returns", "Shipping", "Warranty"):
            trace.kb_result(domain, results.get(domain, "[No results]"))
        return "\n\n".join(
            f"=== {domain} Policy ===\n{results.get(domain, '[No results]')}"
            for domain in ("Returns", "Shipping", "Warranty")
        )

    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID, temperature=0.2
    )
    coordinator_prompt = (
        "You are NovaMart's Policy Agent. ALWAYS call search_all_policies first for every "
        "policy question. Ground every answer in retrieved passages, distinguish the policy domains, "
        "and state clearly when the knowledge bases do not contain an answer."
    )
    return Agent(model=coordinator_model, system_prompt=coordinator_prompt,
                 tools=[search_all_policies])


def build_communication_agent() -> Agent:
    """Build the agent that composes the final customer-facing response."""
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=0.3)
    system_prompt = (
        "You are NovaMart's customer communication specialist. First read the full "
        "WorkflowState using get_full_workflow_context. Compose a concise, empathetic "
        "and accurate response based only on recorded findings. Do not invent actions "
        "or promise a refund unless the state confirms it."
    )

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """Read the complete shared WorkflowState for a session."""
        state = _read_workflow_state(session_id)
        if not state:
            return {"found": False, "session_id": session_id,
                    "message": "Workflow session not found."}
        return state

    return Agent(model=model, system_prompt=system_prompt,
                 tools=[get_full_workflow_context])


def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent,
) -> Agent:
    """Build the Orchestrator that routes work and manages shared WorkflowState."""
    model = BedrockModel(model_id=config.ORCHESTRATOR_MODEL_ID, temperature=0.0)
    system_prompt = (
        "You are NovaMart's Orchestrator. You NEVER write the final customer-facing answer. "
        "Follow these routing rules exactly: (1) call initialize_session first for every request. "
        "(2) Order status/history requests: call route_to_inventory_agent. "
        "(3) Return/refund requests: call route_to_inventory_agent first, then "
        "route_to_refund_agent. (4) Policy-meaning questions about return windows, shipping "
        "rates, or warranty terms: call route_to_policy_agent. (5) Account questions such as "
        "customer tier or Premium status: call route_to_inventory_agent, never PolicyAgent. "
        "(6) Math/calculation requests: skip InventoryAgent, PolicyAgent, and RefundAgent. "
        "For EVERY request, call route_to_communication_agent as your final tool call, "
        "after all other needed work; no tools may be called after it. Pass session_id and "
        "customer_id exactly as supplied."
    )

    def _worker_result(agent: Agent, prompt: str) -> str:
        return str(agent(prompt))

    def _write_result(session_id: str, field: str, value) -> dict:
        state = _read_workflow_state(session_id)
        if not state:
            raise ValueError(f"Workflow session {session_id} has not been initialized")
        return _update_workflow_state(
            session_id, {field: value}, expected_version=int(state["version"])
        )

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """Run InventoryAgent and save its findings to WorkflowState."""
        state = _read_workflow_state(session_id)
        if not state:
            raise ValueError("Call initialize_session before routing to inventory.")
        result = _worker_result(
            inventory_agent,
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\nCustomer request: {request}\n"
            "Use the appropriate order/customer tools and report factual findings.",
        )
        _update_workflow_state(
            session_id, {"inventory_agent": result}, expected_version=int(state["version"])
        )
        return result

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """Run PolicyAgent and save retrieved policy findings to WorkflowState."""
        state = _read_workflow_state(session_id)
        if not state:
            raise ValueError("Call initialize_session before routing to policy.")
        result = _worker_result(
            policy_agent, f"Session ID: {session_id}\nPolicy question: {request}"
        )
        _update_workflow_state(
            session_id, {"policy_agent": result}, expected_version=int(state["version"])
        )
        return result

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """Run RefundAgent after inventory facts are available and save its decision."""
        state = _read_workflow_state(session_id)
        if not state:
            raise ValueError("Call initialize_session before routing to refund.")
        if not state.get("inventory_agent"):
            return "Refund assessment not run: inventory facts are missing. Call route_to_inventory_agent first."
        result = _worker_result(
            refund_agent,
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\nRequest: {request}\n"
            f"Inventory findings: {state.get('inventory_agent')}\n"
            f"Policy findings: {state.get('policy_agent', 'Not retrieved')}\n"
            "Use get_inventory_context and initiate_refund only when the customer and order are clear.",
        )
        _update_workflow_state(
            session_id, {"refund_agent": result}, expected_version=int(state["version"])
        )
        return result

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """Run CommunicationAgent last and save its final draft to WorkflowState."""
        state = _read_workflow_state(session_id)
        if not state:
            raise ValueError("Call initialize_session before routing to communication.")
        result = _worker_result(
            communication_agent,
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
            f"Original request: {original_request}\nWorkflowState: {state}",
        )
        _update_workflow_state(
            session_id, {"communication_agent": result}, expected_version=int(state["version"])
        )
        return result

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """Create the shared WorkflowState record at the start of a customer session."""
        existing = _read_workflow_state(session_id)
        if existing:
            if str(existing.get("customer_id")) != str(customer_id):
                raise ValueError("Session ID is already associated with a different customer.")
            return f"Session {session_id} already initialized (version {existing['version']})."
        try:
            state = _create_workflow_state(session_id, customer_id)
        except Exception:
            # A concurrent initializer may have created the record after our first read.
            state = _read_workflow_state(session_id)
            if not state or str(state.get("customer_id")) != str(customer_id):
                raise
        return f"Session {session_id} initialized for customer {customer_id} (version {state['version']})."

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Small helper: publish a numbered version (so we never use DRAFT)
    def _publish_version(guardrail_id: str) -> str:
        # Wait until the guardrail is READY before versioning
        for _ in range(30):
            status = bedrock_client.get_guardrail(guardrailIdentifier=guardrail_id)['status']
            if status == 'READY':
                break
            time.sleep(2)
        resp = bedrock_client.create_guardrail_version(
            guardrailIdentifier=guardrail_id,
            description='NovaMart production version',
        )
        return resp['version']

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            # Workaround: if only DRAFT exists, publish a numbered version
            if guardrail_version == 'DRAFT':
                guardrail_version = _publish_version(guardrail_id)
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    # ── Create the guardrail ──────────────────────────────
    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description='NovaMart safety guardrail',
        # 1) Harmful content filters
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ]
        },
        # 2) PII: block cards + SSNs, anonymize emails + phones
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL',                    'action': 'ANONYMIZE'},
                {'type': 'PHONE',                    'action': 'ANONYMIZE'},
            ]
        },
        # 3) Off-topic subjects
        topicPolicyConfig={
            'topicsConfig': [
                {
                    'name': 'competitor_products',
                    'definition': 'Questions asking to compare or recommend products or services from competing retailers.',
                    'examples': [
                        'Is Amazon cheaper than NovaMart?',
                        'Which competitor has better headphones?',
                    ],
                    'type': 'DENY',
                },
                {
                    'name': 'pricing_negotiations',
                    'definition': 'Haggling or requests to change an advertised price. Simple math with a stated price or discount is NOT included.',
                    'examples': [
                        'Can you give me a better price on this laptop?',
                        'I will pay half, deal?',
                    ],
                    'type': 'DENY',
                },
                {
                    'name': 'legal_threats',
                    'definition': 'Threats of lawsuits, lawyers, or legal action against the company.',
                    'examples': [
                        'I will sue NovaMart if you do not refund me.',
                        'My lawyer will contact you.',
                    ],
                    'type': 'DENY',
                },
            ],
            'tierConfig': {'tierName': 'STANDARD'},
        },
        # 4) Profanity filter
        wordPolicyConfig={
            'managedWordListsConfig': [{'type': 'PROFANITY'}]
        },
        # 5) Friendly messages
        blockedInputMessaging="Sorry, I can't help with that request. Please ask about orders, returns, shipping, or warranty.",
        blockedOutputsMessaging="Sorry, I can't share that response. Please ask about orders, returns, shipping, or warranty.",
        # Needed for the STANDARD topic tier
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
    )
    guardrail_id = response['guardrailId']

    # ── Promote from DRAFT to a numbered version ──────────
    guardrail_version = _publish_version(guardrail_id)
    print(f"Guardrail created: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    Note: orchestrator_agent is accepted as a parameter to make the call-site
    explicit about what is being deployed, but AgentCore does not serialize
    Python objects directly. The app files are staged in build/runtime/ and
    deployed through the agentcore_cli helper.

    Returns:
        The AgentCore Runtime ARN
    """
    import shutil
    import agentcore_cli

    runtime_name = f"{config.PROJECT_NAME}-runtime".replace('-', '_')

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    # ── Stage the app in build/runtime/ ───────────────────
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    stage_dir = os.path.join(project_root, 'build', 'runtime')

    shutil.rmtree(stage_dir, ignore_errors=True)
    os.makedirs(stage_dir)
    for fname in os.listdir(src_dir):
        if fname.endswith('.py'):
            shutil.copy(os.path.join(src_dir, fname), stage_dir)
    shutil.copy(os.path.join(project_root, 'config.py'), stage_dir)
    with open(os.path.join(stage_dir, 'main.py'), 'w') as f:
        f.write('# NovaMart AgentCore Runtime entry point\n')
    print(f"  Staged app in {stage_dir}")

    # ── The 8 runtime environment variables ───────────────
    env_vars = {
        'AWS_REGION':        str(config.AWS_REGION),
        'PROJECT_NAME':      str(config.PROJECT_NAME),
        'RETURNS_KB_ID':     str(config.RETURNS_KB_ID or 'NOT_SET'),
        'SHIPPING_KB_ID':    str(config.SHIPPING_KB_ID or 'NOT_SET'),
        'WARRANTY_KB_ID':    str(config.WARRANTY_KB_ID or 'NOT_SET'),
        'AGENT_LOG_GROUP':   str(config.AGENT_LOG_GROUP),
        'GUARDRAIL_ID':      str(guardrail_id),
        'GUARDRAIL_VERSION': str(guardrail_version),
    }

    # ── Configure, deploy, and read the ARN via the CLI helper ──
    agentcore_cli.configure_runtime(
        name=runtime_name,
        source_dir=stage_dir,
        entrypoint='main.py',
        env_vars=env_vars,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )
    agentcore_cli.deploy()
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    print(f"AgentCore Runtime deployed: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # ── Create AgentCore Memory ───────────────────────────
    response = agentcore_control.create_memory(
        name=memory_name,
        description=f"{config.PROJECT_NAME} conversational memory",
        eventExpiryDuration=7,   # days (a number, not "7d")
        memoryStrategies=[
            {
                'summaryMemoryStrategy': {
                    'name': 'SessionSummarizer',
                    'namespaces': ['/summaries/{actorId}/{sessionId}'],
                }
            }
        ],
        clientToken=str(uuid.uuid4()),
    )
    memory = response['memory']
    memory_id  = memory['id']
    memory_arn = memory['arn']

    # ── Wait until the memory is ACTIVE ───────────────────
    print("  Waiting for Memory to become ACTIVE", end='', flush=True)
    for _ in range(60):   # up to ~5 minutes
        status = agentcore_control.get_memory(memoryId=memory_id)['memory']['status']
        if status == 'ACTIVE':
            print(' ready.')
            print(f"  Memory ARN: {memory_arn}")
            return memory_arn
        if status == 'FAILED':
            print(' failed.')
            raise RuntimeError(f"Memory {memory_id} failed to create")
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Memory {memory_id} not ACTIVE after 5 minutes")


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    logging_config = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel': 'INFO',
            'enabled': True,
        },
        'xRayConfig': {
            'enabled': True,
            'samplingRate': 1.0,
        },
    }

    apply_observability_config(runtime_arn, logging_config)


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
