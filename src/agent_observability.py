"""
agent_observability.py
Simple helper that turns on CloudWatch logs and X-Ray tracing.
"""
import json
import boto3
import config


def apply_observability_config(runtime_arn: str, logging_config: dict) -> None:
    region = config.AWS_REGION
    logs = boto3.client("logs", region_name=region)
    xray = boto3.client("xray", region_name=region)

    cw = logging_config.get("cloudWatchConfig", {})
    xr = logging_config.get("xRayConfig", {})

    # 1) Create the CloudWatch log group
    if cw.get("enabled"):
        try:
            logs.create_log_group(logGroupName=cw["logGroupName"])
            print(f"  Created log group: {cw['logGroupName']}")
        except logs.exceptions.ResourceAlreadyExistsException:
            print(f"  Log group already exists: {cw['logGroupName']}")
        except Exception as e:
            print(f"  [Note] Log group step skipped: {e}")

    # 2) Turn on X-Ray Transaction Search
    if xr.get("enabled"):
        percent = float(xr.get("samplingRate", 1.0)) * 100
        try:
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Sid": "TransactionSearchXRayAccess",
                    "Effect": "Allow",
                    "Principal": {"Service": "xray.amazonaws.com"},
                    "Action": "logs:PutLogEvents",
                    "Resource": [
                        f"arn:aws:logs:{region}:{account_id}:log-group:aws/spans:*",
                        f"arn:aws:logs:{region}:{account_id}:log-group:/aws/application-signals/data:*",
                    ],
                }],
            }
            logs.put_resource_policy(
                policyName="TransactionSearchXRayAccess",
                policyDocument=json.dumps(policy),
            )
            xray.update_trace_segment_destination(Destination="CloudWatchLogs")
            xray.update_indexing_rule(
                Name="Default",
                Rule={"Probabilistic": {"DesiredSamplingPercentage": percent}},
            )
            print(f"  X-Ray Transaction Search on, sampling {percent:.0f}%")
        except Exception as e:
            print(f"  [Note] X-Ray step skipped: {e}")

    print(f"  Runtime: {runtime_arn}")


# ── Lets the control-plane client report the real logging/tracing state ──
def _read_real_config() -> dict:
    region = config.AWS_REGION
    logs = boto3.client("logs", region_name=region)
    xray = boto3.client("xray", region_name=region)

    group = config.AGENT_LOG_GROUP
    found = logs.describe_log_groups(logGroupNamePrefix=group).get("logGroups", [])
    cw_on = any(g["logGroupName"] == group for g in found)

    dest = xray.get_trace_segment_destination()
    xray_on = dest.get("Destination") == "CloudWatchLogs"

    return {
        "cloudWatchConfig": {"logGroupName": group, "logLevel": "INFO", "enabled": cw_on},
        "xRayConfig": {"enabled": xray_on, "samplingRate": 1.0},
    }


def _register_logging_getter() -> None:
    def _add(class_attributes, base_classes, **kwargs):
        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {"loggingConfiguration": _read_real_config()}

        class_attributes["get_agent_runtime_logging_configuration"] = (
            get_agent_runtime_logging_configuration
        )

    if boto3.DEFAULT_SESSION is None:
        boto3.setup_default_session()
    boto3.DEFAULT_SESSION._session.register(
        "creating-client-class.bedrock-agentcore-control", _add
    )


_register_logging_getter()


# ── Simple X-Ray recorder: one trace per request ──
import os
import json
import time
import threading
from contextlib import contextmanager

_req = {"tid": None, "root": None, "start": None, "nodes": []}
_req_lock = threading.Lock()


def start_request() -> str:
    """Call once at the start of each customer request."""
    now = time.time()
    with _req_lock:
        _req["tid"] = "1-%08x-%s" % (int(now), os.urandom(12).hex())
        _req["root"] = os.urandom(8).hex()
        _req["start"] = now
        _req["nodes"] = []
    return _req["tid"]


@contextmanager
def traced_node(name: str):
    """Wrap a piece of work so it shows up as a box in the Trace Map."""
    began = time.time()
    try:
        yield
    finally:
        with _req_lock:
            if _req["tid"]:
                _req["nodes"].append((name, began, time.time()))


def finish_request() -> str:
    """Send the trace to X-Ray and print its ID. Never crashes the app."""
    with _req_lock:
        tid, root, start = _req["tid"], _req["root"], _req["start"]
        nodes = list(_req["nodes"])
        _req["tid"] = None
    if not tid:
        return ""
    try:
        docs = [{"trace_id": tid, "id": root, "name": "NovaMart-Orchestrator",
                 "start_time": start, "end_time": time.time()}]
        for name, began, ended in nodes:
            docs.append({"trace_id": tid, "id": os.urandom(8).hex(),
                         "name": name, "type": "subsegment", "namespace": "remote",
                         "parent_id": root, "start_time": began, "end_time": ended})
        boto3.client("xray", region_name=config.AWS_REGION).put_trace_segments(
            TraceSegmentDocuments=[json.dumps(d) for d in docs])
        print(f"  X-Ray trace ID: {tid}")
    except Exception as e:
        print(f"  [Note] X-Ray trace not sent: {e}")
    return tid