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