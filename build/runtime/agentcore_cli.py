"""
agentcore_cli.py
================
Simple stand-in for the supplied AgentCore CLI helper (it was missing
from this workspace). It offers the same three functions the project
instructions use:

    configure_runtime(...)    - remember the runtime settings
    deploy()                  - zip the staged app, upload to S3, create the runtime
    deployed_runtime_arn()    - return the runtime ARN
"""

import io
import os
import sys
import time
import zipfile

import boto3

# Make config.py importable (it lives in the project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_settings = {}


def configure_runtime(name, source_dir, entrypoint='main.py', env_vars=None,
                      network_mode='PUBLIC', protocol='HTTP',
                      execution_role_arn=None):
    """Save the runtime settings. Nothing is created in AWS yet."""
    _settings.clear()
    _settings.update({
        'name': name,
        'source_dir': source_dir,
        'entrypoint': entrypoint,
        'env_vars': env_vars or {},
        'network_mode': network_mode,
        'protocol': protocol,
        'role_arn': execution_role_arn,
    })
    print(f"  Runtime configured: {name} ({network_mode}, {protocol})")


def deploy():
    """Zip the staged folder, upload it to S3, and create the runtime."""
    if not _settings:
        raise RuntimeError("Call configure_runtime() before deploy().")

    s3   = boto3.client('s3', region_name=config.AWS_REGION)
    ctrl = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)

    # 1) Zip everything in the staged folder
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(_settings['source_dir']):
            for fname in files:
                full = os.path.join(folder, fname)
                zf.write(full, os.path.relpath(full, _settings['source_dir']))

    # 2) Upload the zip to S3
    key = f"agentcore-artifacts/{_settings['name']}/deployment.zip"
    s3.put_object(Bucket=config.POLICY_BUCKET, Key=key,
                  Body=buffer.getvalue(), ContentType='application/zip')
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{key}")

    # 3) Create the runtime
    response = ctrl.create_agent_runtime(
        agentRuntimeName=_settings['name'],
        description=f"{config.PROJECT_NAME} orchestration runtime",
        roleArn=_settings['role_arn'],
        networkConfiguration={'networkMode': _settings['network_mode']},
        protocolConfiguration={'serverProtocol': _settings['protocol']},
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {'s3': {'bucket': config.POLICY_BUCKET, 'prefix': key}},
                'runtime': 'PYTHON_3_12',
                'entryPoint': [_settings['entrypoint']],
            }
        },
        environmentVariables=_settings['env_vars'],
    )
    _settings['arn'] = response.get('agentRuntimeArn', response.get('arn', ''))
    runtime_id = response['agentRuntimeId']

    # 4) Wait until the runtime is READY
    print("  Waiting for runtime to become READY", end='', flush=True)
    for _ in range(60):   # up to ~5 minutes
        status = ctrl.get_agent_runtime(agentRuntimeId=runtime_id)['status']
        if status == 'READY':
            print(' ready.')
            return
        if 'FAILED' in status:
            print(' failed.')
            raise RuntimeError(f"Runtime {runtime_id} entered status {status}")
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Runtime {runtime_id} not READY after 5 minutes")


def deployed_runtime_arn():
    """Return the ARN of the runtime created by deploy()."""
    return _settings.get('arn', '')