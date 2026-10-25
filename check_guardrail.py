"""Read the guardrail back from AWS and print its key settings (read-only)."""
import boto3

REGION = 'us-east-1'
GUARDRAIL_ID = 'uk9dcv0tx0tj'
VERSION = '1'

client = boto3.client('bedrock', region_name=REGION)

try:
    g = client.get_guardrail(guardrailIdentifier=GUARDRAIL_ID, guardrailVersion=VERSION)
except Exception as e:
    print("Could not read the guardrail:", e)
    raise SystemExit(1)

print("Name / version:", g['name'], "/", g['version'])
print("Status:", g['status'])

print("\nContent filters:")
for f in g.get('contentPolicy', {}).get('filters', []):
    print(f"  {f['type']:<11} input={f['inputStrength']}  output={f['outputStrength']}")

print("\nPII entities:")
for p in g.get('sensitiveInformationPolicy', {}).get('piiEntities', []):
    print(f"  {p['type']:<28} {p['action']}")

print("\nDenied topics:")
topic_policy = g.get('topicPolicy', {})
for t in topic_policy.get('topics', []):
    print(f"  {t['name']} ({t['type']})")
print("  Topic tier:", topic_policy.get('tier', {}).get('tierName'))

print("\nWord policy (managed lists):")
for w in g.get('wordPolicy', {}).get('managedWordLists', []):
    print(" ", w['type'])

print("\nCross-region profile:", g.get('crossRegionDetails', {}).get('guardrailProfileId'))
print("\nBlocked input message :", g.get('blockedInputMessaging'))
print("Blocked output message:", g.get('blockedOutputsMessaging'))