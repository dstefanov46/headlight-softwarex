# Headlight policy

Dry-run validation can exercise candidate enumeration, objective, and safety
logic; measured policy outcomes require completed evaluation runs under the
frozen research-v4 protocol.

Policy-v2 lives in `headlight.policy_v2`; it does not alter the research-v3
`ForecastPlacementPolicy`, Threshold controller, LSTM-P controller, or their
registrations. It registers `headlight-v2-full` with the learned
action-conditioned risk source and `headlight-v2-no-action` with the calibrated
capacity-surrogate source. Both registrations use the same enumeration,
objective, safety, readiness, scale-in, serialization, and logging paths.

## Candidate and objective contract

For each scalable service, the policy enumerates the Cartesian product of every
inclusive per-cluster minimum/maximum range in the hashed service profile. This
is equivalent to enumerating every feasible total from the summed minimum to
the summed maximum and every valid complete placement for each total. Candidate
IDs, iteration, and ties use the frozen profile cluster order.

Each candidate carries projected ready replicas, pending replicas, retained
pending replicas, ready fraction, and retained-pending age. Existing Ready
replicas are capped by the candidate placement; every remaining candidate
replica is treated as pending. Existing pending age is retained only where a
currently pending replica remains relevant. New replica deltas begin with age
zero.

Normal candidate selection minimizes:

`resource + lambda * calibrated_SLA_risk + gamma * churn`

Resource is normalized by the profile-wide maximum weighted allocation. Churn
is the normalized L1 distance from the current complete placement, so both
replica changes and cross-cluster movement are charged. The model-output adapter
requires an explicit calibrated marker, verifies that the output source matches
the controller registration, requires endpoint and horizon identities to match
the bundle, and validates the exact `[batch,candidate,endpoint,horizon]` shape
for both outputs. It then conservatively takes the maximum end-to-end p95 and
SLA probability over all required endpoints and horizons. Missing candidate
predictions cause a hold rather than a partial candidate comparison.

The deterministic tie order is objective, churn, total replicas, complete
placement, then candidate ID. Exhaustive enumeration is guarded before material
allocation: the current two-cluster profiles produce 48 candidates per scalable
service, or 1,968 candidates for all 41 scalable services in `full.json`. The
per-service limit is configurable and fails clearly instead of truncating the
candidate space.

## Safety and serialization

Fresh CPU utilization, queue depth, end-to-end p95 latency, and error rate can
each trigger reactive scale-out. Stale or missing values cannot trigger the
guard. The guard is evaluated before and bypasses model scoring, using
conservative deterministic fallback scores so inference cannot block an
emergency. Emergency scale-out constrains selection to the configured larger
total and to placements that do not reduce any cluster, may affect multiple
services in an interval, and is marked
`action_origin=reactive_emergency` and `out_of_distribution=true`.

Without an emergency, at most one model-selected service action is retained per
interval. Other otherwise-valid predictive actions become explicitly reasoned
holds, allowing re-observation/readiness handling before another predictive
action. Any emergency decision suppresses predictive actions for the interval,
including an emergency already at maximum capacity. Pending replicas block
normal predictive action. Reactive scale-out is also cluster-local with respect
to readiness: an unready desired replica in the selected hot/receiving cluster
holds another emergency scale-out into that same cluster, while pending
replicas in the other cluster do not block it. The hold lasts for 240 seconds
from the runtime's first observation of pending capacity; after that, one
additional emergency scale-out may pass per further 240-second age bucket.
The reason records `pending-readiness`, `pending-timeout`, or
`pending-timeout-cooldown` with the target cluster. Pending age resets when the
cluster has no unready desired replicas; a controller restart conservatively
starts the wait again. This operational safeguard is a separate policy-code
revision, not a model or bundle change. Scale-in additionally
requires every configured scale-in telemetry field to be fresh and healthy, a
safe calibrated candidate, zero pending replicas, the configured healthy-window
count, and cooldown completion. The live adapter configures CPU utilization,
latency, and error rate because it has no queue-depth source; queue depth remains
explicitly missing and can still be required by adapters that actually collect
it. Scale-in is not hard-pinned to a replica count derived from the demand
forecast, so safely scored surplus replicas can be released.

Shadow execution records every service decision and exposes the would-actuate
set but never calls the actuator. Strict decision-record construction emits all
candidate predictions, normalized objective terms, weighted objectives,
readiness projections, selected candidate, deterministic reason, telemetry
masks, replica state, policy/schema versions, and caller-supplied action
timestamps. It never invents lifecycle timestamps.

## Reported evaluation policy

The frozen evaluation objective is `0.25 * resource + violation_risk +
0.10 * churn + 0.50 * failure_probability`. Development seeds 711–713 were used
for objective selection. Model weights and policy inputs were frozen before
evaluation seeds 101–105.

Predictive total replica changes are bounded to one. Scale-in requires three
fresh healthy windows, safe predictions, no pending replicas and at least 60 s
cooldown. At most one service change is submitted per interval. The reported
evaluation disables reactive emergency overrides. Inference failure causes a
hold and submission failure is recorded. No live cluster admission is performed.
