# Actions Handler

Go service that batches placement decisions and patches ControlledReplicaPropagation
resources through Karmada. See the canonical [component documentation](../docs/components/actions-handler.md)
for the API, configuration, and deployment contract.

Run component tests with `just test`. The repository-wide dry-run checks
exercise contracts; measured artifacts come from the deployment and validation
workflow.
