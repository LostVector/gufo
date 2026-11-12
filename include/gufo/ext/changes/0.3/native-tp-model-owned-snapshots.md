# Native tensor-parallel model-owned snapshots

The fork retains optional model-owned state, immutable snapshots, persistence
identities and serialization hooks for native TP runners. API0.3 runners keep
component-cache execution as the default and its resource/bind/reconciliation
methods remain required. `LegacyTextModelRunner` explicitly selects the
model-owned fallback and refuses component operations; it is used by the TP
instruction wrapper and its legacy-state fixtures. Snapshot capabilities and
persistence metadata are optional. Native parallel provider factories coexist
with upstream sampling, unsized-tensor and speculative-default metadata.

All extensions and hosts must be rebuilt. No existing binary ABI is promised.
