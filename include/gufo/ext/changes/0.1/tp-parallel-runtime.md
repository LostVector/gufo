# Host-only tensor-parallel text runtime

Text runners gain optional decode-plan agreement, prompt-context transfer and
state/snapshot reuse veto hooks for mirrored execution. Default implementations
preserve single-process behavior. `TextRunnerDecode` gains an optional per-member
draft count; `ChatRequest` retains the original response-format JSON and envelope
for peer constraint reconstruction. These are additive source contracts; rebuild
extensions as usual. An optional `TextProvider::create_parallel` factory returns a host-only
`TextParallelRuntime`: providers own GPU transport and runner creation; serving
keeps control messages, scheduling and verdicts. `TextParallelTransportOptions`
contains the existing bootstrap/device/RDMA settings without GPU types.
