"""Instrument the actual normal-controller path; add no waits or plan overrides."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess

lab = Path(__file__).resolve().parent
root = lab.parents[1]
lock = (root / 'overnight/results/model-matrix.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (lab / 'build-protocol.json').exists()
helper = r'''
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
namespace {
struct StrixHcStageProbe {
  const char* unit;
  std::uint64_t id;
  bool enabled;
  std::chrono::steady_clock::time_point start;
  StrixHcStageProbe(bool active, const char* name) : unit(name), id(0), enabled(false), start(std::chrono::steady_clock::now()) {
    static std::atomic<std::uint64_t> next{0};
    if (active) { id = next.fetch_add(1, std::memory_order_relaxed); enabled = id < 16; }
    Mark("entry");
  }
  void Mark(const char* stage) const {
    if (!enabled) return;
    const auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "HC_STAGE unit=%s call=%llu stage=%s t_ns=%lld elapsed_us=%.3f\n", unit, (unsigned long long)id, stage,
      (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(),
      std::chrono::duration<double, std::micro>(now-start).count());
  }
  void Value(const char* key, std::uint64_t value) const {
    if (enabled) std::fprintf(stderr, "HC_VALUE unit=%s call=%llu key=%s value=%llu\n", unit, (unsigned long long)id, key, (unsigned long long)value);
  }
  ~StrixHcStageProbe() { Mark("leave"); }
};
}
'''


def replace_once(source, old, new):
    assert source.count(old) == 1, (old[:120], source.count(old))
    return source.replace(old, new)


env = dict(os.environ)
env['CPLUS_INCLUDE_PATH'] = str(root / 'dependencies/gufo-dev/usr/include/x86_64-linux-gnu')
env['LD_LIBRARY_PATH'] = str(root / 'dependencies/gufo-dev/usr/lib/x86_64-linux-gnu') + ':/home/rkuo/.local/opt/rocm-7.14.1/lib'
records = {}
for role, stem in [('baseline', 'upstream-ffn-hc-baseline'), ('candidate', 'upstream-ffn-hc')]:
    out = lab / role
    out.mkdir()
    src = root / 'tp-cache-retention' / (stem + '-worktree')
    build = root / 'tp-cache-retention' / (stem + '-release-build')
    engine = (src / 'src/models/qwen38_flash_next/engine.cpp').read_text()
    anchor = '  auto& exec = *requests.front().session->model_->executor_;\n  std::optional<std::uint32_t> batch_drafts;'
    engine = replace_once(engine, anchor, '  StrixHcStageProbe probe(true, "engine");\n  probe.Value("requests", requests.size());\n  probe.Value("position0", requests.front().session->Position());\n' + anchor)
    engine = replace_once(engine, '  const auto cycle_start = std::chrono::steady_clock::now();\n  std::vector<PendingDecode> pending(requests.size());', '  probe.Value("planned_drafts", batch_drafts.value_or(999));\n  probe.Mark("policy_done");\n  const auto cycle_start = std::chrono::steady_clock::now();\n  std::vector<PendingDecode> pending(requests.size());')
    engine = replace_once(engine, '  std::vector<AdvanceRequest> catchup;\n  std::vector<std::uint8_t> propose;', '  probe.Mark("prepare_decode_done");\n  std::vector<AdvanceRequest> catchup;\n  std::vector<std::uint8_t> propose;')
    engine = replace_once(engine, '  // Each round shares predictor projections across ready sessions.', '  probe.Mark("draft_catchup_done");\n  // Each round shares predictor projections across ready sessions.')
    engine = replace_once(engine, '  // In an all-greedy cohort, full copies can widen verification far beyond', '  probe.Mark("drafting_done");\n  // In an all-greedy cohort, full copies can widen verification far beyond')
    engine = replace_once(engine, '  if (!items.empty() && !exec.ForwardBatch(items, error_msg))\n    return false;\n  std::uint32_t offset = 0;', '  for (const auto& item : items) probe.Value("verification_rows", item.tokens.size());\n  probe.Mark("before_target_forward");\n  if (!items.empty() && !exec.ForwardBatch(items, error_msg))\n    return false;\n  probe.Mark("target_forward_done");\n  std::uint32_t offset = 0;')
    engine = replace_once(engine, '  const bool used_lookup =\n', '  probe.Mark("finish_decode_done");\n  const bool used_lookup =\n')
    (out / 'engine.cpp').write_text(helper + engine)

    whole_batch = (src / 'src/models/qwen38_flash_next/kernels/rocm/batch.cpp').read_text()
    forward_start = whole_batch.index('bool Executor::ForwardBatch(')
    forward_end = whole_batch.index('bool Executor::SelectBatchLogits(', forward_start)
    batch = whole_batch[forward_start:forward_end]
    anchor = 'bool Executor::ForwardBatch(std::span<const BatchItem> items,\n                            std::string* error) const {\n'
    batch = replace_once(batch, anchor, anchor + '  StrixHcStageProbe probe(true, "batch");\n')
    anchor = '  if (rows > options_.max_batch) {'
    batch = replace_once(batch, anchor, '  probe.Value("rows", rows);\n  probe.Value("requests", items.size());\n' + anchor)
    anchor = '  for (const auto& item : items) {\n    if (!item.session->Cancelled())\n      item.session->PreserveSnapshots'
    batch = replace_once(batch, anchor, '  probe.Mark("rollback_start");\n' + anchor)
    anchor = '  if (!AllocateBatch(error)) {\n'
    batch = replace_once(batch, anchor, '  probe.Mark("rollback_done");\n' + anchor)
    anchor = '  if (batch_controls_ == nullptr &&\n'
    batch = replace_once(batch, anchor, '  probe.Mark("allocate_batch_done");\n' + anchor)
    anchor = '  if (ple_pending_ && !WaitPle(error)) {'
    batch = replace_once(batch, anchor, '  probe.Mark("controls_done");\n' + anchor)
    anchor = '  FinishPrefetch();\n  batch_rows_ = 0;'
    batch = replace_once(batch, anchor, '  FinishPrefetch();\n  probe.Mark("prior_ple_prefetch_done");\n  batch_rows_ = 0;')
    anchor = '  const Scratch base = s_;\n  // Every exit restores'
    batch = replace_once(batch, anchor, '  probe.Mark("ngram_submitted");\n  const Scratch base = s_;\n  // Every exit restores')
    anchor = '    const auto& layers = model_->layers();\n    bool normed = false;'
    batch = replace_once(batch, anchor, '    probe.Mark("before_layers");\n' + anchor)
    anchor = '      if (c.IsPleLayer(il) && !WaitPle(error))\n        return false;'
    batch = replace_once(batch, anchor, '      if (c.IsPleLayer(il)) probe.Mark("before_ple_wait");\n' + anchor + '\n      if (c.IsPleLayer(il)) probe.Mark("after_ple_wait");')
    anchor = '  const auto status = hipStreamSynchronize(stream_);\n  UseScratch(base);'
    batch = replace_once(batch, anchor, '  probe.Mark("body_submitted");\n  const auto status = hipStreamSynchronize(stream_);\n  probe.Mark("stream_complete");\n  UseScratch(base);')
    (out / 'batch.cpp').write_text(helper + whole_batch[:forward_start] + batch + whole_batch[forward_end:])
    entries = json.loads((build / 'compile_commands.json').read_text())
    commands = []
    for suffix, filename in [('/qwen38_flash_next/engine.cpp', 'engine.cpp'), ('/kernels/rocm/batch.cpp', 'batch.cpp')]:
        entry = next(e for e in entries if e['file'].endswith(suffix))
        command = shlex.split(entry['command'])
        command[command.index('-o') + 1] = str(out / (filename + '.o'))
        command[-1] = str(out / filename)
        assert '-DNDEBUG' in command and not any('TP2_RDMA' in x for x in command)
        commands.append(command)
    archive = build / 'extensions/build/qwen38_flash_next/build/libgufo_qwen38_flash_next_hip.a'
    copy = out / 'diagnostic-hip.a'
    shutil.copy2(archive, copy)
    commands.append(['ar', 'r', str(copy), str(out / 'engine.cpp.o'), str(out / 'batch.cpp.o')])
    link = shlex.split(subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'gufo'], text=True).splitlines()[-1])
    assert link[:2] == [':', '&&'] and link[-2:] == ['&&', ':']
    link = link[2:-2]
    relative = str(archive.relative_to(build))
    assert link.count(relative) >= 1
    link = [str(copy) if x == relative else x for x in link]
    link[link.index('-o') + 1] = str(out / 'gufo')
    index = link.index('-Xlinker')
    assert link[index + 1].startswith('--dependency-file=')
    del link[index:index + 2]
    commands.append(link)
    with (out / 'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, cwd=build, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    records[role] = {'source': subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip(),
                     'commands': commands, 'cwd': str(build),
                     'source_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ('engine.cpp', 'batch.cpp')},
                     'binary_sha256': hashlib.sha256((out / 'gufo').read_bytes()).hexdigest()}
(lab / 'build-protocol.json').write_text(json.dumps({'records': records, 'environment': {key: env[key] for key in ('CPLUS_INCLUDE_PATH', 'LD_LIBRARY_PATH')}, 'scope': 'Diagnostic only: first16 actual normal-controller C2 cycles and ForwardBatch stages, no forced plans, no new GPU waits/math changes. Logging can alter adaptive timing. Not a release performance artifact.'}, indent=2) + '\n')
print('Built both diagnostic CLI executables; no production source modified.', flush=True)
