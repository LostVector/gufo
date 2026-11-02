from pathlib import Path
import json
r=Path('/home/rkuo/strix-dispatch-investigation');p=r/'tp-cache-retention';out=Path(__file__).resolve().parent;src=p/'upstream-ffn-hc-worktree';old=p/'upstream-batched-hc-q8-v2';assert not(out/'tp1-diagnostic-batch.cpp').exists()
s=(src/'src/models/qwen38_flash_next/kernels/rocm/batch.cpp').read_text().replace('#include <algorithm>','#include <algorithm>\n#include <atomic>',1)
prior=(p/'batched-ffn-hc-state-v1/diagnostic-batch.cpp').read_text();a=prior.index('namespace {\nstd::atomic<bool> ffn_hc_probe_enabled');b=prior.index('namespace {\n\nconstexpr',a);helper=prior[a:b]
needle='namespace gufo::models::qwen38_flash_next::rocm {\n';assert s.count(needle)==1;s=s.replace(needle,needle+helper,1)
needle='      if (rows != 0 && FusedMixer(l.hc_ffn, rows) &&';assert s.count(needle)==1;s=s.replace(needle,'      if (ffn_hc_probe_enabled.load(std::memory_order_relaxed) &&\n          rows != 0 && FusedMixer(l.hc_ffn, rows) &&',1)
needle='''      } else {
        CombineBatch(base.res, l.hc_ffn.norm.f32(), rows);''';assert s.count(needle)==1;s=s.replace(needle,'        ffn_hc_probe_produced[rows].fetch_add(1, std::memory_order_relaxed);\n'+needle,1)
needle='    return HcMixBatch(model_->hc_head(), base.res, false, base.mixed, nullptr,';assert s.count(needle)==1;s=s.replace(needle,'    if (rows <= 64) ffn_hc_probe_heads[rows].fetch_add(1, std::memory_order_relaxed);\n'+needle,1);(out/'tp1-diagnostic-batch.cpp').write_text(s)
s=(src/'src/models/qwen38_flash_next/engine.cpp').read_text().replace('#include <algorithm>','#include <algorithm>\n#include <atomic>',1);needle='namespace gufo::models::qwen38_flash_next {\n';assert s.count(needle)==1;s=s.replace(needle,needle+'namespace { std::atomic<unsigned> ffn_hc_probe_plan{0}; }\nvoid FfnHcProbePlan(unsigned n) { ffn_hc_probe_plan.store(n, std::memory_order_relaxed); }\n',1)
needle='''    batch_drafts =
        policy.Choose(std::span(rows).first(requests.size()), batch_context);''';assert s.count(needle)==1;s=s.replace(needle,needle+'\n    batch_drafts = ffn_hc_probe_plan.load(std::memory_order_relaxed);',1);(out/'tp1-diagnostic-engine.cpp').write_text(s)
s=(old/'tp1-probe.cpp').read_text().replace('HcQ8Probe','FfnHcProbe').replace('fused_q8','fused_ffn_hc').replace('actual_q8_enabled_batches','actual_ffn_hc_enabled_batches').replace('guarded Q8 producer activation','guarded FFN HC route activation')
s=s.replace('''            // Canonical API uses its production proposal policy; unlike the
            // fork it exposes no per-request proposal ceiling. Record the real
            // packed verification width rather than inventing an equivalent.''','''            // The test-only engine hook supplies equal predeclared greedy
            // plans (1/0); sampled planning is unchanged. Record actual packed
            // rows, rather than assuming the fork's per-request caps.''')
a=s.index('            Need(rows > 0 && (!enabled ? gc == 0 : gc > 0)');b=s.index('            gathers +=',a);s=s[:a]+'''            const bool eligible = enabled && rows > 0 && rows <= 8;
            Need(rows > 0 && (eligible ?
                     (q::rocm::FfnHcProbeProduced(rows) > 0 &&
                      gc == q::rocm::FfnHcProbeProduced(rows)) : gc == 0),
                 "actual batch shape and exclusive FFN HC route/fallback activation");
'''+s[b:];(out/'tp1-probe.cpp').write_text(s)
s=(old/'analyze.py').read_text().replace('fused_q8','fused_ffn_hc').replace('actual_q8_enabled_batches','actual_ffn_hc_enabled_batches').replace('producer-off/on','FFNHcRoute-off/on').replace("retained_prior_failure='../upstream-batched-hc-q8-v1/tp1-result.json'","scope_prior_canonical_Q8_failure='Separate olderQ8 experiment; not a failureof thisHCport or pooled evidence.'");(out/'analyze.py').write_text(s)
design={'canonical_base':'8ab834a5902bf19341750daecf27516a57840dd1','scope':'Freshcanonicalonly ABI/headers/GPUtest objects. CurrentHCFusiontargetFFNroute disabledmatchesoriginalstock8ab; enablednewport. Existinghelperpreserves104records/32complete-statepairs/full248320finite logits/RNG/tokens/positions/work. Greedyenginehook overridesafterChoose withpredeclared1/0plans forbotharms; sampledpolicyunchanged. Not productiongreedycontroller replay, timing or teacherquality. Branchcounter isnot independentGPUkerneltrace. Fourallocatedsessions/twoactive/context65536/actualshort32768; noRDMA/profiler/adminsampler/tablewarm/systemchange.','expected_records':104,'expected_enabled_cycles':44,'expected_full_state_pairs':32,'activation_assertion':'Exclusive FFNbranch counters at actualtargetrows; fallbackrequiresallcounterzero. Stronger shape-local control than inheritedglobalQ8counter; no numeric/state/output assertions relaxed.'};(out/'tp1-design.json').write_text(json.dumps(design,indent=2)+'\n')
print('Prepared canonical fixed-history FFN HC state probe')
