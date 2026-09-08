"""Morning readout for the dex-full real-capture battery (EVAL_GUIDANCE.md §7.27).
Writes outputs/diagnostics/DEXFULL_RESULTS.md from whatever result files exist."""
import json, os, subprocess, glob
import pandas as pd
D = "/projects/gcaddeo/train_flow_conditioned/TRELLIS/outputs/diagnostics"
INF = "/projects/gcaddeo/inference/TRELLIS"
out = []
def P(s=""): out.append(s)
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def m(d, arm, k):
    try: return f"{d[arm][k]['mean']:.4f}"
    except Exception: return "—"

P("# dex-full REAL-CAPTURE multi-view battery — readout"); P()
ids = open(f"{D}/dexfull_scratch/all_job_ids.txt").read().strip().replace(":", ",")
try:
    P("## Job states"); P("```"); P(subprocess.run(["sacct", "-j", ids, "-o", "JobID%8,JobName%16,State%12,Elapsed,ExitCode", "-n", "-X"], capture_output=True, text=True).stdout.strip()); P("```"); P()
except Exception as e: P(f"sacct failed: {e}")

P("## In-grid harness metrics (ref camera grid vs posed GT; IoU / CD / F@0.02 / EMD), 439 benchmark groups")
P()
P("| model | pose | arm | n | IoU | CD | F@0.02 | EMD | contact_abs |"); P("|---|---|---|---|---|---|---|---|---|")
for model, tag in [("frozen wp4", "wp4fin"), ("A@165k plain", "a165k")]:
    for mode in ("gtpose", "posefree"):
        d = load(f"{D}/stream_{tag}_dexfull_full_{mode}.json")
        if not d: P(f"| {model} | {mode} | (streaming missing) | | | | | | |"); continue
        for arm in ("single", "ringbuffer_median", "stream_final", "stream_median"):
            P(f"| {model} | {mode} | {arm} | {d['meta']['num_groups']} | {m(d,arm,'occ_iou_gt')} | {m(d,arm,'cd')} | {m(d,arm,'f@0.02')} | {m(d,arm,'emd')} | {m(d,arm,'contact_abs')} |")
for model, tag in [("frozen wp4", "wp4fin"), ("teacher v2@52k", "teacher52k")]:
    for mode in ("gtpose", "posefree"):
        d = load(f"{D}/fusion_{tag}_dexfull_full_{mode}.json")
        if not d: P(f"| {model} | {mode} | (fusion missing) | | | | | | |"); continue
        for arm in ("single", "median_K2", "median_K4", "median_K8", "hybrid_K8"):
            P(f"| {model} | {mode} | fusion {arm} | {d['meta']['num_groups']} | {m(d,arm,'occ_iou_gt')} | {m(d,arm,'cd')} | {m(d,arm,'f@0.02')} | {m(d,arm,'emd')} | {m(d,arm,'contact_abs')} |")
P()
P("## Per-frame integration trajectory (IoU, streamed / no-prior twin)"); P()
for tag in ("wp4fin", "a165k"):
    for mode in ("gtpose", "posefree"):
        d = load(f"{D}/stream_{tag}_dexfull_full_{mode}.json")
        if not d: continue
        P(f"- {tag} {mode}: " + "  ".join(f"f{t}: {d[f'frame_{t}']['occ_iou_gt']['mean']:.3f}/{d[f'frame_{t}_noprior']['occ_iou_gt']['mean']:.3f}" for t in range(1, 9) if f"frame_{t}" in d))
        r = d["meta"].get("registration")
        if r: P(f"  registration: {r}")
P()
P("## Canonical ICP dex eval (CD² / NC / F@0.02, bbox[-1,1], ICP) on dumped meshes, frozen wp4"); P()
P("| pose | arm | n | CD² mean | CD² median | NC mean | F@0.02 |"); P("|---|---|---|---|---|---|---|")
def fs(d):
    f = d.get("Fscore", {})
    for k, v in f.items():
        if "0.02" in str(k): return f"{v['mean']:.4f}" if isinstance(v, dict) else f"{v:.4f}"
    return "—"
for mode in ("gtpose", "posefree"):
    for arm in ("gt", "single", "ringbuffer_median", "stream_final", "stream_median"):
        d = load(f"{INF}/summary_dexfull_wp4fin_{mode}_{arm}_icp.json")
        if not d: P(f"| {mode} | {arm} | (missing) | | | | |"); continue
        P(f"| {mode} | {arm} | {d.get('num_pairs_evaluated')} | {d['CD']['mean']:.4f} | {d['CD']['median']:.4f} | {d['NC']['mean']:.4f} | {fs(d)} |")
P()
P("## Single-image reference rows (canonical ICP), full n=989 and restricted to the 439 groups' ref camera (view 0)"); P()
man = load(f"{INF}/dex-full-groups/groups_manifest.json")
meta = pd.read_csv(f"{INF}/dex-full-groups/metadata.csv")
usable = set(meta[meta.cond_rendered].sha256)
ref_inst = {v["instances"][0] for g, v in man["groups"].items() if g in usable}
grasp_inst = {i for g, v in man["groups"].items() if g in usable for i in v["instances"]}
P("same-frame = old instance IS the group's ref camera (view 0); same-grasp = old instance is any camera of a usable group."); P()
P("| model | n full | CD² mean / median (full) | n same-frame | CD² mean / median | n same-grasp | CD² mean / median | NC | F@0.02 |"); P("|---|---|---|---|---|---|---|---|---|")
for name, tag in [("teacher v2@37k", "teacher_v2"), ("teacher v2@52k s8", "teacher_v2_52k_s8"), ("A@113k", "student_a113k_s8"), ("A@165k", "student_a165k_s8"), ("frozen wp4", "student_wp4fin_s8")]:
    p = f"{INF}/summary_{tag}_dex_total_total_icp.csv"
    if not os.path.exists(p): P(f"| {name} | (missing {tag}) | | | | | |"); continue
    c = pd.read_csv(p); c["inst"] = c["instance"].str.replace("__00__sample.ply", "", regex=False).str.replace("custom_object_", "custom_objects_", regex=False)
    s = c[c["inst"].isin(ref_inst)]; g = c[c["inst"].isin(grasp_inst)]
    f = f"{g['F@0.02'].mean():.4f}" if "F@0.02" in c.columns else "—"
    P(f"| {name} | {len(c)} | {c.CD.mean():.4f} / {c.CD.median():.4f} | {len(s)} | {s.CD.mean():.4f} / {s.CD.median():.4f} | {len(g)} | {g.CD.mean():.4f} / {g.CD.median():.4f} | {g.NC.mean():.4f} | {f} |")
P()
cp = load(f"{D}/dexfull_scratch/completeness.json")
P("## Completeness sweep (all 508,384 instances)"); P()
P("```"); P(json.dumps({k: v for k, v in (cp or {"status": "not finished"}).items() if k != "missing_examples"}, indent=1)); P("```")
open(f"{D}/DEXFULL_RESULTS.md", "w").write("\n".join(out)); print("\n".join(out))
