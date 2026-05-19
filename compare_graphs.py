#!/usr/bin/env python3
"""Compare old (iltur_graphs/) vs new (graphs_enriched_v2/) graphs."""
import json, sys, os
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

old_dir = Path("iltur_graphs")
new_dir = Path("graphs_enriched_v2")

# Find matched cases
old_files = {f.stem for f in old_dir.glob("*.json") if f.name != "checkpoint.json"}
new_files = {f.stem for f in new_dir.glob("*.json") if f.name != "checkpoint.json"}
matched = sorted(old_files & new_files)
print(f"Matched cases: {len(matched)}")

# Collect stats
old_stats = {"edges": [], "chains": [], "tiers": Counter(), "clusters": [], "unlisted_nodes": [], "concept_ids": set()}
new_stats = {"edges": [], "chains": [], "tiers": Counter(), "clusters": [], "unlisted_nodes": [], "concept_ids": set()}

for case_id in matched:
    for stats, d_dir in [(old_stats, old_dir), (new_stats, new_dir)]:
        path = d_dir / f"{case_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)

        meta = g.get("_meta", {})
        cluster_summary = meta.get("cluster_summary") or {}

        stats["edges"].append(len(g.get("edges", [])))
        stats["chains"].append(len(g.get("reasoning_chains", [])))
        stats["tiers"][meta.get("quality_tier", "unknown")] += 1
        stats["clusters"].append(len(cluster_summary))

        # Count nodes in UNLISTED_* clusters
        unlisted_count = 0
        for cid, cdata in cluster_summary.items():
            if cid.startswith("UNLISTED_"):
                unlisted_count += sum(
                    len(cdata.get(k, []))
                    for k in ["facts", "concepts", "issues", "arguments", "holdings", "precedents"]
                )
            stats["concept_ids"].add(cid)

        stats["unlisted_nodes"].append(unlisted_count)

# Compute averages
def avg(lst):
    return sum(lst) / len(lst) if lst else 0

n = len(matched)
print(f"\n{'='*70}")
print(f"COMPARISON: old (iltur_graphs) vs new (graphs_enriched_v2)")
print(f"Matched cases: {n}")
print(f"{'='*70}\n")

print(f"{'Metric':<45} {'Old':>10} {'New':>10} {'Delta':>10}")
print(f"{'-'*75}")
print(f"{'Avg edges per graph':<45} {avg(old_stats['edges']):>10.1f} {avg(new_stats['edges']):>10.1f} {avg(new_stats['edges'])-avg(old_stats['edges']):>+10.1f}")
print(f"{'Avg reasoning chains per graph':<45} {avg(old_stats['chains']):>10.1f} {avg(new_stats['chains']):>10.1f} {avg(new_stats['chains'])-avg(old_stats['chains']):>+10.1f}")
print(f"{'Avg clusters per graph':<45} {avg(old_stats['clusters']):>10.1f} {avg(new_stats['clusters']):>10.1f} {avg(new_stats['clusters'])-avg(old_stats['clusters']):>+10.1f}")
print(f"{'Avg UNLISTED_* nodes per graph':<45} {avg(old_stats['unlisted_nodes']):>10.1f} {avg(new_stats['unlisted_nodes']):>10.1f} {avg(new_stats['unlisted_nodes'])-avg(old_stats['unlisted_nodes']):>+10.1f}")
print(f"{'Total unique ontology concept IDs':<45} {len(old_stats['concept_ids']):>10} {len(new_stats['concept_ids']):>10} {len(new_stats['concept_ids'])-len(old_stats['concept_ids']):>+10}")

print(f"\n{'Quality Tier Distribution':<45} {'Old':>10} {'New':>10} {'Delta':>10}")
print(f"{'-'*75}")
all_tiers = ["gold", "silver", "bronze", "reject"]
for tier in all_tiers:
    o = old_stats["tiers"].get(tier, 0)
    n_val = new_stats["tiers"].get(tier, 0)
    o_pct = o / n * 100
    n_pct = n_val / n * 100
    print(f"  {tier:<43} {f'{o} ({o_pct:.1f}%)':>10} {f'{n_val} ({n_pct:.1f}%)':>10} {n_val - o:>+10}")

# Bonus: top UNLISTED concepts in old vs new
old_unlisted = Counter()
new_unlisted = Counter()
for case_id in matched:
    for unlisted_counter, d_dir in [(old_unlisted, old_dir), (new_unlisted, new_dir)]:
        path = d_dir / f"{case_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)
        cs = (g.get("_meta", {}).get("cluster_summary") or {})
        for cid in cs:
            if cid.startswith("UNLISTED_"):
                unlisted_counter[cid] += 1

print(f"\nTop 10 UNLISTED_* clusters (by case frequency):")
print(f"  {'Old':<50} {'New'}")
print(f"  {'-'*90}")
old_top = old_unlisted.most_common(10)
new_top = new_unlisted.most_common(10)
for i in range(10):
    o_str = f"{old_top[i][0]} ({old_top[i][1]})" if i < len(old_top) else "-"
    n_str = f"{new_top[i][0]} ({new_top[i][1]})" if i < len(new_top) else "-"
    print(f"  {o_str:<50} {n_str}")
