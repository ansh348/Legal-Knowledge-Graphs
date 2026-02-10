#!/usr/bin/env python3
"""
tally_commons.py

Find cases present in both structured-nongraph-cases/ and iltur_graphs/.
Saves commons.json with the list + stats.

Usage:
    python tally_commons.py
"""

import json
from pathlib import Path
from datetime import datetime

DIR_NONGRAPH = Path("structured-nongraph-cases")
DIR_GRAPH = Path("iltur_graphs")
OUTPUT = Path("commons.json")


def get_case_ids(directory: Path) -> set[str]:
    """Get all case IDs (filenames without .json) from a directory."""
    if not directory.exists():
        print(f"⚠ Directory not found: {directory}")
        return set()
    return {
        f.stem for f in directory.glob("*.json")
        if f.name != "checkpoint.json"
    }


def main():
    print("=" * 60)
    print("CASE TALLY: structured-nongraph-cases vs iltur_graphs")
    print("=" * 60)

    nongraph_ids = get_case_ids(DIR_NONGRAPH)
    graph_ids = get_case_ids(DIR_GRAPH)

    common = sorted(nongraph_ids & graph_ids)
    only_nongraph = sorted(nongraph_ids - graph_ids)
    only_graph = sorted(graph_ids - nongraph_ids)

    print(f"\nNon-graph cases:  {len(nongraph_ids)}")
    print(f"Graph cases:      {len(graph_ids)}")
    print(f"Common (both):    {len(common)}")
    print(f"Only non-graph:   {len(only_nongraph)}")
    print(f"Only graph:       {len(only_graph)}")

    result = {
        "generated_at": datetime.now().isoformat(),
        "counts": {
            "nongraph_total": len(nongraph_ids),
            "graph_total": len(graph_ids),
            "common": len(common),
            "only_nongraph": len(only_nongraph),
            "only_graph": len(only_graph),
        },
        "common_case_ids": common,
        "only_nongraph": only_nongraph,
        "only_graph": only_graph,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Saved to {OUTPUT}")


if __name__ == "__main__":
    main()