"""
Fix script for iltur_graphs JSON data issues.
Addresses 6 categories of problems across 12 files.
"""
import json
import copy
from pathlib import Path

GRAPH_DIR = Path("iltur_graphs")


def load(name):
    fp = GRAPH_DIR / name
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save(name, data):
    fp = GRAPH_DIR / name
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {name}")


def remove_edge(data, edge_id):
    before = len(data["edges"])
    data["edges"] = [e for e in data["edges"] if e["id"] != edge_id]
    removed = before - len(data["edges"])
    if removed:
        print(f"    Removed edge {edge_id}")
    else:
        print(f"    WARNING: edge {edge_id} not found!")
    return removed


def find_holding(data, hid):
    for h in data.get("holdings", []):
        if h["id"] == hid:
            return h
    return None


def find_edge(data, eid):
    for e in data["edges"]:
        if e["id"] == eid:
            return e
    return None


def clean_validation_warnings(data, patterns):
    """Remove validation_warnings entries matching any of the given substrings."""
    meta = data.get("_meta", {})
    warnings = meta.get("validation_warnings", [])
    if not warnings:
        return
    original = len(warnings)
    warnings = [w for w in warnings if not any(p in w for p in patterns)]
    meta["validation_warnings"] = warnings
    removed = original - len(warnings)
    if removed:
        print(f"    Cleaned {removed} validation warning(s)")


# =============================================================================
# 1. Self-loop fix — 1958_88.json
# =============================================================================
def fix_1958_88():
    print("Fixing 1958_88.json (self-loop)...")
    data = load("1958_88.json")
    edge = find_edge(data, "e_39da50b9_15")
    assert edge is not None, "Edge e_39da50b9_15 not found"
    assert edge["source"] == "a12" and edge["target"] == "a12", "Unexpected edge state"
    edge["target"] = "a15"
    print(f"    Changed edge target from a12 to a15")
    clean_validation_warnings(data, ["self-loop", "e_39da50b9_15"])
    save("1958_88.json", data)


# =============================================================================
# 2. Empty citation — 1971_82.json
# =============================================================================
def fix_1971_82():
    print("Fixing 1971_82.json (empty citation)...")
    data = load("1971_82.json")
    for p in data.get("precedents", []):
        if p["id"] == "p15":
            assert p["citation"] is None, f"Unexpected citation: {p['citation']}"
            p["citation"] = "Unreported"
            print(f"    Set p15 citation to 'Unreported'")
            break
    else:
        raise ValueError("Precedent p15 not found")
    clean_validation_warnings(data, ["p15", "citation"])
    save("1971_82.json", data)


# =============================================================================
# 3. String "null" → JSON null — 1958_152.json, 1971_76.json
# =============================================================================
def fix_string_null(filename, holding_ids, edge_ids):
    print(f"Fixing {filename} (string 'null' -> null)...")
    data = load(filename)
    for hid in holding_ids:
        h = find_holding(data, hid)
        assert h is not None, f"Holding {hid} not found"
        assert h["resolves_issue"] == "null", f"Unexpected resolves_issue: {h['resolves_issue']}"
        h["resolves_issue"] = None
        print(f"    Set {hid}.resolves_issue to null")
    for eid in edge_ids:
        remove_edge(data, eid)
    clean_validation_warnings(data, ["resolves_null", "target 'null'", "target not found"])
    save(filename, data)


# =============================================================================
# 4. Non-existent issue IDs → null
# =============================================================================
def fix_nonexistent_issue(filename, holdings_map, edge_ids):
    """holdings_map: {holding_id: old_issue_id}"""
    print(f"Fixing {filename} (non-existent issue IDs)...")
    data = load(filename)
    # Verify the issue IDs don't exist
    issue_ids = {i["id"] for i in data.get("issues", [])}
    for hid, old_val in holdings_map.items():
        assert old_val not in issue_ids, f"Issue {old_val} actually exists!"
        h = find_holding(data, hid)
        assert h is not None, f"Holding {hid} not found"
        assert h["resolves_issue"] == old_val, f"Unexpected resolves_issue: {h['resolves_issue']}"
        h["resolves_issue"] = None
        print(f"    Set {hid}.resolves_issue from '{old_val}' to null")
    for eid in edge_ids:
        remove_edge(data, eid)
    clean_validation_warnings(data,
        [f"target '{v}'" for v in holdings_map.values()] +
        [eid for eid in edge_ids] +
        ["target not found"])
    save(filename, data)


# =============================================================================
# 5. Comma/semicolon-separated issue IDs
# =============================================================================
def fix_multi_issue(filename, holding_id, old_target, issue_ids):
    """Split a comma/semicolon-separated resolves_issue into proper edges."""
    print(f"Fixing {filename} (multi-issue '{old_target}')...")
    data = load(filename)

    # Fix holding: set to first issue
    h = find_holding(data, holding_id)
    assert h is not None, f"Holding {holding_id} not found"
    assert h["resolves_issue"] == old_target, f"Unexpected resolves_issue: {h['resolves_issue']}"
    h["resolves_issue"] = issue_ids[0]
    print(f"    Set {holding_id}.resolves_issue to '{issue_ids[0]}'")

    # Find and remove the bad edge
    bad_edge_id = f"e_{holding_id}_resolves_{old_target}"
    bad_edge = find_edge(data, bad_edge_id)
    assert bad_edge is not None, f"Bad edge {bad_edge_id} not found"

    # Create replacement edges
    new_edges = []
    for iid in issue_ids:
        new_edge = copy.deepcopy(bad_edge)
        new_edge["id"] = f"e_{holding_id}_resolves_{iid}"
        new_edge["target"] = iid
        new_edges.append(new_edge)
        print(f"    Created edge {new_edge['id']}")

    # Remove old, add new
    remove_edge(data, bad_edge_id)
    data["edges"].extend(new_edges)

    clean_validation_warnings(data, [bad_edge_id, old_target, "target not found"])
    save(filename, data)


# =============================================================================
# 6. Duplicate node IDs — rename argument "c" prefixes to "a" prefixes
# =============================================================================
def fix_duplicate_ids(filename, num_bad_args, max_existing_arg):
    """Rename argument nodes that use 'c' prefix to 'a' prefix."""
    print(f"Fixing {filename} (duplicate IDs)...")
    data = load(filename)

    # Build rename map: c1→a(max+1), c2→a(max+2), etc.
    rename = {}
    for i in range(1, num_bad_args + 1):
        old_id = f"c{i}"
        new_id = f"a{max_existing_arg + i}"
        rename[old_id] = new_id

    print(f"    Rename map: {rename}")

    # Rename argument node IDs (only arguments, not concepts)
    renamed_count = 0
    for arg in data.get("arguments", []):
        if arg["id"] in rename:
            old = arg["id"]
            arg["id"] = rename[old]
            renamed_count += 1
            print(f"    Renamed argument {old} -> {arg['id']}")

    assert renamed_count == num_bad_args, \
        f"Expected to rename {num_bad_args} arguments, but renamed {renamed_count}"

    # Update edge references: source and target pointing to renamed argument IDs
    # IMPORTANT: Only update edges that reference argument nodes, not concept nodes.
    # We need to determine which edges reference the argument vs the concept.
    # Since both "c1" (concept) and "c1" (argument) exist, edges referencing "c1"
    # actually point to concepts (verified in plan). So we do NOT change edges.
    #
    # However, we DO need to update _meta.cluster_summary[].arguments[] references.
    cluster_summary = data.get("_meta", {}).get("cluster_summary", {})
    for cluster_key, cluster in cluster_summary.items():
        args_list = cluster.get("arguments", [])
        updated = False
        for idx, aid in enumerate(args_list):
            if aid in rename:
                args_list[idx] = rename[aid]
                updated = True
        if updated:
            print(f"    Updated cluster_summary[{cluster_key}].arguments")

    # Also update _meta.cluster_membership if argument IDs appear there
    cluster_membership = data.get("_meta", {}).get("cluster_membership", {})
    keys_to_rename = {k: rename[k] for k in list(cluster_membership.keys()) if k in rename}
    for old_key, new_key in keys_to_rename.items():
        # Only rename if this is a membership entry for the argument (not the concept)
        # Since concepts also have c1 etc., we need to be careful.
        # cluster_membership typically maps node_id -> [concept_ids]
        # The argument "c1" would map to concept clusters, while concept "c1" also maps.
        # Since we can't distinguish, and the concept entry is the legitimate one,
        # we should NOT rename cluster_membership keys.
        pass

    clean_validation_warnings(data, ["Duplicate node ID"])
    save(filename, data)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Fixing iltur_graphs data issues")
    print("=" * 60)

    # 1. Self-loop
    fix_1958_88()
    print()

    # 2. Empty citation
    fix_1971_82()
    print()

    # 3. String "null" → JSON null
    fix_string_null("1958_152.json", ["h3", "h6", "h7"],
                    ["e_h3_resolves_null", "e_h6_resolves_null", "e_h7_resolves_null"])
    print()
    fix_string_null("1971_76.json", ["h1", "h2"],
                    ["e_h1_resolves_null", "e_h2_resolves_null"])
    print()

    # 4. Non-existent issue IDs
    fix_nonexistent_issue("1959_73.json", {"h6": "i25"}, ["e_h6_resolves_i25"])
    print()
    fix_nonexistent_issue("1967_332.json", {"h1": "i14", "h9": "i13"},
                          ["e_h1_resolves_i14", "e_h9_resolves_i13"])
    print()

    # 5. Comma/semicolon-separated
    fix_multi_issue("1964_199.json", "h5", "i1,i2", ["i1", "i2"])
    print()
    fix_multi_issue("1966_240.json", "h9", "i1,i2", ["i1", "i2"])
    print()
    fix_multi_issue("1971_311.json", "h5", "i1;i2", ["i1", "i2"])
    print()

    # 6. Duplicate IDs
    fix_duplicate_ids("1954_96.json", 1, 14)
    print()
    fix_duplicate_ids("1971_191.json", 7, 6)
    print()
    fix_duplicate_ids("1971_467.json", 9, 12)
    print()

    print("=" * 60)
    print("All fixes applied!")
    print("=" * 60)


if __name__ == "__main__":
    main()
