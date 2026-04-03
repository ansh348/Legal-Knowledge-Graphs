#!/usr/bin/env python3
"""
HUDOC Turkish ECHR Case Scraper v2
====================================
Downloads full-text judgments from the European Court of Human Rights
where Turkey is the respondent state.

Designed to feed directly into run_iltur.py --dataset echr --local_dir ./echr_turkey_cases

Key improvements over v1:
  - Importance filtering (default: 1-3, skipping low-value importance-4 cases)
  - Conclusion filtering: skips friendly settlements and struck-out cases by default
  - Parses conclusion field into binary violation label for outcome evaluation
  - Stores kpthesaurus (ECHR keyword tags) for clustering validation
  - Adds COLLECTION field to header (GRANDCHAMBER / CHAMBER / COMMITTEE)
  - Rate limiting per download thread (avoids HUDOC 429s)
  - Min text length filter (skips stub cases with no real content)
  - Produces labels_manifest.json mapping case_id -> {label, conclusion, importance, ...}
    so run_iltur.py can evaluate outcome prediction accuracy

Output:
  echr_turkey_cases/
    ├── metadata.csv                        (all case metadata, unfiltered)
    ├── labels_manifest.json                (case_id -> label/conclusion for evaluation)
    ├── CASE_OF_X_v._TURKEY__001-XXXXX.txt  (one per case, full text)
    └── scrape_log.txt                      (run log)

Requirements:
    pip install requests pandas

Usage:
    # Good defaults: importance 1-3, skip settlements, min 3000 chars
    python scrape_turkey_echr_v2.py

    # Only the most important cases (Grand Chamber + key judgments)
    python scrape_turkey_echr_v2.py --max-importance 2

    # Everything including importance 4 and settlements
    python scrape_turkey_echr_v2.py --max-importance 4 --include-settlements

    # First 200 cases, judgments only
    python scrape_turkey_echr_v2.py --count 200 --judgments-only

    # From 2015 onward
    python scrape_turkey_echr_v2.py --start-date 2015-01-01

Resume: Re-run the same command — already-downloaded files are skipped.
"""

import os
import re
import sys
import html
import json
import time
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

# ─── Configuration ───────────────────────────────────────────────────────────

OUTPUT_DIR = Path("echr_turkey_cases")

HUDOC_META_URL = (
    "https://hudoc.echr.coe.int/app/query/results"
    "?query=contentsitename:ECHR"
    " AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD))"
    ' AND (respondent:"TUR")'
    ' AND (documentcollectionid2:"JUDGMENTS"'
    ' OR documentcollectionid2:"DECISIONS"'
    ' OR documentcollectionid2:"GRANDCHAMBER"'
    ' OR documentcollectionid2:"CHAMBER")'
    ' AND (languageisocode:"ENG")'
    "{date_filter}"
    "&select=itemid,docname,appno,conclusion,importance,"
    "respondent,documentcollectionid2,languageisocode,"
    "kpthesaurus,ECLIIdentifier,kpdate"
    "&sort=itemid Ascending"
    "&start={start}&length={length}"
)

# Primary full-text endpoint (returns HTML body of judgment)
HUDOC_FULLTEXT_URL = (
    "https://hudoc.echr.coe.int/app/conversion/docx/html/body"
    "?library=ECHR&id={item_id}"
)

BATCH_SIZE = 500
MAX_THREADS = 8
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
PER_DOWNLOAD_DELAY = 0.25  # seconds between downloads per thread

# ─── Conclusion Parsing ──────────────────────────────────────────────────────

# Patterns in HUDOC conclusion field that indicate no substantive reasoning
SETTLEMENT_PATTERNS = [
    "struck out",
    "friendly settlement",
    "struck off",
    "unilateral declaration",
]

# Patterns indicating no violation (checked FIRST — more specific)
NO_VIOLATION_PATTERNS = [
    r"no violation",
    r"non-violation",
    r"inadmissible",
    r"incompatible",
    r"manifestly ill-founded",
]

# Patterns indicating a violation was found (at least one article)
# Uses negative lookbehind to avoid matching "no violation" as "violation"
VIOLATION_PATTERNS = [
    r"(?<!no )(?<!non-)violation of art",
    r"(?<!no )(?<!non-)violation of p\d",   # Protocol violations
    r"(?<!no )(?<!non-)violations of art",
]


def parse_conclusion_label(conclusion: str) -> Optional[int]:
    """Parse HUDOC conclusion string into binary label.

    HUDOC conclusions are semicolon-separated, e.g.:
        "Violation of Art. 6-1;No violation of Art. 8;Non-pecuniary damage - award"

    Returns:
        1 = at least one violation found
        0 = no violation / inadmissible / dismissed
        None = cannot determine (e.g., purely procedural, settlement, ambiguous)
    """
    if not conclusion:
        return None

    c = conclusion.lower().strip()

    # Skip settlements/struck-out — these aren't real decisions
    for pat in SETTLEMENT_PATTERNS:
        if pat in c:
            return None

    # Check each semicolon-delimited segment independently
    # This prevents "no violation of art. 8" from triggering a positive match
    segments = [s.strip() for s in c.split(";") if s.strip()]

    has_violation = False
    has_no_violation = False

    for seg in segments:
        if any(re.search(p, seg) for p in NO_VIOLATION_PATTERNS):
            has_no_violation = True
        if any(re.search(p, seg) for p in VIOLATION_PATTERNS):
            has_violation = True

    if has_violation:
        # Any violation found (even alongside some non-violations) = label 1
        return 1
    if has_no_violation:
        return 0

    # Check for inadmissible as a standalone conclusion (no segments matched above)
    if "inadmissible" in c or "incompatible" in c:
        return 0

    return None


def is_settlement_or_struck_out(conclusion: str) -> bool:
    """Check if case was a friendly settlement or struck out."""
    if not conclusion:
        return False
    c = conclusion.lower()
    return any(pat in c for pat in SETTLEMENT_PATTERNS)


def classify_collection(collection_str: str) -> str:
    """Extract the most significant collection type."""
    if not collection_str:
        return "UNKNOWN"
    c = collection_str.upper()
    if "GRANDCHAMBER" in c:
        return "GRANDCHAMBER"
    if "CHAMBER" in c:
        return "CHAMBER"
    if "JUDGMENTS" in c:
        return "JUDGMENT"
    if "DECISIONS" in c:
        return "DECISION"
    if "COMMITTEE" in c:
        return "COMMITTEE"
    return collection_str.strip()


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("echr_scraper")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(output_dir / "scrape_log.txt", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Metadata ────────────────────────────────────────────────────────────────

def fetch_metadata(
    start_date=None,
    end_date=None,
    max_count=None,
    judgments_only=False,
):
    """Fetch Turkish case metadata from HUDOC in batches of 500."""
    logger = logging.getLogger("echr_scraper")
    all_results = []
    start = 0
    total = None

    date_filter = ""
    if start_date:
        date_filter += f' AND (kpdate>="{start_date}")'
    if end_date:
        date_filter += f' AND (kpdate<="{end_date}")'

    while True:
        url = HUDOC_META_URL.format(
            start=start, length=BATCH_SIZE, date_filter=date_filter
        )

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                logger.warning(f"Metadata attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error("Giving up on metadata fetch.")
                    return all_results
                time.sleep(2 ** attempt)

        results = data.get("results", [])
        if total is None:
            total = data.get("resultcount", 0)
            logger.info(f"Total Turkish ECHR cases on HUDOC: {total}")

        if not results:
            break

        for item in results:
            row = item.get("columns", {})
            if judgments_only:
                coll = row.get("documentcollectionid2", "")
                if not any(k in coll for k in ("JUDGMENTS", "CHAMBER", "GRANDCHAMBER")):
                    continue
            all_results.append(row)

        start += len(results)
        logger.info(f"  Fetched {start}/{total} metadata records ({len(all_results)} kept)")

        if max_count and len(all_results) >= max_count:
            all_results = all_results[:max_count]
            break
        if start >= total:
            break
        time.sleep(0.3)

    logger.info(f"Metadata complete: {len(all_results)} cases")
    return all_results


# ─── Full Text ───────────────────────────────────────────────────────────────

def clean_html(raw: str) -> str:
    """Strip HTML to clean readable text."""
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(p|div|tr|li|h[1-6])[^>]*>', '\n', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def download_full_text(item_id: str, session: requests.Session) -> Optional[str]:
    """Download full judgment text for one case."""
    url = HUDOC_FULLTEXT_URL.format(item_id=item_id)
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.text) > 200:
                return clean_html(resp.text)
            if resp.status_code == 429:
                # Rate limited — back off significantly
                wait = 5 * (attempt + 1)
                logging.getLogger("echr_scraper").debug(
                    f"Rate limited on {item_id}, waiting {wait}s"
                )
                time.sleep(wait)
                continue
        except Exception:
            pass
        time.sleep(1.5 ** attempt)
    return None


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name).strip('_.')
    return name[:max_len]


# ─── Filtering ───────────────────────────────────────────────────────────────

def filter_cases(
    metadata: List[Dict],
    max_importance: int = 3,
    include_settlements: bool = False,
    judgments_only: bool = False,
    min_importance: int = 1,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Apply quality filters to metadata and return (filtered, stats).

    HUDOC importance levels:
        1 = Key case (Grand Chamber, landmark)
        2 = Important (significant legal contribution)
        3 = Normal (routine application of established law)
        4 = Low (repetitive, formulaic, minimal reasoning)
    """
    logger = logging.getLogger("echr_scraper")
    stats = {
        "total_before_filter": len(metadata),
        "skipped_importance": 0,
        "skipped_settlement": 0,
        "skipped_collection": 0,
        "kept": 0,
    }

    filtered = []
    for row in metadata:
        # Importance filter
        try:
            imp = int(row.get("importance", 4))
        except (ValueError, TypeError):
            imp = 4

        if imp < min_importance or imp > max_importance:
            stats["skipped_importance"] += 1
            continue

        # Settlement / struck-out filter
        conclusion = row.get("conclusion", "") or ""
        if not include_settlements and is_settlement_or_struck_out(conclusion):
            stats["skipped_settlement"] += 1
            continue

        # Judgments-only filter (applied on top of metadata fetch filter)
        if judgments_only:
            coll = row.get("documentcollectionid2", "")
            if not any(k in coll for k in ("JUDGMENTS", "CHAMBER", "GRANDCHAMBER")):
                stats["skipped_collection"] += 1
                continue

        filtered.append(row)

    stats["kept"] = len(filtered)

    logger.info(f"Filtering results:")
    logger.info(f"  Before:              {stats['total_before_filter']}")
    logger.info(f"  Skipped (importance): {stats['skipped_importance']}")
    logger.info(f"  Skipped (settlement): {stats['skipped_settlement']}")
    logger.info(f"  Skipped (collection): {stats['skipped_collection']}")
    logger.info(f"  Kept:                {stats['kept']}")

    return filtered, stats


def print_distribution(metadata: List[Dict], label: str = ""):
    """Print importance and collection distributions for the dataset."""
    logger = logging.getLogger("echr_scraper")

    logger.info(f"\n{'─' * 50}")
    logger.info(f"Dataset distribution{f' ({label})' if label else ''}:")

    # Importance distribution
    imp_dist: Dict[str, int] = {}
    for row in metadata:
        imp = str(row.get("importance", "?"))
        imp_dist[imp] = imp_dist.get(imp, 0) + 1
    logger.info(f"  Importance: {dict(sorted(imp_dist.items()))}")

    # Collection distribution
    coll_dist: Dict[str, int] = {}
    for row in metadata:
        coll = classify_collection(row.get("documentcollectionid2", ""))
        coll_dist[coll] = coll_dist.get(coll, 0) + 1
    logger.info(f"  Collection: {dict(sorted(coll_dist.items()))}")

    # Conclusion type distribution (top 10)
    conclusion_types: Dict[str, int] = {}
    for row in metadata:
        conclusion = (row.get("conclusion", "") or "").strip()
        if not conclusion:
            key = "(empty)"
        elif "violation" in conclusion.lower():
            key = "violation"
        elif "no violation" in conclusion.lower():
            key = "no_violation"
        elif "struck out" in conclusion.lower() or "friendly" in conclusion.lower():
            key = "settlement/struck"
        elif "inadmissible" in conclusion.lower():
            key = "inadmissible"
        else:
            key = "other"
        conclusion_types[key] = conclusion_types.get(key, 0) + 1
    logger.info(f"  Conclusion: {dict(sorted(conclusion_types.items()))}")

    # Label distribution
    label_dist = {0: 0, 1: 0, "unknown": 0}
    for row in metadata:
        lbl = parse_conclusion_label(row.get("conclusion", ""))
        if lbl is None:
            label_dist["unknown"] += 1
        else:
            label_dist[lbl] += 1
    logger.info(f"  Labels:     {dict(label_dist)}")

    logger.info(f"{'─' * 50}\n")


# ─── Labels Manifest ─────────────────────────────────────────────────────────

def build_labels_manifest(
    metadata: List[Dict],
    output_dir: Path,
) -> Dict[str, Dict]:
    """Build a manifest mapping case filenames to labels + metadata.

    This allows run_iltur.py to evaluate outcome prediction accuracy
    when using --local_dir mode (where labels are otherwise unavailable).
    """
    manifest: Dict[str, Dict] = {}

    for row in metadata:
        item_id = row.get("itemid", "")
        doc_name = row.get("docname", "") or row.get("DocName", "") or f"case_{item_id}"
        conclusion = row.get("conclusion", "") or ""
        importance = row.get("importance", None)
        collection = classify_collection(row.get("documentcollectionid2", ""))
        kpthesaurus = row.get("kpthesaurus", "") or ""
        appno = row.get("appno", "") or ""
        kpdate = row.get("kpdate", "") or ""

        safe = sanitize_filename(doc_name)
        file_stem = f"{safe}__{item_id}"

        label = parse_conclusion_label(conclusion)

        manifest[file_stem] = {
            "item_id": item_id,
            "case_name": doc_name,
            "appno": appno,
            "date": kpdate,
            "importance": importance,
            "collection": collection,
            "conclusion": conclusion,
            "label": label,
            "kpthesaurus": kpthesaurus,
        }

    # Save manifest
    manifest_path = output_dir / "labels_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Turkish ECHR cases from HUDOC (v2 — quality-filtered)"
    )

    # Count / date range
    parser.add_argument("--count", type=int, default=None,
                        help="Max cases to download (default: all that pass filters)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date YYYY-MM-DD")

    # Quality filters
    parser.add_argument("--max-importance", type=int, default=3,
                        help="Skip cases with importance > N (1=key 2=important 3=normal 4=low). Default: 3")
    parser.add_argument("--min-importance", type=int, default=1,
                        help="Skip cases with importance < N. Default: 1")
    parser.add_argument("--include-settlements", action="store_true",
                        help="Include friendly settlements and struck-out cases (skipped by default)")
    parser.add_argument("--judgments-only", action="store_true",
                        help="Skip admissibility decisions, keep only judgments")
    parser.add_argument("--min-text-length", type=int, default=3000,
                        help="Skip cases with full text shorter than N chars (default: 3000)")

    # Download settings
    parser.add_argument("--threads", type=int, default=MAX_THREADS,
                        help=f"Download threads (default: {MAX_THREADS})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--delay", type=float, default=PER_DOWNLOAD_DELAY,
                        help=f"Delay between downloads per thread in seconds (default: {PER_DOWNLOAD_DELAY})")

    # Metadata-only mode
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only fetch and save metadata, don't download full texts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch metadata, apply filters, show stats, but download nothing")

    args = parser.parse_args()

    output_dir = Path(args.output)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("HUDOC Turkish ECHR Case Scraper v2")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info(f"Output:          {output_dir.resolve()}")
    logger.info(f"Threads:         {args.threads}")
    logger.info(f"Importance:      {args.min_importance}-{args.max_importance}")
    logger.info(f"Settlements:     {'included' if args.include_settlements else 'excluded'}")
    logger.info(f"Min text length: {args.min_text_length} chars")
    logger.info(f"Per-thread delay: {args.delay}s")
    if args.count:
        logger.info(f"Limit:           {args.count} cases")
    if args.start_date:
        logger.info(f"From:            {args.start_date}")
    if args.end_date:
        logger.info(f"To:              {args.end_date}")
    if args.judgments_only:
        logger.info(f"Mode:            judgments only")

    # ── 1. Metadata ──────────────────────────────────────────────────────
    logger.info("\n[1/4] Fetching metadata from HUDOC...")
    raw_metadata = fetch_metadata(
        start_date=args.start_date,
        end_date=args.end_date,
        max_count=None,  # Fetch all, filter locally for better stats
        judgments_only=False,  # Filter locally for better stats
    )
    if not raw_metadata:
        logger.error("No cases found. Check your network or HUDOC availability.")
        sys.exit(1)

    # Save full unfiltered metadata
    df_all = pd.DataFrame(raw_metadata)
    csv_path = output_dir / "metadata_all.csv"
    df_all.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved unfiltered metadata: {csv_path} ({len(df_all)} cases)")

    # Show distribution before filtering
    print_distribution(raw_metadata, label="before filtering")

    # ── 2. Filter ────────────────────────────────────────────────────────
    logger.info("[2/4] Applying quality filters...")
    filtered_metadata, filter_stats = filter_cases(
        raw_metadata,
        max_importance=args.max_importance,
        min_importance=args.min_importance,
        include_settlements=args.include_settlements,
        judgments_only=args.judgments_only,
    )

    if not filtered_metadata:
        logger.error("No cases pass filters. Try relaxing --max-importance or --include-settlements.")
        sys.exit(1)

    # Apply --count limit AFTER filtering (so you get the N best, not N random)
    if args.count and len(filtered_metadata) > args.count:
        # Sort by importance (ascending = best first), then by date (newest first)
        filtered_metadata.sort(
            key=lambda r: (
                int(r.get("importance", 4)),
                -(len(r.get("kpdate", "") or "")),  # proxy for recency
            )
        )
        filtered_metadata = filtered_metadata[:args.count]
        logger.info(f"Limited to {args.count} cases (best importance first)")

    # Show distribution after filtering
    print_distribution(filtered_metadata, label="after filtering")

    # Save filtered metadata
    df_filtered = pd.DataFrame(filtered_metadata)
    csv_filtered_path = output_dir / "metadata.csv"
    df_filtered.to_csv(csv_filtered_path, index=False, encoding="utf-8")
    logger.info(f"Saved filtered metadata: {csv_filtered_path} ({len(df_filtered)} cases)")

    # Build labels manifest
    logger.info("Building labels manifest...")
    manifest = build_labels_manifest(filtered_metadata, output_dir)
    n_labeled = sum(1 for v in manifest.values() if v["label"] is not None)
    logger.info(f"Labels manifest: {len(manifest)} entries, {n_labeled} with binary labels")

    if args.dry_run:
        logger.info("\n[DRY RUN] Stopping before download. Use without --dry-run to proceed.")
        return

    if args.metadata_only:
        logger.info("\n[METADATA ONLY] Skipping full text download.")
        return

    # ── 3. Download full texts ───────────────────────────────────────────
    logger.info(f"\n[3/4] Downloading full texts ({len(filtered_metadata)} cases)...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ECHR-Research-Scraper/2.0 (academic-legal-nlp)",
        "Accept": "text/html,application/xhtml+xml",
    })

    success, failed, skipped, too_short = 0, 0, 0, 0
    download_delay = args.delay

    def process_case(row):
        nonlocal download_delay

        item_id = row.get("itemid", "")
        doc_name = (
            row.get("docname", "")
            or row.get("DocName", "")
            or f"case_{item_id}"
        )
        if not item_id:
            return None, "failed", doc_name, 0

        safe = sanitize_filename(doc_name)
        fpath = output_dir / f"{safe}__{item_id}.txt"

        # Resume: skip already-downloaded files
        if fpath.exists() and fpath.stat().st_size > 100:
            return fpath, "skipped", doc_name, fpath.stat().st_size

        # Rate limiting
        time.sleep(download_delay)

        text = download_full_text(item_id, session)
        if not text or len(text) < 100:
            return None, "failed", doc_name, 0

        # Min text length filter (applied after download since we don't know
        # length from metadata)
        if len(text) < args.min_text_length:
            return None, "too_short", doc_name, len(text)

        # Parse metadata for header
        conclusion = row.get("conclusion", "") or "N/A"
        importance = row.get("importance", "N/A")
        collection = classify_collection(row.get("documentcollectionid2", ""))
        appno = row.get("appno", "N/A")
        ecli = row.get("ECLIIdentifier", "N/A")
        kpthesaurus = row.get("kpthesaurus", "") or ""
        kpdate = row.get("kpdate", "") or "N/A"
        label = parse_conclusion_label(row.get("conclusion", ""))
        label_str = str(label) if label is not None else "N/A"

        header = (
            f"{'=' * 70}\n"
            f"CASE:       {doc_name}\n"
            f"ITEM ID:    {item_id}\n"
            f"APP NO:     {appno}\n"
            f"IMPORTANCE: {importance}\n"
            f"COLLECTION: {collection}\n"
            f"CONCLUSION: {conclusion}\n"
            f"LABEL:      {label_str}\n"
            f"DATE:       {kpdate}\n"
            f"ECLI:       {ecli}\n"
            f"KEYWORDS:   {kpthesaurus}\n"
            f"HUDOC:      https://hudoc.echr.coe.int/eng?i={item_id}\n"
            f"{'=' * 70}\n\n"
        )
        fpath.write_text(header + text, encoding="utf-8")
        return fpath, "success", doc_name, len(text)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(process_case, r): i
            for i, r in enumerate(filtered_metadata)
        }
        for fut in as_completed(futures):
            try:
                _, status, doc_name, text_len = fut.result()
                if status == "success":
                    success += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "too_short":
                    too_short += 1
                    logger.debug(f"Too short ({text_len} chars): {doc_name}")
                else:
                    failed += 1
                    logger.debug(f"Failed: {doc_name}")
            except Exception as e:
                failed += 1
                logger.debug(f"Error: {e}")

            done = success + failed + skipped + too_short
            if done % 50 == 0 or done == len(filtered_metadata):
                logger.info(
                    f"  [{done}/{len(filtered_metadata)}] "
                    f"ok={success}  skip={skipped}  short={too_short}  fail={failed}"
                )

    # ── 4. Summary ───────────────────────────────────────────────────────
    txt_files = [
        f for f in output_dir.glob("*.txt")
        if f.name not in ("scrape_log.txt",)
    ]
    size_mb = sum(f.stat().st_size for f in txt_files) / (1024 * 1024)

    # Text length distribution of downloaded files
    lengths = sorted([f.stat().st_size for f in txt_files])
    p25 = lengths[len(lengths) // 4] if lengths else 0
    p50 = lengths[len(lengths) // 2] if lengths else 0
    p75 = lengths[3 * len(lengths) // 4] if lengths else 0

    logger.info(f"\n[4/4] Done!")
    logger.info(f"  Downloaded: {success}")
    logger.info(f"  Skipped:    {skipped}  (already existed)")
    logger.info(f"  Too short:  {too_short}  (< {args.min_text_length} chars)")
    logger.info(f"  Failed:     {failed}")
    logger.info(f"  Files:      {len(txt_files)}")
    logger.info(f"  Size:       {size_mb:.1f} MB")
    logger.info(f"  Text sizes: p25={p25//1024}KB  p50={p50//1024}KB  p75={p75//1024}KB")
    logger.info(f"  Directory:  {output_dir.resolve()}")

    if failed > 0:
        logger.info(
            f"\n  Tip: Re-run the same command to retry {failed} failed downloads "
            f"(existing files are skipped automatically)."
        )

    # Print usage hint for run_iltur.py
    logger.info(f"\n{'─' * 60}")
    logger.info("To extract legal reasoning graphs from these cases:")
    logger.info(f"  python run_iltur.py \\")
    logger.info(f"    --dataset echr \\")
    logger.info(f"    --local_dir {output_dir} \\")
    logger.info(f"    --ontology echr_ontology_compiled_v3_1.json \\")
    logger.info(f"    --jurisdiction echr \\")
    logger.info(f"    --n {min(len(txt_files), 100)} --concurrent 10")
    logger.info(f"{'─' * 60}")


if __name__ == "__main__":
    main()