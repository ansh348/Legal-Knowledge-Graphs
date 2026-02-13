#!/usr/bin/env python3
"""
fetch_ik_ground_truth.py — Match IL-TUR CJPE cases to Indian Kanoon and
extract statute tags as independent ground truth for retrieval evaluation.

This script:
  1. Loads CJPE cases from IL-TUR (HuggingFace)
  2. Extracts distinctive search snippets from each case
  3. Searches Indian Kanoon API to find matching documents
  4. Pulls "Acts referred" statute tags for matched cases
  5. Saves a mapping file: {cjpe_id -> {ik_docid, acts_referred, ...}}

Usage:
    # Full pipeline
    python fetch_ik_ground_truth.py --api_token YOUR_TOKEN --n 2517

    # Step 1 only: extract snippets and search for matches
    python fetch_ik_ground_truth.py --api_token YOUR_TOKEN --n 100 --step match

    # Step 2 only: pull statute tags for already-matched cases
    python fetch_ik_ground_truth.py --api_token YOUR_TOKEN --step tags

    # Resume (skips already-matched cases)
    python fetch_ik_ground_truth.py --api_token YOUR_TOKEN --n 2517 --resume

Requirements:
    pip install datasets requests

Get API token:
    Sign up at https://api.indiankanoon.org/ (₹500 free credit)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("ik_ground_truth")
MATCH_FILE = OUTPUT_DIR / "ik_matches.json"
TAGS_FILE = OUTPUT_DIR / "ik_statute_tags.json"
QRELS_FILE = OUTPUT_DIR / "ik_qrels.json"
LOG_FILE = OUTPUT_DIR / "fetch_log.jsonl"

IK_API_BASE = "https://api.indiankanoon.org"

# Rate limiting: Indian Kanoon API doesn't document limits clearly,
# so be conservative
REQUEST_DELAY = 1.0  # seconds between requests


# ---------------------------------------------------------------------------
# SNIPPET EXTRACTION FROM CJPE TEXT
# ---------------------------------------------------------------------------

def extract_case_name(text: str) -> Optional[str]:
    """Try to extract party names (e.g., 'X vs Y') from case text."""
    # Common patterns in Indian SC judgments
    patterns = [
        # "Appellant vs Respondent" or "Petitioner v. Respondent"
        r'(?:^|\n)\s*(.{5,80}?)\s+(?:vs?\.?|versus)\s+(.{5,80?}?)(?:\s*\n|\s*$)',
        # "In the matter of X"
        r'[Ii]n\s+the\s+matter\s+of\s+(.{5,80}?)(?:\s*\n|\s*$)',
        # "X ... Appellant" on one line, "Y ... Respondent" on next
        r'(.{5,60}?)\s*\.{2,}\s*(?:Appellant|Petitioner)',
    ]
    for pat in patterns:
        m = re.search(pat, text[:3000])
        if m:
            name = m.group(0).strip()
            # Clean up
            name = re.sub(r'\s+', ' ', name)
            name = name[:120]
            return name
    return None


def extract_search_snippet(text: str, case_id: str) -> str:
    """Extract a distinctive search snippet from case text.

    Strategy:
    1. Try to find case name (party names)
    2. Extract year from case_id
    3. Find a distinctive sentence from early in the text
    4. Combine into a search query
    """
    year = case_id.split("_")[0] if "_" in case_id else None

    # Skip common boilerplate at the start
    # Many ILDC texts start with bench info, then get to substance
    lines = text.split('\n')
    substantive_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 30:
            continue
        # Skip common header patterns
        if re.match(r'^(JUDGMENT|ORDER|REPORTABLE|NON.REPORTABLE|IN THE SUPREME)', line, re.I):
            continue
        if re.match(r'^(CIVIL|CRIMINAL|WRIT|SPECIAL)\s+(APPEAL|PETITION|CASE)', line, re.I):
            continue
        if re.match(r'^(BENCH|CORAM|HON|JUSTICE|DATED|DATE)', line, re.I):
            continue
        substantive_lines.append(line)
        if len(substantive_lines) >= 10:
            break

    # Try to get case name
    case_name = extract_case_name(text)

    # Build search query
    parts = []

    if case_name:
        # Use case name as primary query
        # Clean it for search
        clean_name = re.sub(r'[^\w\s\.]', ' ', case_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        parts.append(clean_name[:80])
    elif substantive_lines:
        # Use first distinctive sentence
        snippet = substantive_lines[0][:150]
        # Quote it for exact matching
        parts.append(f'"{snippet[:80]}"')

    # Add year constraint
    if year and year.isdigit():
        parts.append(f"fromdate: 1-1-{year}")
        parts.append(f"todate: 31-12-{year}")

    # Always filter to Supreme Court
    parts.append("doctypes: supremecourt")

    return " ".join(parts)


def extract_fallback_snippets(text: str) -> List[str]:
    """Generate multiple fallback search queries if first search fails."""
    snippets = []

    # Try quoted distinctive phrases from different parts of the text
    sentences = re.split(r'[.!?]\s+', text[:5000])
    good_sentences = [
        s.strip() for s in sentences
        if 30 < len(s.strip()) < 200
           and not re.match(r'^(The |This |It |We |In )', s.strip())
    ]

    for sent in good_sentences[:3]:
        # Use a distinctive middle portion
        words = sent.split()
        if len(words) > 6:
            chunk = ' '.join(words[2:8])
            snippets.append(f'"{chunk}" doctypes: supremecourt')

    return snippets


# ---------------------------------------------------------------------------
# INDIAN KANOON API CLIENT
# ---------------------------------------------------------------------------

class IndianKanoonClient:
    """Simple client for Indian Kanoon API."""

    def __init__(self, api_token: str, delay: float = REQUEST_DELAY):
        self.api_token = api_token
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_token}",
            "Accept": "application/json",
        })
        self._last_request_time = 0
        self.total_requests = 0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def search(self, query: str, pagenum: int = 0) -> Optional[Dict]:
        """Search Indian Kanoon. Returns parsed JSON response."""
        self._rate_limit()
        try:
            resp = self.session.post(
                f"{IK_API_BASE}/search/",
                data={"formInput": query, "pagenum": pagenum},
                timeout=30,
            )
            self.total_requests += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  Search error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"  Search exception: {e}")
            return None

    def get_doc(self, docid: int) -> Optional[Dict]:
        """Get full document with metadata."""
        self._rate_limit()
        try:
            resp = self.session.post(
                f"{IK_API_BASE}/doc/{docid}/",
                timeout=30,
            )
            self.total_requests += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  Doc error {resp.status_code} for {docid}")
                return None
        except Exception as e:
            print(f"  Doc exception for {docid}: {e}")
            return None

    def get_doc_meta(self, docid: int) -> Optional[Dict]:
        """Get document metadata (structural analysis, acts, etc)."""
        self._rate_limit()
        try:
            resp = self.session.post(
                f"{IK_API_BASE}/docmeta/{docid}/",
                timeout=30,
            )
            self.total_requests += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  Meta error {resp.status_code} for {docid}")
                return None
        except Exception as e:
            print(f"  Meta exception for {docid}: {e}")
            return None


# ---------------------------------------------------------------------------
# MATCHING: CJPE -> INDIAN KANOON
# ---------------------------------------------------------------------------

def match_case(client: IndianKanoonClient, case_id: str, text: str) -> Optional[Dict]:
    """Try to match a CJPE case to an Indian Kanoon document.

    Returns {docid, title, score, method} or None.
    """
    # Strategy 1: Primary search with case name / snippet
    query = extract_search_snippet(text, case_id)
    result = client.search(query)

    if result and result.get("docs"):
        doc = result["docs"][0]
        docid = doc.get("tid")
        title = re.sub(r'<[^>]+>', '', doc.get("title", ""))  # Strip HTML tags
        return {
            "docid": docid,
            "title": title,
            "method": "primary",
            "query": query[:200],
        }

    # Strategy 2: Fallback with quoted text snippets
    fallbacks = extract_fallback_snippets(text)
    for i, fb_query in enumerate(fallbacks):
        result = client.search(fb_query)
        if result and result.get("docs"):
            doc = result["docs"][0]
            return {
                "docid": doc.get("tid"),
                "title": re.sub(r'<[^>]+>', '', doc.get("title", "")),
                "method": f"fallback_{i}",
                "query": fb_query[:200],
            }

    return None


def validate_match(client: IndianKanoonClient, docid: int, case_text: str) -> bool:
    """Light validation: check if IK doc text overlaps significantly with CJPE text.

    Uses a simple word overlap heuristic on first 500 words.
    """
    doc = client.get_doc(docid)
    if not doc:
        return False

    # Extract text content from IK doc (it returns HTML)
    ik_text = doc.get("doc", "")
    # Strip HTML tags
    ik_text = re.sub(r'<[^>]+>', ' ', ik_text)
    ik_text = re.sub(r'\s+', ' ', ik_text).strip()

    # Compare word sets from first 500 words
    case_words = set(case_text.lower().split()[:500])
    ik_words = set(ik_text.lower().split()[:500])

    if not case_words or not ik_words:
        return False

    overlap = len(case_words & ik_words) / min(len(case_words), len(ik_words))
    return overlap > 0.3  # 30% word overlap threshold


# ---------------------------------------------------------------------------
# STATUTE TAG EXTRACTION
# ---------------------------------------------------------------------------

def extract_statute_tags(doc_data: Dict) -> List[Dict]:
    """Extract statute/act references from an Indian Kanoon document response.

    Indian Kanoon embeds act/section references as <a href="/doc/XXXXX/">...</a>
    links in the document HTML. We extract and deduplicate these.
    """
    tags = []
    doc_html = doc_data.get("doc", "")

    # Extract all internal links
    act_links = re.findall(
        r'<a\s+href="/doc/(\d+)/"[^>]*>([^<]+)</a>',
        doc_html
    )

    seen_normalized = set()
    for link_docid, link_text in act_links:
        # Clean whitespace
        link_text = re.sub(r'\s+', ' ', link_text.strip())

        # Skip very short or generic references
        if len(link_text) < 3:
            continue
        if link_text.lower() in ('the act', 'this act', 'the code', 'section', 'article',
                                 'act', 'code', 'rule', 'order', 'the said act'):
            continue

        # Must contain a statute/section keyword
        if not any(kw in link_text.lower() for kw in [
            'section', 'article', 'rule', 'order', 'act', 'code',
            'constitution', 'schedule', 'clause',
        ]):
            continue

        normalized = normalize_statute_ref(link_text)
        if not normalized or len(normalized) < 3:
            continue

        # Dedup by normalized form
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            tags.append({
                "raw": link_text,
                "normalized": normalized,
                "ik_docid": link_docid,
            })

    return tags


def extract_acts_from_meta(meta_data: Dict) -> List[Dict]:
    """Extract acts from the /docmeta/ endpoint response."""
    tags = []

    # The meta endpoint may include act references differently
    # Check for 'acts' or similar fields
    for key in ['acts', 'Acts', 'acts_referred', 'statutes']:
        if key in meta_data:
            acts = meta_data[key]
            if isinstance(acts, list):
                for act in acts:
                    if isinstance(act, str):
                        tags.append({"raw": act, "normalized": normalize_statute_ref(act)})
                    elif isinstance(act, dict):
                        text = act.get("name", act.get("text", act.get("title", "")))
                        tags.append({"raw": text, "normalized": normalize_statute_ref(text)})

    return tags


def normalize_statute_ref(text: str) -> str:
    """Normalize a statute reference to a canonical form.

    E.g., 'Section 302 in Indian Penal Code' -> 'ipc_section_302'
         'Article 21 in Constitution of India' -> 'article_21_constitution'
    """
    text = text.lower().strip()

    # Remove common noise
    text = re.sub(r'\s+in\s+', ' ', text)
    text = re.sub(r'\s+of\s+', ' ', text)
    text = re.sub(r'\s+the\s+', ' ', text)

    # Common act abbreviations
    replacements = {
        'indian penal code': 'ipc',
        'code of criminal procedure': 'crpc',
        'code of civil procedure': 'cpc',
        'constitution of india': 'constitution',
        'constitution india': 'constitution',
        'bharatiya nyaya sanhita': 'bns',
        'bharatiya nagarik suraksha sanhita': 'bnss',
        'evidence act': 'evidence_act',
        'limitation act': 'limitation_act',
        'arbitration and conciliation act': 'arbitration_act',
        'companies act': 'companies_act',
        'transfer of property act': 'transfer_property_act',
        'negotiable instruments act': 'ni_act',
        'motor vehicles act': 'mv_act',
        'prevention of corruption act': 'pc_act',
        'narcotic drugs and psychotropic substances act': 'ndps_act',
        'armed forces special powers act': 'afspa',
        'right to information act': 'rti_act',
        'scheduled castes and scheduled tribes prevention of atrocities act': 'sc_st_act',
        'protection of women from domestic violence act': 'dv_act',
        'hindu marriage act': 'hm_act',
        'hindu succession act': 'hs_act',
        'muslim personal law': 'muslim_personal_law',
        'specific relief act': 'specific_relief_act',
        'land acquisition act': 'land_acquisition_act',
        'income tax act': 'it_act',
        'customs act': 'customs_act',
        'central excise act': 'excise_act',
        'goods and services tax': 'gst',
    }

    for full, abbr in replacements.items():
        text = text.replace(full, abbr)

    # Extract section/article number
    m = re.search(r'(section|article|rule|order|clause)\s*(\d+[a-z]?)', text)
    if m:
        kind = m.group(1)
        num = m.group(2)
        # Get the act part
        act_part = text.replace(m.group(0), '').strip()
        act_part = re.sub(r'[^\w\s]', '', act_part).strip()
        act_part = re.sub(r'\s+', '_', act_part)
        if act_part:
            return f"{act_part}_{kind}_{num}"
        else:
            return f"{kind}_{num}"

    # Fallback: clean and join
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '_', text).strip('_')
    return text if text else ""


# ---------------------------------------------------------------------------
# BUILD QRELS FROM STATUTE TAGS
# ---------------------------------------------------------------------------

def build_qrels_from_tags(
        tags_data: Dict[str, Dict],
        min_cases: int = 3,
        max_cases_pct: float = 0.5,
) -> Dict:
    """Build IR qrels from Indian Kanoon statute tags.

    Returns:
        {
            "queries": {concept_id: {"label": ..., "cases": {cjpe_id: 1}}},
            "stats": {...}
        }
    """
    # Count concept frequency across cases
    concept_cases: Dict[str, Dict[str, int]] = {}  # concept -> {cjpe_id: 1}

    for cjpe_id, case_data in tags_data.items():
        for tag in case_data.get("statute_tags", []):
            norm = tag.get("normalized", "")
            if not norm:
                continue
            if norm not in concept_cases:
                concept_cases[norm] = {}
            concept_cases[norm][cjpe_id] = 1  # binary relevance

    n_cases = len(tags_data)
    max_df = int(n_cases * max_cases_pct)

    # Filter by frequency
    eligible = {
        concept: cases
        for concept, cases in concept_cases.items()
        if min_cases <= len(cases) <= max_df
    }

    # Sort by frequency, take top concepts
    sorted_concepts = sorted(eligible.items(), key=lambda x: len(x[1]), reverse=True)

    queries = {}
    for concept, cases in sorted_concepts:
        queries[concept] = {
            "label": concept,
            "df": len(cases),
            "cases": cases,
        }

    return {
        "queries": queries,
        "stats": {
            "total_concepts": len(concept_cases),
            "eligible_concepts": len(eligible),
            "total_matched_cases": n_cases,
            "min_cases": min_cases,
            "max_cases_pct": max_cases_pct,
        }
    }


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def step_match(client: IndianKanoonClient, n: int, resume: bool, validate: bool):
    """Step 1: Match CJPE cases to Indian Kanoon documents."""
    from datasets import load_dataset

    print("Loading IL-TUR CJPE dataset...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    all_cases = ds['single_train']
    print(f"  Total cases: {len(all_cases)}")

    # Load existing matches if resuming
    matches = {}
    if resume and MATCH_FILE.exists():
        matches = json.loads(MATCH_FILE.read_text())
        print(f"  Resuming with {len(matches)} existing matches")

    # Process cases
    n_to_process = min(n, len(all_cases))
    matched = 0
    failed = 0

    for i in range(n_to_process):
        case = all_cases[i]
        cjpe_id = case['id']

        if cjpe_id in matches:
            continue

        print(f"[{i + 1}/{n_to_process}] {cjpe_id}...", end=" ", flush=True)

        match = match_case(client, cjpe_id, case['text'])

        if match:
            # Optional: validate the match
            if validate and match.get("docid"):
                is_valid = validate_match(client, match["docid"], case['text'])
                match["validated"] = is_valid
                if not is_valid:
                    print(f"MATCH FAILED VALIDATION (docid={match['docid']})")
                    failed += 1
                    # Log but still save — manual review can decide
                    match["validation_warning"] = True

            matches[cjpe_id] = match
            matched += 1
            print(f"✓ docid={match['docid']} ({match['method']})")
        else:
            matches[cjpe_id] = None
            failed += 1
            print("✗ no match")

        # Save periodically
        if (i + 1) % 50 == 0:
            MATCH_FILE.write_text(json.dumps(matches, indent=2))
            print(f"  [checkpoint] {matched} matched, {failed} failed, "
                  f"{client.total_requests} API calls")

    # Final save
    MATCH_FILE.write_text(json.dumps(matches, indent=2))
    print(f"\nMatching complete: {matched} matched, {failed} failed")
    print(f"Total API requests: {client.total_requests}")
    print(f"Saved to {MATCH_FILE}")


def step_tags(client: IndianKanoonClient):
    """Step 2: Pull statute tags for matched cases.

    Only uses /doc/ endpoint — /docmeta/ is sparse and doesn't contain
    statute tags. This means 1 API call per case, not 2.
    """
    if not MATCH_FILE.exists():
        print(f"Error: {MATCH_FILE} not found. Run --step match first.")
        return

    matches = json.loads(MATCH_FILE.read_text())
    valid_matches = {k: v for k, v in matches.items() if v and v.get("docid")}
    print(f"Pulling statute tags for {len(valid_matches)} matched cases...")
    print(f"  (1 API call per case — /doc/ endpoint only)")

    tags_data = {}
    if TAGS_FILE.exists():
        tags_data = json.loads(TAGS_FILE.read_text())
        print(f"  Resuming with {len(tags_data)} existing tag records")

    processed = 0
    for cjpe_id, match in valid_matches.items():
        if cjpe_id in tags_data:
            continue

        docid = match["docid"]
        print(f"[{processed + 1}/{len(valid_matches)}] {cjpe_id} (docid={docid})...", end=" ", flush=True)

        doc_data = client.get_doc(docid)
        statute_tags = []

        if doc_data:
            statute_tags = extract_statute_tags(doc_data)

        tags_data[cjpe_id] = {
            "ik_docid": docid,
            "ik_title": match.get("title", ""),
            "statute_tags": statute_tags,
            "n_tags": len(statute_tags),
        }

        processed += 1
        print(f"{len(statute_tags)} tags")

        # Checkpoint
        if processed % 50 == 0:
            TAGS_FILE.write_text(json.dumps(tags_data, indent=2))
            print(f"  [checkpoint] {processed} processed, "
                  f"{client.total_requests} API calls")

    # Final save
    TAGS_FILE.write_text(json.dumps(tags_data, indent=2))

    # Build qrels
    qrels = build_qrels_from_tags(tags_data)
    QRELS_FILE.write_text(json.dumps(qrels, indent=2))

    # Summary
    n_with_tags = sum(1 for v in tags_data.values() if v["n_tags"] > 0)
    all_tags = [t["normalized"] for v in tags_data.values() for t in v["statute_tags"]]
    unique_tags = set(all_tags)

    print(f"\nTag extraction complete:")
    print(f"  Cases with tags: {n_with_tags}/{len(tags_data)}")
    print(f"  Total tag instances: {len(all_tags)}")
    print(f"  Unique statutes: {len(unique_tags)}")
    print(f"  Eligible query concepts: {qrels['stats']['eligible_concepts']}")
    print(f"Saved to {TAGS_FILE}")
    print(f"Qrels saved to {QRELS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Indian Kanoon ground truth for IL-TUR CJPE retrieval evaluation"
    )
    parser.add_argument("--api_token", required=True, help="Indian Kanoon API token")
    parser.add_argument("--n", type=int, default=2517, help="Number of cases to process")
    parser.add_argument("--step", choices=["match", "tags", "both"], default="both",
                        help="Which step to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--validate", action="store_true",
                        help="Validate matches by comparing text (slower, costs more API calls)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API requests (seconds)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    client = IndianKanoonClient(args.api_token, delay=args.delay)

    if args.step in ("match", "both"):
        step_match(client, args.n, args.resume, args.validate)

    if args.step in ("tags", "both"):
        step_tags(client)

    print(f"\nTotal API requests: {client.total_requests}")
    print("Done!")


if __name__ == "__main__":
    main()