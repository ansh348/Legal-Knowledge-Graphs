#!/usr/bin/env python3
"""
run_iltur.py

Batch extraction on IL-TUR dataset using the real extractor.py (v4 ontology-driven).

Usage:
    python run_iltur.py [--n 50] [--start 0] [--version v4] [--concurrent 5]

Requirements:
    pip install datasets python-dotenv httpx

Environment:
    XAI_API_KEY in .env file
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
from datasets import load_dataset
from dotenv import load_dotenv

# Import from extractor
from extractor import (
    GrokClient,
    ExtractionConfig,
    LegalReasoningExtractor,
)

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path("iltur_graphs")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
ONTOLOGY_PATH = "ontology_compiled.json"  # Update path if needed


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_checkpoint() -> tuple[Set[str], Dict]:
    """Load checkpoint of completed case IDs and stats.

    Also scans output directory for existing .json files to recover
    from corrupted checkpoints.
    """
    completed = set()
    stats = {}

    # First, scan output directory for existing extractions
    if OUTPUT_DIR.exists():
        for json_file in OUTPUT_DIR.glob("*.json"):
            if json_file.name != "checkpoint.json":
                # Case ID is the filename without extension
                completed.add(json_file.stem)

    # Then try to load checkpoint for stats (but completed already populated from files)
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                # Merge any additional completed from checkpoint
                completed.update(data.get('completed', []))
                stats = data.get('stats', {})
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load checkpoint ({e}), using file scan instead")

    return completed, stats


def save_checkpoint(completed: Set[str], stats: Dict):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'completed': [str(c) for c in completed],  # Fix: ensure strings
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)


# =============================================================================
# EXTRACTION
# =============================================================================

async def extract_one(
        extractor: LegalReasoningExtractor,
        case_id: str,
        text: str,
        label: int,
        semaphore: asyncio.Semaphore
) -> Dict:
    """Extract graph for one case."""

    async with semaphore:
        try:
            graph = await extractor.extract(
                text=text,
                case_id=case_id
            )

            # Defensive: filter out any None values from node lists
            # This can happen if node creation fails silently
            graph.facts = [n for n in graph.facts if n is not None]
            graph.concepts = [n for n in graph.concepts if n is not None]
            graph.issues = [n for n in graph.issues if n is not None]
            graph.arguments = [n for n in graph.arguments if n is not None]
            graph.holdings = [n for n in graph.holdings if n is not None]
            graph.precedents = [n for n in graph.precedents if n is not None]
            graph.justification_sets = [n for n in graph.justification_sets if n is not None]
            graph.edges = [e for e in graph.edges if e is not None]
            graph.reasoning_chains = [rc for rc in graph.reasoning_chains if rc is not None]

            # Save individual graph
            output_path = OUTPUT_DIR / f"{case_id}.json"
            try:
                json_str = graph.to_json()
                with open(output_path, 'w') as f:
                    f.write(json_str)
            except Exception as ser_err:
                return {
                    'case_id': case_id,
                    'success': False,
                    'error': f"Serialization error: {str(ser_err)[:150]}"
                }

            # Compute metrics
            n_facts = len(graph.facts)
            n_concepts = len(graph.concepts)
            n_issues = len(graph.issues)
            n_arguments = len(graph.arguments)
            n_holdings = len(graph.holdings)
            n_precedents = len(graph.precedents)
            n_edges = len(graph.edges)
            n_js = len(graph.justification_sets)
            n_chains = len(graph.reasoning_chains)

            # Check if outcome prediction matches label
            # Handle case where outcome might be None
            if graph.outcome is None:
                # No outcome extracted - treat as prediction failure
                outcome_accepted = False
                outcome_value = None
            else:
                outcome_value = graph.outcome.disposition.value
                outcome_accepted = outcome_value in ['allowed', 'partly_allowed', 'set_aside', 'remanded', 'modified']

            label_accepted = label == 1
            outcome_correct = outcome_accepted == label_accepted

            return {
                'case_id': case_id,
                'success': True,
                'label': label,
                'outcome': outcome_value,
                'outcome_correct': outcome_correct,
                'quality_tier': graph.quality_tier,
                'n_facts': n_facts,
                'n_concepts': n_concepts,
                'n_issues': n_issues,
                'n_arguments': n_arguments,
                'n_holdings': n_holdings,
                'n_precedents': n_precedents,
                'n_edges': n_edges,
                'n_justification_sets': n_js,
                'n_reasoning_chains': n_chains,
                'warnings': len(graph.validation_warnings),
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Include more context in error
            return {
                'case_id': case_id,
                'success': False,
                'error': f"{type(e).__name__}: {str(e)[:150]}"
            }


async def run_batch(
        n_cases: int = 50,
        start_idx: int = 0,
        pipeline_version: str = "v4",
        max_concurrent: int = 5
):
    """Run extraction on IL-TUR cases."""

    print("=" * 70)
    print(f"LEGAL GRAPH EXTRACTION - IL-TUR ({pipeline_version.upper()})")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    # Load dataset
    print("\nLoading IL-TUR dataset (cjpe)...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    all_cases = list(ds['single_train'])
    print(f"Total cases available: {len(all_cases)}")

    # Create output dir
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load checkpoint (now also scans existing output files)
    completed, stats = load_checkpoint()
    print(f"Already completed: {len(completed)}")

    # Select cases to process
    cases_to_process = []
    for i, case in enumerate(all_cases[start_idx:start_idx + n_cases], start=start_idx):
        if case['id'] not in completed:
            cases_to_process.append((i, case))

    print(f"Cases to process: {len(cases_to_process)}")

    if not cases_to_process:
        print("All requested cases already processed!")
        return

    # Initialize stats if empty
    if not stats:
        stats = {
            'success': 0,
            'errors': 0,
            'total_facts': 0,
            'total_concepts': 0,
            'total_issues': 0,
            'total_holdings': 0,
            'total_edges': 0,
            'total_chains': 0,
            'outcome_correct': 0,
            'quality_gold': 0,
            'quality_silver': 0,
            'quality_bronze': 0,
            'quality_reject': 0,
        }

    # Create client and extractor
    model_id = "grok-4-1-fast-reasoning"
    client = GrokClient(api_key, model_id=model_id)

    config = ExtractionConfig(
        model_id=model_id,
        pipeline_version=pipeline_version,
        ontology_path=ONTOLOGY_PATH if Path(ONTOLOGY_PATH).exists() else None,
    )

    extractor = LegalReasoningExtractor(client, config)

    print(f"\nPipeline: {pipeline_version}")
    print(f"Model: {model_id}")
    print(f"Ontology: {config.ontology_path or 'None'}")
    print(f"Concurrency: {max_concurrent}")
    print("\n" + "-" * 70)

    # Process
    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = datetime.now()

    for batch_start in range(0, len(cases_to_process), max_concurrent):
        batch = cases_to_process[batch_start:batch_start + max_concurrent]

        tasks = [
            asyncio.create_task(
                extract_one(extractor, case['id'], case['text'], case['label'], semaphore)
            )
            for _, case in batch
        ]

        try:
            results = await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Cancel outstanding tasks and save progress
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Best-effort: also count already-written JSONs as completed
            existing = {p.stem for p in OUTPUT_DIR.glob("*.json") if p.name != "checkpoint.json"}
            completed.update(existing)
            save_checkpoint(completed, stats)

            print("Interrupted. Checkpoint saved. Exiting cleanly.")
            return

        for result in results:
            if result['success']:
                completed.add(result['case_id'])
                stats['success'] += 1
                stats['total_facts'] += result['n_facts']
                stats['total_concepts'] += result['n_concepts']
                stats['total_issues'] += result['n_issues']
                stats['total_holdings'] += result['n_holdings']
                stats['total_edges'] += result['n_edges']
                stats['total_chains'] += result['n_reasoning_chains']

                if result['outcome_correct']:
                    stats['outcome_correct'] += 1

                tier = result['quality_tier']
                stats[f'quality_{tier}'] = stats.get(f'quality_{tier}', 0) + 1

                # Progress line
                oc = "✓" if result['outcome_correct'] else "✗"
                print(
                    f"[{stats['success']:3d}] {result['case_id']}: "
                    f"{oc} {result['quality_tier']:6s} | "
                    f"F:{result['n_facts']:2d} C:{result['n_concepts']:2d} "
                    f"H:{result['n_holdings']:2d} E:{result['n_edges']:2d} "
                    f"chains:{result['n_reasoning_chains']}"
                )
            else:
                stats['errors'] += 1
                print(f"[ERR] {result['case_id']}: {result['error']}")

        # Save checkpoint after each batch
        save_checkpoint(completed, stats)

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Processed: {stats['success']} | Errors: {stats['errors']}")
    print(f"Time: {elapsed:.1f}s ({elapsed / max(stats['success'], 1):.1f}s per case)")

    if stats['success'] > 0:
        print(f"\n📊 QUALITY:")
        print(f"  Gold:   {stats.get('quality_gold', 0)}")
        print(f"  Silver: {stats.get('quality_silver', 0)}")
        print(f"  Bronze: {stats.get('quality_bronze', 0)}")
        print(f"  Reject: {stats.get('quality_reject', 0)}")

        print(f"\n📈 AVERAGES:")
        n = stats['success']
        print(f"  Facts/case:    {stats['total_facts'] / n:.1f}")
        print(f"  Concepts/case: {stats['total_concepts'] / n:.1f}")
        print(f"  Holdings/case: {stats['total_holdings'] / n:.1f}")
        print(f"  Edges/case:    {stats['total_edges'] / n:.1f}")
        print(f"  Chains/case:   {stats['total_chains'] / n:.1f}")

        print(f"\n🎯 OUTCOME PREDICTION:")
        print(f"  Correct: {stats['outcome_correct']}/{n} ({stats['outcome_correct'] / n * 100:.1f}%)")

    print(f"\n✅ Outputs saved to: {OUTPUT_DIR}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run extractor on IL-TUR dataset")
    parser.add_argument("--n", type=int, default=50, help="Number of cases to process")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--version", choices=["v3", "v4"], default="v4", help="Pipeline version")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent extractions")

    args = parser.parse_args()

    asyncio.run(run_batch(
        n_cases=args.n,
        start_idx=args.start,
        pipeline_version=args.version,
        max_concurrent=args.concurrent
    ))


if __name__ == "__main__":
    main()