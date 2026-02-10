#!/usr/bin/env python3
"""
structure_cases.py

Simple structured legal case extractor — NO graph, NO ontology, NO edges.
Just takes raw Indian court judgment text and structures it into clean JSON sections.

Uses Grok LLM (x.ai) for extraction.

Usage:
    python structure_cases.py [--n 50] [--start 0] [--concurrent 20]

Requirements:
    pip install datasets python-dotenv httpx

Environment:
    XAI_API_KEY in .env file
"""

import os
import json
import asyncio
import argparse
import random
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
from datasets import load_dataset
from dotenv import load_dotenv
import httpx

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path("structured-nongraph-cases")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
MODEL_ID = "grok-4-1-fast-reasoning"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("structure_cases.log", encoding="utf-8")],
)
logger = logging.getLogger("StructureCases")


# =============================================================================
# GROK CLIENT (minimal, reused from your extractor)
# =============================================================================

class GrokClient:
    def __init__(self, api_key: str, model_id: str = MODEL_ID):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.x.ai/v1"
        self._http = httpx.AsyncClient(
            timeout=180.0,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=20),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    async def close(self):
        await self._http.aclose()

    async def complete(self, prompt: str, system: str, temperature: float = 0.1, max_tokens: int = 8192) -> str:
        system += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no ```json blocks."
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await self._http.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_id,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = min(2 ** attempt + random.uniform(0, 1), 60)
                        logger.warning(f"HTTP {resp.status_code}, retrying in {delay:.1f}s ({attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage")
                if usage:
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.total_requests += 1
                return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt + random.uniform(0, 1), 60)
                    logger.warning(f"Timeout, retrying in {delay:.1f}s ({attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                raise

        raise RuntimeError("Exhausted all retries")


# =============================================================================
# EXTRACTION PROMPT — single-pass structured extraction
# =============================================================================

SYSTEM_PROMPT = """You are a legal document structuring system for Indian court judgments.
Your job is to read a raw judgment and extract its content into clean, structured sections.
Do NOT build a graph. Do NOT create edges or relations. Just organize the text.
Be exhaustive — extract everything relevant. Be precise — use the court's own language where possible."""

STRUCTURE_PROMPT = """Read the following Indian court judgment and extract a structured summary.

Return a JSON object with these sections:

{{
  "metadata": {{
    "case_name": "Party A v. Party B",
    "case_year": 2020,
    "court": "Supreme Court of India / High Court of ...",
    "judges": ["Justice X", "Justice Y"],
    "case_number": "Civil Appeal No. ..." or null,
    "date_of_judgment": "YYYY-MM-DD" or null
  }},
  "facts": [
    {{
      "id": "f1",
      "text": "Clear statement of what happened",
      "type": "material | procedural | background | disputed | admitted",
      "date": "YYYY-MM-DD or null",
      "source": "petitioner | respondent | court | prosecution | lower_court"
    }}
  ],
  "legal_issues": [
    {{
      "id": "i1",
      "text": "The legal question the court needs to decide",
      "framed_by": "court | petitioner | respondent"
    }}
  ],
  "petitioner_arguments": [
    {{
      "id": "pa1",
      "text": "Argument made by petitioner/appellant side",
      "supporting_facts": ["f1", "f3"],
      "legal_basis": "Section/Article/Precedent cited, if any"
    }}
  ],
  "respondent_arguments": [
    {{
      "id": "ra1",
      "text": "Argument made by respondent/state side",
      "supporting_facts": ["f2"],
      "legal_basis": "Section/Article/Precedent cited, if any"
    }}
  ],
  "court_reasoning": [
    {{
      "id": "cr1",
      "text": "The court's own analysis or reasoning step",
      "addresses_issue": "i1",
      "key_observation": "The crux of what the court found"
    }}
  ],
  "holdings": [
    {{
      "id": "h1",
      "text": "The court's legal determination / rule of law applied",
      "resolves_issue": "i1",
      "in_favor_of": "petitioner | respondent | partial"
    }}
  ],
  "precedents_cited": [
    {{
      "case_name": "Name of cited case",
      "citation": "AIR 2010 SC 123 or (2010) 5 SCC 100 etc.",
      "treatment": "followed | distinguished | overruled | referred | doubted",
      "relevance": "Brief note on why this case was cited"
    }}
  ],
  "statutes_cited": [
    {{
      "name": "Name of the Act",
      "sections": ["Section 302", "Article 21"],
      "relevance": "How the statute was applied"
    }}
  ],
  "outcome": {{
    "disposition": "accepted | dismissed | partially_accepted | remanded",
    "summary": "One-paragraph summary of the final order",
    "relief_granted": "What relief was given, if any",
    "costs": "Any cost orders"
  }},
  "key_quotes": [
    {{
      "text": "Important verbatim quote from the judgment (max 200 chars)",
      "speaker": "court | petitioner_counsel | respondent_counsel",
      "significance": "Why this quote matters"
    }}
  ]
}}

RULES:
- Extract ALL facts, not just a few. Be thorough.
- Keep the court's language where it's precise. Paraphrase only for clarity.
- If a section has no content (e.g. no statutes cited), use an empty array [].
- IDs must be sequential: f1, f2, ... / i1, i2, ... / pa1, pa2, ... etc.
- For outcome.disposition, pick the closest: accepted, dismissed, partially_accepted, or remanded.
- Do NOT invent information not in the text.

JUDGMENT TEXT:
{text}"""


# =============================================================================
# JSON PARSING HELPER
# =============================================================================

def parse_json_response(raw: str) -> Optional[Dict]:
    """Try to parse JSON from LLM response, handling common quirks."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        start = 1
        end = len(lines)
        for idx in range(len(lines) - 1, 0, -1):
            if lines[idx].strip().startswith("```"):
                end = idx
                break
        text = "\n".join(lines[start:end])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost { ... }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint() -> tuple[Set[str], Dict]:
    completed = set()
    stats = {"success": 0, "errors": 0}

    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.json"):
            if f.name == "checkpoint.json":
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    json.load(fh)
                completed.add(f.stem)
            except Exception:
                logger.warning(f"Corrupt file {f.name}, removing")
                f.unlink()

    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
                completed.update(data.get("completed", []))
                stats = data.get("stats", stats)
        except Exception:
            pass

    if not stats.get("success") and completed:
        stats["success"] = len(completed)

    return completed, stats


def save_checkpoint(completed: Set[str], stats: Dict):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".tmp", prefix="ckpt_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump({"completed": list(completed), "stats": stats, "timestamp": datetime.now().isoformat()}, f)
        os.replace(tmp_path, CHECKPOINT_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# SINGLE CASE EXTRACTION
# =============================================================================

async def extract_one(
    client: GrokClient,
    case_id: str,
    text: str,
    label: int,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Structure one case via a single LLM call."""
    async with semaphore:
        try:
            # Truncate very long texts to avoid token limits
            max_chars = 80_000
            truncated = text[:max_chars] if len(text) > max_chars else text

            prompt = STRUCTURE_PROMPT.format(text=truncated)

            raw = await asyncio.wait_for(
                client.complete(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=8192),
                timeout=300,
            )

            structured = parse_json_response(raw)
            if structured is None:
                return {"case_id": case_id, "success": False, "error": "JSON parse failed"}

            # Attach ground truth label + case_id
            structured["_case_id"] = case_id
            structured["_label"] = label  # 0 = dismissed, 1 = accepted (from IL-TUR)
            structured["_text_length"] = len(text)
            structured["_extracted_at"] = datetime.now().isoformat()

            # Check if outcome matches label
            disp = (structured.get("outcome") or {}).get("disposition", "")
            predicted_accepted = disp in ("accepted", "partially_accepted")
            outcome_correct = predicted_accepted == (label == 1)

            # Save atomically
            output_path = OUTPUT_DIR / f"{case_id}.json"
            tmp_path = OUTPUT_DIR / f"{case_id}.json.tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(structured, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, output_path)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                return {"case_id": case_id, "success": False, "error": f"Write error: {e}"}

            n_facts = len(structured.get("facts", []))
            n_issues = len(structured.get("legal_issues", []))
            n_holdings = len(structured.get("holdings", []))
            n_precedents = len(structured.get("precedents_cited", []))

            return {
                "case_id": case_id,
                "success": True,
                "n_facts": n_facts,
                "n_issues": n_issues,
                "n_holdings": n_holdings,
                "n_precedents": n_precedents,
                "outcome_correct": outcome_correct,
                "disposition": disp,
            }

        except asyncio.TimeoutError:
            return {"case_id": case_id, "success": False, "error": "Timeout (300s)"}
        except Exception as e:
            return {"case_id": case_id, "success": False, "error": str(e)[:200]}


# =============================================================================
# BATCH RUNNER
# =============================================================================

async def run_batch(n_cases: int = 50, start_idx: int = 0, max_concurrent: int = 20):
    print("=" * 70)
    print("STRUCTURED CASE EXTRACTION (non-graph)")
    print("=" * 70)

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    print("\nLoading IL-TUR dataset (cjpe)...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    all_cases = ds["single_train"]
    print(f"Total cases available: {len(all_cases)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    completed, stats = load_checkpoint()
    print(f"Already completed: {len(completed)}")

    end_idx = min(start_idx + n_cases, len(all_cases))
    cases_to_process = []
    for i in range(start_idx, end_idx):
        case = all_cases[i]
        if case["id"] not in completed:
            cases_to_process.append((i, case))

    print(f"Cases to process: {len(cases_to_process)}")
    if not cases_to_process:
        print("All requested cases already processed!")
        return

    for key in ("success", "errors", "outcome_correct"):
        stats.setdefault(key, 0)

    client = GrokClient(api_key, model_id=MODEL_ID)

    print(f"\nModel: {MODEL_ID}")
    print(f"Concurrency: {max_concurrent}")
    print(f"Output: {OUTPUT_DIR}/")
    print("-" * 70)

    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = datetime.now()

    for batch_start in range(0, len(cases_to_process), max_concurrent):
        batch = cases_to_process[batch_start : batch_start + max_concurrent]

        tasks = [
            asyncio.create_task(extract_one(client, case["id"], case["text"], case["label"], semaphore))
            for _, case in batch
        ]

        try:
            results = await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            existing = {p.stem for p in OUTPUT_DIR.glob("*.json") if p.name != "checkpoint.json"}
            completed.update(existing)
            save_checkpoint(completed, stats)
            await client.close()
            print("\nInterrupted. Checkpoint saved.")
            return

        for r in results:
            if r["success"]:
                completed.add(r["case_id"])
                stats["success"] += 1
                if r["outcome_correct"]:
                    stats["outcome_correct"] += 1
                oc = "✓" if r["outcome_correct"] else "✗"
                disp = r.get("disposition") or "unknown"
                print(
                    f"[{stats['success']:3d}] {r['case_id']}: {oc} {disp:20s} | "
                    f"F:{r['n_facts']:2d} I:{r['n_issues']:2d} H:{r['n_holdings']:2d} P:{r['n_precedents']:2d}"
                )
            else:
                stats["errors"] += 1
                print(f"[ERR] {r['case_id']}: {r['error']}")

        save_checkpoint(completed, stats)

        batch_num = batch_start // max_concurrent + 1
        total_batches = (len(cases_to_process) + max_concurrent - 1) // max_concurrent
        print(
            f"  [batch {batch_num}/{total_batches}] "
            f"tokens: {client.total_prompt_tokens + client.total_completion_tokens:,} "
            f"({client.total_requests} requests)"
        )

    await client.close()

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Processed: {stats['success']} | Errors: {stats['errors']}")
    print(f"Time: {elapsed:.1f}s ({elapsed / max(stats['success'], 1):.1f}s per case)")

    if stats["success"] > 0:
        n = stats["success"]
        print(f"\n🎯 OUTCOME PREDICTION:")
        print(f"  Correct: {stats['outcome_correct']}/{n} ({stats['outcome_correct'] / n * 100:.1f}%)")

    print(f"\n💰 TOKEN USAGE:")
    print(f"  Prompt:     {client.total_prompt_tokens:,}")
    print(f"  Completion: {client.total_completion_tokens:,}")
    print(f"  Total:      {client.total_prompt_tokens + client.total_completion_tokens:,}")
    print(f"  Requests:   {client.total_requests:,}")

    print(f"\n✅ Outputs saved to: {OUTPUT_DIR}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Structure IL-TUR cases (non-graph)")
    parser.add_argument("--n", type=int, default=50, help="Number of cases")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--concurrent", type=int, default=20, help="Max concurrent")
    args = parser.parse_args()
    asyncio.run(run_batch(n_cases=args.n, start_idx=args.start, max_concurrent=args.concurrent))


if __name__ == "__main__":
    main()