#!/usr/bin/env python3
"""
run_iltur.py

Batch extraction runner for IL-TUR / ECHR / Turkish (and custom HF datasets),
using extractor.py (v4 ontology-driven).

Examples
--------
# IL-TUR (default)
python run_iltur.py --dataset iltur --n 2500 --start 0 --version v4 --concurrent 20

# ECHR (preset tries LexGLUE ecthr_b first, then falls back to ecthr_cases)
python run_iltur.py --dataset echr --n 2500 --start 0 --version v4 --concurrent 20

# Turkish Constitutional Court via HuggingFace (icgcihan, 13.3k cases)
python run_iltur.py --dataset aym_hf --n 2500 --start 0 --version v4 --concurrent 20

# AYM HF — violations only (filter labels=1)
python run_iltur.py --dataset aym_hf --violations_only --n 2500 --version v4 --concurrent 20

# Turkish (preset tries koc-lab/law-turk)
python run_iltur.py --dataset tr --n 2500 --start 0 --version v4 --concurrent 20

# AYM local files (original scraper approach)
python run_iltur.py --dataset aym --local_dir ./aym_decisions --n 2500 --version v4

# Fully custom HF dataset (you can mix-and-match)
python run_iltur.py --hf_dataset lex_glue --hf_config ecthr_b --hf_split train --jurisdiction echr \
  --ontology echr_ontology_compiled.cleaned.json --n 2500

Requirements:
    pip install datasets python-dotenv httpx

Environment:
    XAI_API_KEY in .env file
"""

import os
import re
import json
import asyncio
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional, Tuple, Any, List

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # local-file mode still works without HuggingFace datasets

from dotenv import load_dotenv

# Import from extractor
from extractor import (
    GrokClient,
    ExtractionConfig,
    LegalReasoningExtractor,
)

# Citation pre-processing (optional but recommended)
try:
    from citation_preprocess import CitationPreprocessor
except ImportError:
    CitationPreprocessor = None

load_dotenv()

# =============================================================================
# DATASET PRESETS
# =============================================================================

# These are "best guess" presets; you can always override with --hf_dataset/--hf_config/--hf_split.
DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "iltur": {
        "hf_dataset": "Exploration-Lab/IL-TUR",
        "hf_config": "cjpe",
        "split_preference": ["single_train", "train"],
        "jurisdiction": "in",
        "ontology_candidates": [
            "ontology_compiled_fixed_cleaned_ids_normalized.json",
            "ontology_compiled_fixed_cleaned.json",
            "ontology_compiled.json",
        ],
        "language_instruction": None,
    },
    # ECHR: try LexGLUE first (common for violation prediction), fallback to ecthr_cases.
    "echr": {
        "hf_dataset": "lex_glue",
        "hf_config": "ecthr_b",
        "split_preference": ["train", "validation", "test"],
        "jurisdiction": "echr",
        "ontology_candidates": [
            "echr_ontology_compiled.cleaned.json",
            "echr_ontology_compiled_cleaned.json",
        ],
        "language_instruction": None,
    },
    "tr": {
        "hf_dataset": "koc-lab/law-turk",
        "hf_config": None,
        "split_preference": ["train", "validation", "test"],
        "jurisdiction": "tr",
        "ontology_candidates": [
            "turkish_legal_ontology_cleaned.json",
            "turkish_legal_ontology_cleaned.json",
        ],
        "language_instruction": (
            "The document may be in Turkish. Extract all structured fields in English. "
            "Anchors (start_char/end_char) MUST point into the original Turkish text."
        ),
    },
    # AYM-HF: icgcihan/Turkish_Constutional_Court_Decisions on HuggingFace.
    # 13.3k AYM individual complaint decisions with full text, rights labels (21 categories),
    # violation labels (0/1), and direct AYM links. No scraping needed.
    "aym_hf": {
        "hf_dataset": "icgcihan/Turkish_Constutional_Court_Decisions",
        "hf_config": None,
        "split_preference": ["train", "test", "validation"],
        "jurisdiction": "tr",
        "ontology_candidates": [
            "turkish_legal_ontology_cleaned.json",
            "turkish_legal_ontology_cleaned.json",
        ],
        "language_instruction": (
            "The document is a Turkish Constitutional Court (Anayasa Mahkemesi) individual complaint decision. "
            "Extract all structured fields in English. "
            "Anchors (start_char/end_char) MUST point into the original Turkish text."
        ),
        # Filter: set to True to only process violation cases (labels=1)
        "filter_violations_only": False,
    },
    # AYM: Local files scraped from Anayasa Mahkemesi (Turkish Constitutional Court).
    # Use with --local_dir pointing to a folder of .txt files.
    "aym": {
        "hf_dataset": None,  # signals local-file mode
        "hf_config": None,
        "split_preference": [],
        "jurisdiction": "tr",
        "ontology_candidates": [
            "turkish_legal_ontology_cleaned.json",
            "turkish_legal_ontology_cleaned.json",
        ],
        "language_instruction": (
            "The document is a Turkish Constitutional Court (Anayasa Mahkemesi) individual complaint decision. "
            "Extract all structured fields in English. "
            "Anchors (start_char/end_char) MUST point into the original Turkish text."
        ),
    },
}

# =============================================================================
# DEFAULT PROMPT CONTEXT + ACTOR ALIASES (OPTIONAL)
# =============================================================================

DEFAULT_PROMPT_CONTEXT: Dict[str, Dict[str, str]] = {
    "tr": {
        "language_preamble": (
            "IMPORTANT: The source text may be in Turkish. "
            "Return JSON fields in English, but anchors MUST align to the Turkish text."
        ),
        "metadata": (
            "Turkish court metadata tips:\n"
            "- Case numbers follow 'E. YYYY/NNNN, K. YYYY/NNNN' (Esas/Karar) format.\n"
            "- Başvuru numarası (application number) is used for AYM individual complaints.\n"
            "- Court names: Anayasa Mahkemesi (AYM), Yargıtay, Danıştay, Bölge Adliye Mahkemesi.\n"
            "- Dates are DD.MM.YYYY format (e.g. 12.03.2018 tarihli)."
        ),
        "facts": (
            "Turkish judgment fact sections:\n"
            "- 'OLAYLAR' or 'OLAYLAR VE OLGULAR' = events/facts section.\n"
            "- 'BAŞVURUCUNUN İDDİALARI' = applicant's claims (treat as party submissions, not court findings).\n"
            "- Facts from 'DEĞERLENDİRME' section are the court's own factual findings.\n"
            "- Turkish dates: DD.MM.YYYY; map to YYYY-MM-DD in output.\n"
            "- Actor mapping: başvurucu/davacı = petitioner; idare/davalı = respondent; mahkeme = court."
        ),
        "concepts": (
            "Turkish legal concepts:\n"
            "- Anayasa articles: 'Anayasa'nın 17. maddesi' = constitutional right (map to kind=statute_article).\n"
            "- Kanun references: '5237 sayılı Kanun' = numbered statute (kind=statute_section).\n"
            "- Common doctrines: ölçülülük ilkesi (proportionality), hukuki belirlilik (legal certainty),\n"
            "  silahların eşitliği (equality of arms), adil yargılanma hakkı (right to fair trial).\n"
            "- AYM uses ECHR-aligned concepts: Article 6 → AY m. 36 (fair trial), Article 8 → AY m. 20 (privacy).\n"
            "- Map Turkish concept names to their English equivalents in the output."
        ),
        "issues": (
            "Turkish judgment issue framing:\n"
            "- Issues are often stated in 'DEĞERLENDİRME' section as 'Anayasa'nın X. maddesinin ihlal edilip edilmediği'.\n"
            "- Constitutional Court (AYM) frames issues as whether a constitutional right was violated.\n"
            "- Look for 'incelenmesi gereken husus' (matter to be examined) or 'sorun' (problem/issue).\n"
            "- For Yargıtay: 'uyuşmazlık konusu' (subject of dispute) or 'bozma nedeni' (ground for reversal)."
        ),
        "arguments": (
            "Turkish judgment argument structure:\n"
            "- 'BAŞVURUCUNUN İDDİALARI' = petitioner's arguments (actor=petitioner).\n"
            "- 'BAKANLIĞIN GÖRÜŞÜ' or 'İDARENİN SAVUNMASI' = government/respondent observations.\n"
            "- Court reasoning in 'DEĞERLENDİRME' or 'GEREKÇE' sections (actor=court).\n"
            "- For AYM: look for 'kabul edilebilirlik' (admissibility) vs 'esas' (merits) arguments.\n"
            "- Dissenting opinions in 'KARŞIOY' or 'KARŞIOY GÖRÜŞÜ' section."
        ),
        "holdings": (
            "Turkish judgment holding extraction:\n"
            "- Holdings are in 'HÜKÜM' (operative part) or conclusion of 'DEĞERLENDİRME'.\n"
            "- AYM holdings: 'ihlal edildiğine' (violation found) → ratio; 'ihlal edilmediğine' (no violation) → ratio.\n"
            "- 'yeniden yargılama' = retrial ordered; 'tazminat' = compensation awarded.\n"
            "- For Yargıtay: 'bozma' = reversal; 'onama' = affirmation; 'kısmen bozma' = partial reversal."
        ),
        "precedents": (
            "Turkish citation formats:\n"
            "- AYM: 'E.2018/123, K.2019/456' or 'Başvuru Numarası: 2014/12345'.\n"
            "- Yargıtay: 'Yargıtay 4. Ceza Dairesi, E. 2017/1234, K. 2018/5678'.\n"
            "- ECHR citations in Turkish decisions: often by case name ('X v. Türkiye').\n"
            "- Resmi Gazete (Official Gazette) citations for legislation."
        ),
        "outcome": (
            "For Turkish dispositions: map 'kabul' / 'kısmen kabul' to allowed/partly_allowed; "
            "map 'ret' / 'reddi' to dismissed; map 'bozma' to set_aside/remanded as appropriate. "
            "For AYM individual complaints: 'ihlal' → allowed; 'ihlal yok' / 'kabul edilemezlik' → dismissed. "
            "For costs: 'yargılama giderleri' = court costs; check who bears them in HÜKÜM section."
        ),
    },
    "echr": {
        "metadata": (
            "ECHR metadata tips:\n"
            "- Case name format: 'Applicant v. Country' (e.g. 'Selmouni v. France').\n"
            "- Application number format: no. NNNN/YY (e.g. 36022/97).\n"
            "- Court composition: Chamber, Grand Chamber [GC], Committee.\n"
            "- Judge names often in header; President of the Chamber noted separately."
        ),
        "facts": (
            "ECHR judgment fact structure:\n"
            "- 'THE FACTS' or 'THE CIRCUMSTANCES OF THE CASE' = main facts section.\n"
            "- 'RELEVANT DOMESTIC LAW AND PRACTICE' = legal background (type=background).\n"
            "- The applicant's submissions are in the 'ALLEGED VIOLATION' sections.\n"
            "- Distinguish domestic proceedings facts (procedural) from core event facts (material).\n"
            "- Actors: applicant(s) = petitioner; respondent Government/State = respondent;\n"
            "  third-party intervener = third_party; the Court = court."
        ),
        "concepts": (
            "ECHR legal concepts:\n"
            "- Convention articles: 'Article 3' (prohibition of torture), 'Article 6 § 1' (fair trial).\n"
            "- Protocol rights: 'Article 1 of Protocol No. 1' (property).\n"
            "- Key doctrines: margin of appreciation, proportionality, positive obligations,\n"
            "  exhaustion of domestic remedies, victim status, standing (locus standi).\n"
            "- Standards: 'beyond reasonable doubt', 'arguable claim', 'necessary in a democratic society'.\n"
            "- Tests: three-part test (prescribed by law, legitimate aim, necessary), Osman test,\n"
            "  Engel criteria, Meier criteria.\n"
            "- Map to kind: Convention articles → statute_article; doctrines → doctrine; tests → test."
        ),
        "issues": (
            "ECHR issue framing:\n"
            "- Issues are framed under 'ALLEGED VIOLATION OF ARTICLE X' headings.\n"
            "- Admissibility issues: 'exhaustion of domestic remedies', 'six-month rule', 'victim status'.\n"
            "- Merits issues: whether there has been a 'violation' or 'interference' with a right.\n"
            "- The Court often re-frames the applicant's complaints under different articles.\n"
            "- Joinder: multiple complaints may be joined under one issue."
        ),
        "arguments": (
            "ECHR argument structure:\n"
            "- 'THE PARTIES' SUBMISSIONS' or 'SUBMISSIONS OF THE PARTIES' sections.\n"
            "- 'The applicant submitted/complained/argued that...' (actor=petitioner).\n"
            "- 'The Government submitted/maintained/contended that...' (actor=respondent).\n"
            "- 'Third-party intervener' observations (actor=third_party).\n"
            "- 'THE COURT'S ASSESSMENT' or 'THE LAW' = court's own analysis (actor=court).\n"
            "- Look for 'general principles' paragraphs (argument scheme=precedent_following).\n"
            "- 'Application of these principles to the present case' = rule_application."
        ),
        "holdings": (
            "ECHR holding extraction:\n"
            "- Holdings appear in the 'FOR THESE REASONS, THE COURT' operative clause.\n"
            "- 'Holds that there has been a violation of Article X' → ratio, answer=yes.\n"
            "- 'Holds that there has been no violation of Article X' → ratio, answer=no.\n"
            "- 'Declares the application admissible/inadmissible' → procedural holding.\n"
            "- 'Holds that the respondent State is to pay the applicant EUR X' → remedy holding.\n"
            "- Unanimity vs dissent noted: 'unanimously' or 'by X votes to Y'.\n"
            "- Just satisfaction under Article 41: pecuniary/non-pecuniary damage, costs."
        ),
        "precedents": (
            "ECHR citation conventions:\n"
            "- Case citations: 'Name v. Country [GC], no. NNNNN/YY, § NN, ECHR YYYY-Vol'.\n"
            "- Paragraph references: '§ 45' or '§§ 45-47'.\n"
            "- Grand Chamber cases marked '[GC]'; decisions marked '[dec.]'.\n"
            "- 'General principles' sections often cite multiple precedents establishing a doctrine.\n"
            "- Look for 'see', 'see also', 'compare', 'cited above' as treatment signals."
        ),
        "outcome": (
            "For ECHR outcomes: if the Court finds a violation (any Article), treat as allowed; "
            "if no violation or the application is inadmissible/struck out, treat as dismissed. "
            "Partly allowed: some articles violated, others not. "
            "Just satisfaction (Art. 41) awards go in relief_summary and directions. "
            "Costs: note if 'costs and expenses' awarded and to whom."
        ),
    },
}

# =============================================================================
# CLUSTERING CALIBRATION PER JURISDICTION
# =============================================================================

# These were tuned empirically:
# - Indian ontology has detailed key_phrases and establishing_cases → phrase matching works well
# - ECHR ontology uses structured Convention article IDs → phrase matching is strong
# - Turkish ontology is sparser, relies more on keyword overlap → lower thresholds needed
CLUSTERING_CALIBRATION: Dict[str, Dict[str, int]] = {
    "in": {
        "cluster_min_keyword_overlap": 2,
        "cluster_phrase_weight": 5,
        "cluster_case_name_weight": 4,
        "cluster_keyword_weight": 1,
        "cluster_min_score_for_assignment": 3,
    },
    "echr": {
        "cluster_min_keyword_overlap": 2,
        "cluster_phrase_weight": 6,       # Convention article phrases are high-signal
        "cluster_case_name_weight": 5,    # ECHR case names are distinctive
        "cluster_keyword_weight": 1,
        "cluster_min_score_for_assignment": 3,
    },
    "tr": {
        "cluster_min_keyword_overlap": 1,  # Turkish ontology is sparser
        "cluster_phrase_weight": 4,        # Turkish phrases may have transliteration variants
        "cluster_case_name_weight": 3,     # Turkish case names less distinctive
        "cluster_keyword_weight": 2,       # Lean more on keywords for Turkish
        "cluster_min_score_for_assignment": 2,  # Lower bar since ontology is sparser
    },
}

DEFAULT_ACTOR_ALIASES: Dict[str, Dict[str, str]] = {
    "echr": {
        "grand_chamber": "court",
        "grandchamber": "court",
        "chamber": "court",
        "court": "court",
        "registry": "court",
        "government": "respondent",
        "respondent_government": "respondent",
        "state": "respondent",
        "respondent_state": "respondent",
        "applicant": "petitioner",
        "applicants": "petitioner",
        "third_party_intervener": "third_party",
        "intervener": "third_party",
        "intervenor": "third_party",
    },
    "tr": {
        "başvurucu": "petitioner",
        "basvurucu": "petitioner",
        "davacı": "petitioner",
        "davaci": "petitioner",
        "sanık": "accused",
        "sanik": "accused",
        "cumhuriyet_savcısı": "prosecution",
        "cumhuriyet_savcisi": "prosecution",
        "idare": "respondent",
        "davalı": "respondent",
        "davali": "respondent",
        "mahkeme": "court",
        "anayasa_mahkemesi": "court",
        "aym": "court",
        "yargıtay": "court",
        "yargitay": "court",
        "danıştay": "court",
        "danistay": "court",
    },
}

# Default section headers for structured judgment segmentation.
DEFAULT_SECTION_HEADERS: Dict[str, List[str]] = {
    "tr": [
        "OLAYLAR",
        "OLAYLAR VE OLGULAR",
        "BAŞVURUCUNUN İDDİALARI",
        "BASVURUCUNUN IDDIALARI",
        "DEĞERLENDİRME",
        "DEGERLENDIRME",
        "GEREKÇE",
        "GEREKCE",
        "HÜKÜM",
        "HUKUM",
        "KARŞIOY GÖRÜŞÜ",
        "KARŞIOY",
        "KARSIOY",
        "UZLAŞMA",
        "SONUÇ",
        "SONUC",
        # Additional common Turkish court headers
        "BAKANLIĞIN GÖRÜŞÜ",
        "BAKANLIGIN GORUSU",
        "İDARENİN SAVUNMASI",
        "IDARENIN SAVUNMASI",
        "KABUL EDİLEBİLİRLİK",
        "KABUL EDILEBILIRLIK",
        "ESAS İNCELEMESİ",
        "ESAS INCELEMESI",
    ],
    "echr": [
        "PROCEDURE",
        "THE FACTS",
        "THE CIRCUMSTANCES OF THE CASE",
        "RELEVANT DOMESTIC LAW",
        "RELEVANT DOMESTIC LAW AND PRACTICE",
        "RELEVANT INTERNATIONAL LAW",
        "RELEVANT INTERNATIONAL MATERIAL",
        "RELEVANT LEGAL FRAMEWORK",
        "THE LAW",
        "ALLEGED VIOLATION",
        "ADMISSIBILITY",
        "MERITS",
        "THE PARTIES' SUBMISSIONS",
        "SUBMISSIONS OF THE PARTIES",
        "THE APPLICANT'S SUBMISSIONS",
        "THE GOVERNMENT'S SUBMISSIONS",
        "THE COURT'S ASSESSMENT",
        "APPLICATION OF ARTICLE 41",
        "JUST SATISFACTION",
        "FOR THESE REASONS",
        "OPERATIVE PROVISIONS",
        "DISSENTING OPINION",
        "PARTLY DISSENTING OPINION",
        "CONCURRING OPINION",
        "JOINT DISSENTING OPINION",
    ],
}

# =============================================================================
# UTILITIES
# =============================================================================

def _sanitize_case_id(case_id: str) -> str:
    """Make a safe filename stem from an arbitrary case id."""
    case_id = str(case_id or "").strip()
    if not case_id:
        return "case"
    # Replace path separators and weird chars
    case_id = case_id.replace("/", "_").replace("\\", "_")
    case_id = re.sub(r"[^0-9A-Za-z._-]+", "_", case_id)
    case_id = re.sub(r"_+", "_", case_id).strip("_")
    return case_id or "case"


def _pick_split(ds, preferred: List[str]):
    """Pick the first available split from a preference list."""
    for name in preferred:
        if name in ds:
            return name, ds[name]
    # Fallback: first split
    first = list(ds.keys())[0]
    return first, ds[first]


def _resolve_ontology_path(candidates: List[str], override: Optional[str] = None) -> Optional[str]:
    """Pick an ontology path that exists locally, or use override if provided."""
    if override:
        p = Path(override)
        return str(p) if p.exists() else None
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return None


def _adapt_case(raw: Dict[str, Any], idx: int, jurisdiction: str) -> Tuple[str, str, Optional[int]]:
    """Convert an arbitrary HF sample into (case_id, text, label).

    This is intentionally heuristic-based. If label cannot be inferred, returns label=None.

    Supports icgcihan/Turkish_Constutional_Court_Decisions fields:
        - Text: 'Metin', 'Karar Metni', 'text', etc.
        - Case ID: 'Başvuru Numarası', 'Esas Sayısı', 'id', etc.
        - Label: 'labels' (0=no violation, 1=violation), 'label', etc.
        - Rights: 'Haklar' (right category — stored but not used as label)
        - URL: 'Kararın Bağlantı Linki'
    """
    # -------- case_id --------
    cid = (
        raw.get("Başvuru Numarası")
        or raw.get("Basvuru Numarasi")
        or raw.get("Esas Sayısı")
        or raw.get("Esas Sayisi")
        or raw.get("id")
        or raw.get("case_id")
        or raw.get("uid")
        or raw.get("doc_id")
        or raw.get("guid")
        or raw.get("citation")
    )
    # icgcihan dataset: extract case number from AYM URL if no explicit ID field
    if cid is None:
        url = raw.get("Kararın Bağlantı Linki") or raw.get("url") or ""
        if "anayasa.gov.tr" in url:
            cid = url.rstrip("/").split("/")[-1]
    if cid is None:
        cid = f"{jurisdiction}_{idx}"
    cid = _sanitize_case_id(str(cid))

    # -------- text --------
    text = (
        raw.get("Metin")
        or raw.get("Karar Metni")
        or raw.get("metin")
        or raw.get("text")
        or raw.get("document")
        or raw.get("judgment")
        or raw.get("content")
        or raw.get("full_text")
        or raw.get("case_text")
    )

    if text is None:
        # Common ECHR-style schemas: facts as a list of paragraphs/sentences
        facts = raw.get("facts")
        if isinstance(facts, list):
            text = "\n".join(str(x) for x in facts if x is not None)
        elif isinstance(facts, str):
            text = facts

    if text is None:
        paras = raw.get("paragraphs") or raw.get("paragraph")
        if isinstance(paras, list):
            text = "\n".join(str(x) for x in paras if x is not None)

    if text is None:
        sents = raw.get("sentences") or raw.get("sentence")
        if isinstance(sents, list):
            text = "\n".join(str(x) for x in sents if x is not None)

    if text is None:
        # Last resort: stringify the entire record (better than crashing)
        text = json.dumps(raw, ensure_ascii=False)

    if isinstance(text, list):
        text = "\n".join(str(x) for x in text if x is not None)
    text = str(text)

    # -------- label (optional) --------
    label: Optional[int] = None
    raw_label = None
    if "label" in raw:
        raw_label = raw.get("label")
    elif "labels" in raw:
        raw_label = raw.get("labels")
    elif "accepted" in raw:
        raw_label = raw.get("accepted")
    elif "outcome" in raw:
        raw_label = raw.get("outcome")

    # Normalize to int 0/1 if possible
    if isinstance(raw_label, bool):
        label = int(raw_label)
    elif isinstance(raw_label, int):
        # If already binary, keep; otherwise drop (unknown mapping)
        label = raw_label if raw_label in (0, 1) else None
    elif isinstance(raw_label, (list, tuple)):
        # Two common conventions:
        # (1) multi-hot vector of 0/1 -> any(1) means positive
        # (2) list of violated article IDs -> non-empty means positive
        if len(raw_label) == 0:
            label = 0
        elif all(isinstance(x, (int, bool)) for x in raw_label):
            # If looks like multi-hot vector, treat any positive as 1.
            label = 1 if any(int(x) for x in raw_label) else 0
        else:
            label = 1  # non-empty list of something -> treat as positive
    elif isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if s in ("1", "true", "yes", "accepted", "allow", "allowed", "violation"):
            label = 1
        elif s in ("0", "false", "no", "rejected", "dismiss", "dismissed", "no_violation"):
            label = 0

    return cid, text, label


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_checkpoint(output_dir: Path, checkpoint_file: Path) -> Tuple[Set[str], Dict]:
    """Load checkpoint of completed case IDs and stats.

    Also scans output directory for existing .json files to recover
    from corrupted checkpoints. Validates each JSON file and removes
    corrupt ones so they get re-processed.
    """
    completed: Set[str] = set()
    stats: Dict[str, Any] = {}

    # First, scan output directory for existing extractions — validate each file
    if output_dir.exists():
        for json_file in output_dir.glob("*.json"):
            if json_file.name == checkpoint_file.name:
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json.load(f)
                completed.add(json_file.stem)
            except (json.JSONDecodeError, Exception):
                print(f"Warning: Corrupt output file {json_file.name}, removing for re-processing")
                try:
                    json_file.unlink()
                except OSError:
                    pass

    # Then try to load checkpoint for stats (but completed already populated from files)
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                completed.update(data.get("completed", []))
                stats = data.get("stats", {})
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load checkpoint ({e}), using file scan instead")

    # Reconstruct stats from output files if stats are empty but files exist
    if completed and (not stats or stats.get("success", 0) == 0):
        print(f"Reconstructing stats from {len(completed)} output files...")
        stats = _reconstruct_stats_from_files(completed, output_dir)

    return completed, stats


def _reconstruct_stats_from_files(completed: Set[str], output_dir: Path) -> Dict:
    """Rebuild stats by scanning output JSON files."""
    stats: Dict[str, Any] = {
        "success": 0,
        "errors": 0,
        "total_facts": 0,
        "total_concepts": 0,
        "total_issues": 0,
        "total_holdings": 0,
        "total_edges": 0,
        "total_chains": 0,
        "outcome_correct": 0,
        "outcome_evaluated": 0,
        "quality_gold": 0,
        "quality_silver": 0,
        "quality_bronze": 0,
        "quality_reject": 0,
    }

    for case_id in completed:
        fpath = output_dir / f"{case_id}.json"
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            stats["success"] += 1
            stats["total_facts"] += len(data.get("facts", []))
            stats["total_concepts"] += len(data.get("concepts", []))
            stats["total_issues"] += len(data.get("issues", []))
            stats["total_holdings"] += len(data.get("holdings", []))
            stats["total_edges"] += len(data.get("edges", []))
            stats["total_chains"] += len(data.get("reasoning_chains", []))

            # quality_tier is stored under _meta in schema_v2_1
            tier = (data.get("_meta") or {}).get("quality_tier", "bronze")
            stats[f"quality_{tier}"] = stats.get(f"quality_{tier}", 0) + 1

        except Exception:
            pass

    return stats


def save_checkpoint(completed: Set[str], stats: Dict, output_dir: Path, checkpoint_file: Path):
    """Save checkpoint atomically via temp file + os.replace()."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp", prefix="checkpoint_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed": [str(c) for c in completed],
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(tmp_path, checkpoint_file)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# EXTRACTION
# =============================================================================

async def extract_one(
    extractor: LegalReasoningExtractor,
    case_id: str,
    text: str,
    label: Optional[int],
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> Dict:
    """Extract graph for one case with per-case timeout."""

    async with semaphore:
        try:
            graph = await asyncio.wait_for(
                extractor.extract(text=text, case_id=case_id),
                timeout=900,  # 15 min per case
            )

            # Defensive: filter out any None values from node lists
            graph.facts = [n for n in graph.facts if n is not None]
            graph.concepts = [n for n in graph.concepts if n is not None]
            graph.issues = [n for n in graph.issues if n is not None]
            graph.arguments = [n for n in graph.arguments if n is not None]
            graph.holdings = [n for n in graph.holdings if n is not None]
            graph.precedents = [n for n in graph.precedents if n is not None]
            graph.justification_sets = [n for n in graph.justification_sets if n is not None]
            graph.edges = [e for e in graph.edges if e is not None]
            graph.reasoning_chains = [rc for rc in graph.reasoning_chains if rc is not None]

            # Save individual graph atomically via temp file + os.replace()
            output_path = output_dir / f"{case_id}.json"
            tmp_path = output_dir / f"{case_id}.json.tmp"
            try:
                json_str = graph.to_json()
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                os.replace(tmp_path, output_path)
            except Exception as ser_err:
                # Clean up temp file on failure
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return {
                    "case_id": case_id,
                    "success": False,
                    "error": f"Serialization error: {str(ser_err)[:150]}",
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

            # Outcome correctness (optional)
            outcome_correct: Optional[bool] = None
            if label in (0, 1):
                if graph.outcome is None:
                    outcome_accepted = False
                    outcome_value = None
                else:
                    outcome_value = graph.outcome.disposition.value
                    outcome_accepted = outcome_value in [
                        "allowed",
                        "partly_allowed",
                        "set_aside",
                        "remanded",
                        "modified",
                    ]
                label_accepted = label == 1
                outcome_correct = outcome_accepted == label_accepted
            else:
                outcome_value = graph.outcome.disposition.value if graph.outcome else None

            return {
                "case_id": case_id,
                "success": True,
                "label": label,
                "outcome": outcome_value,
                "outcome_correct": outcome_correct,
                "quality_tier": graph.quality_tier,
                "n_facts": n_facts,
                "n_concepts": n_concepts,
                "n_issues": n_issues,
                "n_arguments": n_arguments,
                "n_holdings": n_holdings,
                "n_precedents": n_precedents,
                "n_edges": n_edges,
                "n_justification_sets": n_js,
                "n_reasoning_chains": n_chains,
                "warnings": len(graph.validation_warnings),
            }

        except Exception as e:
            return {
                "case_id": case_id,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)[:150]}",
            }


async def run_batch(
    n_cases: int = 50,
    start_idx: int = 0,
    pipeline_version: str = "v4",
    max_concurrent: int = 5,
    dataset_preset: str = "iltur",
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    ontology_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    violations_only: bool = False,
    min_text_length: int = 0,
):
    """Run extraction on a chosen dataset preset (or a fully custom HF dataset).

    If ``local_dir`` is provided (or the preset has no ``hf_dataset``),
    cases are loaded from ``.txt`` files in that directory instead of HuggingFace.
    """

    preset = DATASET_PRESETS.get(dataset_preset, DATASET_PRESETS["iltur"])
    jurisdiction = (jurisdiction or preset.get("jurisdiction") or "in").lower().strip()

    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    # -----------------------------------------------------------------
    # Determine data source: local files vs HuggingFace
    # -----------------------------------------------------------------
    use_local = bool(local_dir) or (preset.get("hf_dataset") is None)

    if use_local and not local_dir:
        print(f"Error: preset '{dataset_preset}' requires --local_dir (no HF dataset configured)")
        return

    print("=" * 70)
    source_label = f"local:{local_dir}" if use_local else dataset_preset
    print(f"LEGAL GRAPH EXTRACTION - dataset={source_label} jurisdiction={jurisdiction} ({pipeline_version.upper()})")
    print("=" * 70)

    if use_local:
        # --- LOCAL FILE MODE ---
        local_path = Path(local_dir)
        if not local_path.is_dir():
            print(f"Error: {local_dir} is not a directory")
            return

        # Collect .txt files sorted by name for deterministic ordering
        txt_files = sorted(local_path.glob("*.txt"))
        if not txt_files:
            print(f"Error: no .txt files found in {local_dir}")
            return

        print(f"\nLoading local files from: {local_dir} ({len(txt_files)} .txt files)")

        # Build a list-of-dicts that _adapt_case can consume
        all_cases_list: List[Dict[str, Any]] = []
        for tf in txt_files:
            try:
                text = tf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = tf.read_text(encoding="latin-1")
            all_cases_list.append({
                "id": tf.stem,           # filename without .txt
                "text": text,
                "label": None,           # no ground truth for scraped data
            })

        total_cases = len(all_cases_list)
        split_name = "local"

    else:
        # --- HUGGINGFACE MODE ---
        if load_dataset is None:
            print("Error: 'datasets' library not installed. Install with: pip install datasets")
            print("       Or use --local_dir to load from local .txt files instead.")
            return

        ds_name = hf_dataset or preset.get("hf_dataset")
        ds_config = hf_config if hf_dataset else preset.get("hf_config")
        split_pref = preset.get("split_preference", ["train"])

        print(f"\nLoading dataset: {ds_name!r} config={ds_config!r} ...")
        try:
            if ds_config:
                ds = load_dataset(ds_name, ds_config)
            else:
                ds = load_dataset(ds_name)
        except Exception as e:
            if dataset_preset == "echr" and ds_name == "lex_glue":
                print(f"Primary ECHR load failed ({type(e).__name__}: {e}). Trying fallback dataset 'ecthr_cases' ...")
                ds = load_dataset("ecthr_cases")
            else:
                raise

        split_name, all_cases_raw = _pick_split(ds, [hf_split] if hf_split else split_pref)
        # Wrap HF dataset rows into plain dicts for uniform handling
        all_cases_list = [dict(all_cases_raw[i]) for i in range(len(all_cases_raw))]
        total_cases = len(all_cases_list)

        # Sniff dataset columns and report for icgcihan-style datasets
        if all_cases_list:
            sample_keys = list(all_cases_list[0].keys())
            print(f"Dataset columns ({len(sample_keys)}): {sample_keys}")

            # Report rights distribution if Haklar field exists
            haklar_key = None
            for k in ("Haklar", "haklar", "rights", "Hak"):
                if k in all_cases_list[0]:
                    haklar_key = k
                    break
            if haklar_key:
                from collections import Counter
                rights_dist = Counter(str(r.get(haklar_key, "?")) for r in all_cases_list)
                print(f"\nRights distribution ({haklar_key}, top 10):")
                for right, count in rights_dist.most_common(10):
                    print(f"  {right}: {count}")

            # Report label distribution
            label_key = None
            for k in ("labels", "label", "Label"):
                if k in all_cases_list[0]:
                    label_key = k
                    break
            if label_key:
                from collections import Counter
                label_dist = Counter(str(r.get(label_key, "?")) for r in all_cases_list)
                print(f"\nLabel distribution ({label_key}): {dict(label_dist)}")

        # Filter violations only if requested
        if violations_only or preset.get("filter_violations_only"):
            before_count = len(all_cases_list)
            all_cases_list = [
                r for r in all_cases_list
                if r.get("labels") == 1 or r.get("label") == 1
            ]
            total_cases = len(all_cases_list)
            print(f"\nFiltered to violations only: {total_cases}/{before_count} cases")

        # Filter by minimum text length if requested
        if min_text_length and min_text_length > 0:
            before_count = len(all_cases_list)
            text_keys = ["text", "Metin", "Karar Metni", "document", "judgment", "content", "full_text"]
            filtered = []
            for r in all_cases_list:
                txt = None
                for k in text_keys:
                    if k in r and r[k]:
                        txt = r[k]
                        break
                if txt and len(str(txt)) >= min_text_length:
                    filtered.append(r)
            all_cases_list = filtered
            total_cases = len(all_cases_list)
            print(f"Filtered by min_text_length >= {min_text_length}: {total_cases}/{before_count} cases")

    print(f"Using split: {split_name} | total cases: {total_cases}")

    # Output paths
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(f"graphs_{dataset_preset}_{pipeline_version}")
    out_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_file = out_dir / "checkpoint.json"

    # Load checkpoint (also scans existing output files)
    completed, stats = load_checkpoint(out_dir, checkpoint_file)
    print(f"Already completed: {len(completed)}")

    # Select cases to process (lazy indexing)
    end_idx = min(start_idx + n_cases, total_cases)
    cases_to_process: List[Tuple[int, str, str, Optional[int]]] = []
    skipped = 0

    for i in range(start_idx, end_idx):
        raw = all_cases_list[i]
        cid, text, label = _adapt_case(raw, i, jurisdiction)
        if cid in completed:
            continue
        if not text or not text.strip():
            skipped += 1
            continue
        cases_to_process.append((i, cid, text, label))

    print(f"Cases to process: {len(cases_to_process)} (skipped empty: {skipped})")

    if not cases_to_process:
        print("All requested cases already processed (or empty)!")
        return

    # Initialize stats if empty
    if not stats:
        stats = {
            "success": 0,
            "errors": 0,
            "total_facts": 0,
            "total_concepts": 0,
            "total_issues": 0,
            "total_holdings": 0,
            "total_edges": 0,
            "total_chains": 0,
            "outcome_correct": 0,
            "outcome_evaluated": 0,
            "quality_gold": 0,
            "quality_silver": 0,
            "quality_bronze": 0,
            "quality_reject": 0,
        }

    # Resolve ontology path
    ontology_candidates = preset.get("ontology_candidates", [])
    resolved_ontology = _resolve_ontology_path(ontology_candidates, override=ontology_path)

    # Prompt context + aliases (defaults are lightweight; safe to keep empty)
    prompt_context = DEFAULT_PROMPT_CONTEXT.get(jurisdiction, {})
    actor_aliases = DEFAULT_ACTOR_ALIASES.get(jurisdiction, {})

    # Create client and extractor
    model_id = "grok-4-1-fast-reasoning"
    client = GrokClient(api_key, model_id=model_id)

    section_headers = DEFAULT_SECTION_HEADERS.get(jurisdiction, [])

    # Clustering calibration (jurisdiction-specific thresholds)
    cluster_cal = CLUSTERING_CALIBRATION.get(jurisdiction, CLUSTERING_CALIBRATION.get("in", {}))

    config = ExtractionConfig(
        model_id=model_id,
        pipeline_version=pipeline_version,
        ontology_path=resolved_ontology,
        jurisdiction=jurisdiction,
        extraction_language_instruction=preset.get("language_instruction"),
        prompt_context=prompt_context,
        actor_aliases=actor_aliases,
        section_headers=section_headers,
        # Clustering calibration
        cluster_min_keyword_overlap=cluster_cal.get("cluster_min_keyword_overlap", 2),
        cluster_phrase_weight=cluster_cal.get("cluster_phrase_weight", 5),
        cluster_case_name_weight=cluster_cal.get("cluster_case_name_weight", 4),
        cluster_keyword_weight=cluster_cal.get("cluster_keyword_weight", 1),
        cluster_min_score_for_assignment=cluster_cal.get("cluster_min_score_for_assignment", 3),
    )

    extractor = LegalReasoningExtractor(client, config)

    print(f"\nPipeline: {pipeline_version}")
    print(f"Model: {model_id}")
    print(f"Ontology: {config.ontology_path or 'None'}")
    print(f"Concurrency: {max_concurrent}")
    print(f"Citation regex: {'enabled' if CitationPreprocessor else 'disabled (citation_preprocess.py not found)'}")
    print(f"Clustering calibration: phrase_w={config.cluster_phrase_weight} "
          f"min_assign={config.cluster_min_score_for_assignment}")
    print(f"Prompt context passes: {list(prompt_context.keys()) if prompt_context else 'none'}")
    print("\n" + "-" * 70)

    # Process
    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = datetime.now()

    for batch_start in range(0, len(cases_to_process), max_concurrent):
        batch = cases_to_process[batch_start : batch_start + max_concurrent]

        tasks = [
            asyncio.create_task(extract_one(extractor, cid, txt, label, semaphore, out_dir))
            for _, cid, txt, label in batch
        ]

        try:
            results = await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Best-effort: also count already-written JSONs as completed
            existing = {p.stem for p in out_dir.glob("*.json") if p.name != checkpoint_file.name}
            completed.update(existing)
            save_checkpoint(completed, stats, out_dir, checkpoint_file)
            await client.close()

            print("Interrupted. Checkpoint saved. Exiting cleanly.")
            return

        for result in results:
            if result["success"]:
                completed.add(result["case_id"])
                stats["success"] += 1
                stats["total_facts"] += result["n_facts"]
                stats["total_concepts"] += result["n_concepts"]
                stats["total_issues"] += result["n_issues"]
                stats["total_holdings"] += result["n_holdings"]
                stats["total_edges"] += result["n_edges"]
                stats["total_chains"] += result["n_reasoning_chains"]

                if result["outcome_correct"] is not None:
                    stats["outcome_evaluated"] += 1
                    if result["outcome_correct"]:
                        stats["outcome_correct"] += 1

                tier = result["quality_tier"]
                stats[f"quality_{tier}"] = stats.get(f"quality_{tier}", 0) + 1

                oc = (
                    "✓"
                    if result["outcome_correct"] is True
                    else ("✗" if result["outcome_correct"] is False else "-")
                )
                print(
                    f"[{stats['success']:3d}] {result['case_id']}: "
                    f"{oc} {result['quality_tier']:6s} | "
                    f"F:{result['n_facts']:2d} C:{result['n_concepts']:2d} "
                    f"H:{result['n_holdings']:2d} E:{result['n_edges']:2d} "
                    f"chains:{result['n_reasoning_chains']}"
                )
            else:
                stats["errors"] += 1
                print(f"[ERR] {result['case_id']}: {result['error']}")

        save_checkpoint(completed, stats, out_dir, checkpoint_file)

        batch_num = batch_start // max_concurrent + 1
        total_batches = (len(cases_to_process) + max_concurrent - 1) // max_concurrent
        print(
            f"  [batch {batch_num}/{total_batches}] "
            f"tokens: {client.total_prompt_tokens + client.total_completion_tokens:,} "
            f"({client.total_requests} requests)"
        )

    await client.close()

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Processed: {stats['success']} | Errors: {stats['errors']}")
    print(f"Time: {elapsed:.1f}s ({elapsed / max(stats['success'], 1):.1f}s per case)")

    if stats["success"] > 0:
        print(f"\n📊 QUALITY:")
        print(f"  Gold:   {stats.get('quality_gold', 0)}")
        print(f"  Silver: {stats.get('quality_silver', 0)}")
        print(f"  Bronze: {stats.get('quality_bronze', 0)}")
        print(f"  Reject: {stats.get('quality_reject', 0)}")

        print(f"\n📈 AVERAGES:")
        n = stats["success"]
        print(f"  Facts/case:    {stats['total_facts'] / n:.1f}")
        print(f"  Concepts/case: {stats['total_concepts'] / n:.1f}")
        print(f"  Holdings/case: {stats['total_holdings'] / n:.1f}")
        print(f"  Edges/case:    {stats['total_edges'] / n:.1f}")
        print(f"  Chains/case:   {stats['total_chains'] / n:.1f}")

        if stats.get("outcome_evaluated", 0) > 0:
            ev = stats["outcome_evaluated"]
            print(f"\n🎯 OUTCOME PREDICTION:")
            print(f"  Correct: {stats['outcome_correct']}/{ev} ({stats['outcome_correct'] / ev * 100:.1f}%)")
        else:
            print(f"\n🎯 OUTCOME PREDICTION: (not evaluated — no binary labels detected)")

    print(f"\n💰 TOKEN USAGE:")
    print(f"  Prompt tokens:     {client.total_prompt_tokens:,}")
    print(f"  Completion tokens: {client.total_completion_tokens:,}")
    print(f"  Total tokens:      {client.total_prompt_tokens + client.total_completion_tokens:,}")
    print(f"  API requests:      {client.total_requests:,}")

    print(f"\n✅ Outputs saved to: {out_dir}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run legal graph extractor on HF datasets")
    parser.add_argument("--n", type=int, default=50, help="Number of cases to process")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--version", choices=["v3", "v4"], default="v4", help="Pipeline version")
    parser.add_argument("--concurrent", type=int, default=20, help="Max concurrent extractions")

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PRESETS.keys()),
        default="iltur",
        help="Dataset preset (you can override with --hf_dataset/--hf_config/--hf_split)",
    )
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name (override preset)")
    parser.add_argument("--hf_config", type=str, default=None, help="HuggingFace dataset config (override preset)")
    parser.add_argument("--hf_split", type=str, default=None, help="Dataset split (override preset default)")
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Path to local directory of .txt files (bypasses HuggingFace). Each file = one case.",
    )
    parser.add_argument(
        "--jurisdiction",
        type=str,
        default=None,
        help="Jurisdiction id used for prompting (e.g., in, tr, echr). Defaults from preset.",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=None,
        help="Path to ontology JSON (override preset default)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: graphs_<dataset>_<version>)",
    )
    parser.add_argument(
        "--violations_only",
        action="store_true",
        default=False,
        help="Only process cases where labels=1 (violation found). Useful for AYM datasets.",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=0,
        help="Skip cases with text shorter than this (chars). Recommended: 5000 for AYM.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_batch(
            n_cases=args.n,
            start_idx=args.start,
            pipeline_version=args.version,
            max_concurrent=args.concurrent,
            dataset_preset=args.dataset,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            jurisdiction=args.jurisdiction,
            ontology_path=args.ontology,
            output_dir=args.output_dir,
            local_dir=args.local_dir,
            violations_only=args.violations_only,
            min_text_length=args.min_text_length,
        )
    )


if __name__ == "__main__":
    main()