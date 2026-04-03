#!/usr/bin/env python3
"""
citation_preprocess.py

Deterministic citation extraction via regex for Indian, ECHR, and Turkish
legal judgments.  Runs BEFORE the LLM precedents pass to:

  1. Pre-tag citations with exact char offsets (eliminates LLM offset drift).
  2. Feed a citation manifest into the precedents prompt so the LLM only needs
     to determine treatment / relevance, not *find* the citation string.
  3. Provide an accuracy floor the LLM can only improve upon.

Usage
-----
    from citation_preprocess import CitationPreprocessor

    cpp = CitationPreprocessor(jurisdiction="echr")
    hits = cpp.extract(text)
    # hits = [{"citation": "...", "start_char": ..., "end_char": ..., "citation_type": "..."}, ...]
    manifest = cpp.build_prompt_manifest(hits)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Pattern

# =============================================================================
# CITATION HIT
# =============================================================================

@dataclass
class CitationHit:
    """A single regex-detected citation span."""
    citation: str          # The matched text (cleaned)
    start_char: int        # Offset in original document
    end_char: int          # Offset in original document
    citation_type: str     # Category label (e.g. "air", "scc", "echr_appno")
    case_name: Optional[str] = None     # Extracted name if available
    case_year: Optional[int] = None     # Extracted year if available
    normalized_id: Optional[str] = None # Stable ID for dedup

    def to_dict(self) -> Dict:
        return {
            "citation": self.citation,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "citation_type": self.citation_type,
            "case_name": self.case_name,
            "case_year": self.case_year,
        }


# =============================================================================
# INDIAN CITATION PATTERNS
# =============================================================================

# AIR 2005 SC 3220, AIR 1976 SC 1207
_IN_AIR = re.compile(
    r'\bAIR\s+(\d{4})\s+(SC|Del|Bom|Mad|Cal|All|Ker|Kar|Pat|P&H|Guj|AP|HP|J&K|Ori|Raj|MP|Gau|Tri|NOC)\s+\d+',
    re.IGNORECASE
)

# (2018) 5 SCC 1, (2003) 4 SCC 601
_IN_SCC_PAREN = re.compile(
    r'\(\d{4}\)\s+\d{1,2}\s+SCC\s+\d+',
    re.IGNORECASE
)

# 2019 SCC OnLine SC 1234, 2020 SCC OnLine Del 456
_IN_SCC_ONLINE = re.compile(
    r'\b\d{4}\s+SCC\s+OnLine\s+(?:SC|Del|Bom|Mad|Cal|All|Ker|Kar|Pat|P[\s&]*H|Guj|AP|HP|J[\s&]*K|Ori|Raj|MP|Gau|Tri|Chh|Utt|Jhar)\s+\d+',
    re.IGNORECASE
)

# (2005) 2 SCR 1, 1978 SCR (2) 621
_IN_SCR = re.compile(
    r'(?:\(\d{4}\)\s+\d{1,2}\s+SCR\s+\d+|\d{4}\s+SCR\s*\(\d{1,2}\)\s+\d+)',
    re.IGNORECASE
)

# Crl.A. No. 1234 of 2019, W.P.(C) No. 567 of 2020, SLP (Crl.) No. 789 of 2021
_IN_CASE_NUMBER = re.compile(
    r'\b(?:Crl\.?\s*A\.?|W\.?P\.?\s*\(?\s*(?:C|Crl)\.?\s*\)?|SLP\s*\(?\s*(?:C|Crl)\.?\s*\)?|C\.?A\.?|T\.?C\.?\s*\(?\s*C\.?\s*\)?|M\.?A\.?)\s*(?:No\.?\s*)?\d+\s+of\s+\d{4}',
    re.IGNORECASE
)

# MANU/SC/1234/2019, MANU/DE/0567/2020
_IN_MANU = re.compile(
    r'\bMANU/[A-Z]{2}/\d+/\d{4}',
    re.IGNORECASE
)

# ILR 1982 KAR 1234
_IN_ILR = re.compile(
    r'\bILR\s+\d{4}\s+[A-Z]{2,4}\s+\d+',
    re.IGNORECASE
)

# Section references (often cited as precedent-adjacent): S. 302 IPC, Section 498-A
_IN_SECTION = re.compile(
    r'\b(?:Section|S\.)\s+\d+[\-A-Z]?\s+(?:of\s+(?:the\s+)?)?(?:IPC|CrPC|Cr\.P\.C\.|CPC|C\.P\.C\.|Constitution|Evidence\s+Act|IT\s+Act|NDPS\s+Act)',
    re.IGNORECASE
)


INDIAN_PATTERNS: List[Tuple[Pattern, str]] = [
    (_IN_AIR, "air"),
    (_IN_SCC_PAREN, "scc"),
    (_IN_SCC_ONLINE, "scc_online"),
    (_IN_SCR, "scr"),
    (_IN_MANU, "manu"),
    (_IN_ILR, "ilr"),
    (_IN_CASE_NUMBER, "case_number"),
]


# =============================================================================
# ECHR CITATION PATTERNS
# =============================================================================

# Application no. 36022/97, application nos. 36022/97 and 36023/97
_ECHR_APPNO = re.compile(
    r'\b[Aa]pplication\s+nos?\.?\s+(\d{1,6}/\d{2,4})(?:\s+and\s+\d{1,6}/\d{2,4})*',
)

# Bare app numbers in ECHR context: no. 12345/06
_ECHR_BARE_APPNO = re.compile(
    r'\bnos?\.?\s+(\d{1,6}/\d{2,4})',
)

# Selmouni v. France [GC], no. 25803/94, Handyside v. the United Kingdom
# Pattern: Name v. Name [optional tags]
# Handles Mc/Mac/O' prefixes, internal capitals, "and Others", multi-word country names
_ECHR_NAME = r'[A-ZÀ-Ž][A-Za-zÀ-žà-ž\'\-]+'  # Allows internal capitals (McCann, etc.)
_ECHR_CASE_V = re.compile(
    rf'({_ECHR_NAME}(?:\s+(?:and|et)\s+(?:Others|Autres|{_ECHR_NAME}))*)\s+v\.?\s+'
    rf'((?:the\s+)?{_ECHR_NAME}(?:\s+{_ECHR_NAME})*)'
    r'\s*(?:\[(?:GC|dec\.|comm\.)\])?',
)

# § 45, §§ 45-47, paragraphs 45-50
_ECHR_PARA_REF = re.compile(
    r'(?:§§?\s*\d+(?:\s*[-–]\s*\d+)?|paragraphs?\s+\d+(?:\s*[-–]\s*\d+)?)',
    re.IGNORECASE
)

# Series A no. 24, Reports 1999-VII
_ECHR_SERIES = re.compile(
    r'\b(?:Series\s+A\s+no\.?\s*\d+|Reports?\s+(?:of\s+Judgments\s+and\s+Decisions\s+)?\d{4}(?:\-[IVXLCDM]+)?)',
    re.IGNORECASE
)

# ECHR 2003-XI, ECHR 2014 (extracts)
_ECHR_REPORT_YEAR = re.compile(
    r'\bECHR\s+\d{4}(?:\s*[-–]\s*[IVXLCDM]+)?(?:\s*\(extracts?\))?',
    re.IGNORECASE
)

# Article 3 of the Convention, Articles 6 § 1 and 13
_ECHR_ARTICLE = re.compile(
    r'\bArticles?\s+\d+(?:\s*§\s*\d+)?(?:\s+(?:of\s+(?:the\s+)?(?:Convention|Protocol(?:\s+No\.?\s*\d+)?)))?',
    re.IGNORECASE
)

# Protocol No. 1, Protocol No. 12
_ECHR_PROTOCOL = re.compile(
    r'\bProtocol\s+No\.?\s*\d+(?:\s+to\s+the\s+Convention)?',
    re.IGNORECASE
)

ECHR_PATTERNS: List[Tuple[Pattern, str]] = [
    (_ECHR_APPNO, "echr_appno"),
    (_ECHR_CASE_V, "echr_case_v"),
    (_ECHR_SERIES, "echr_series"),
    (_ECHR_REPORT_YEAR, "echr_report"),
    (_ECHR_BARE_APPNO, "echr_bare_appno"),
]


# =============================================================================
# TURKISH CITATION PATTERNS
# =============================================================================

# AYM E.2018/123, K.2019/456 (Anayasa Mahkemesi / Constitutional Court)
_TR_AYM_EK = re.compile(
    r'\b(?:AYM|Anayasa\s+Mahkemesi)\s*[,;]?\s*E\.?\s*(\d{4})/(\d+)\s*[,;]\s*K\.?\s*(\d{4})/(\d+)',
    re.IGNORECASE
)

# Esas numarası patterns: E. 2018/12345, 2018/12345 E.
_TR_ESAS = re.compile(
    r'\b(?:E\.?\s*(\d{4})/(\d+)|(\d{4})/(\d+)\s*E\.)',
    re.IGNORECASE
)

# Karar numarası: K. 2019/6789, 2019/6789 K.
_TR_KARAR = re.compile(
    r'\b(?:K\.?\s*(\d{4})/(\d+)|(\d{4})/(\d+)\s*K\.)',
    re.IGNORECASE
)

# Combined E., K. pattern: 2018/12345 E., 2019/6789 K.
_TR_EK_COMBINED = re.compile(
    r'(\d{4})/(\d+)\s*E\.\s*[,;]\s*(\d{4})/(\d+)\s*K\.',
    re.IGNORECASE
)

# Yargıtay (Court of Cassation) patterns
# Yargıtay 4. Ceza Dairesi, E. 2017/1234, K. 2018/5678
_TR_YARGITAY = re.compile(
    r'\bYarg[ıi]tay\s+(?:\d+\.\s*)?(?:Ceza|Hukuk|Daire)\s*(?:si|Dairesi)?\s*[,;]?\s*(?:E\.?\s*\d{4}/\d+)?',
    re.IGNORECASE
)

# Danıştay (Council of State) patterns
_TR_DANISTAY = re.compile(
    r'\bDan[ıi][şs]tay\s+(?:\d+\.\s*)?(?:Daire|İdari\s+Dava)\s*(?:si|Dairesi)?\s*[,;]?\s*(?:E\.?\s*\d{4}/\d+)?',
    re.IGNORECASE
)

# Resmi Gazete (Official Gazette) references
_TR_RG = re.compile(
    r'\b(?:Resm[iî]\s+Gazete|R\.?\s*G\.?)\s*[,:;]?\s*(?:tarih|say[ıi])?\s*[,:;]?\s*\d+[./]\d+[./]?\d*',
    re.IGNORECASE
)

# Kanun references: 5237 sayılı Kanun, 6100 sayılı HMK
_TR_KANUN = re.compile(
    r'\b(\d{3,5})\s+say[ıi]l[ıi]\s+(?:Kanun|(?:T\.?)?(?:C\.?)?K\.?|HMK|CMK|TMK|TTK|[A-ZÇĞİÖŞÜ]{2,5})',
    re.IGNORECASE
)

# Anayasa article references: Anayasa'nın 17. maddesi, AY m. 36
_TR_ANAYASA_ART = re.compile(
    r'\b(?:Anayasa[\'ʼ]?n[ıi]n\s+(\d+)\.?\s*madde|AY\s+m\.?\s*(\d+))',
    re.IGNORECASE
)

# Generic Turkish court date citation: 12.03.2018 tarihli
_TR_DATE_REF = re.compile(
    r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\s+tarihli',
    re.IGNORECASE
)

# Başvuru numarası (application number for AYM individual complaints)
_TR_BASVURU = re.compile(
    r'\b(?:[Bb]a[şs]vuru\s+(?:numaras[ıi]|[Nn]o\.?))\s*[,:;]?\s*(\d{4}/\d+)',
    re.IGNORECASE
)

TURKISH_PATTERNS: List[Tuple[Pattern, str]] = [
    (_TR_AYM_EK, "tr_aym"),
    (_TR_EK_COMBINED, "tr_ek_combined"),
    (_TR_YARGITAY, "tr_yargitay"),
    (_TR_DANISTAY, "tr_danistay"),
    (_TR_BASVURU, "tr_basvuru"),
    (_TR_RG, "tr_resmi_gazete"),
    (_TR_KANUN, "tr_kanun"),
    (_TR_ESAS, "tr_esas"),
    (_TR_KARAR, "tr_karar"),
]


# =============================================================================
# PATTERN REGISTRY
# =============================================================================

JURISDICTION_PATTERNS: Dict[str, List[Tuple[Pattern, str]]] = {
    "in": INDIAN_PATTERNS,
    "india": INDIAN_PATTERNS,
    "echr": ECHR_PATTERNS,
    "tr": TURKISH_PATTERNS,
    "turkey": TURKISH_PATTERNS,
}


# =============================================================================
# CITATION PREPROCESSOR
# =============================================================================

class CitationPreprocessor:
    """Deterministic citation extraction via regex.

    Usage::

        cpp = CitationPreprocessor(jurisdiction="echr")
        hits = cpp.extract(text)
        manifest = cpp.build_prompt_manifest(hits)
    """

    def __init__(self, jurisdiction: str = "in"):
        self.jurisdiction = jurisdiction.lower().strip()
        self.patterns = JURISDICTION_PATTERNS.get(self.jurisdiction, [])

    def extract(self, text: str) -> List[CitationHit]:
        """Scan *text* and return all citation hits with char offsets."""
        if not text or not self.patterns:
            return []

        raw_hits: List[CitationHit] = []

        for pattern, ctype in self.patterns:
            for m in pattern.finditer(text):
                citation_text = m.group(0).strip()
                start = m.start()
                end = m.end()

                # Extract year if present in the match
                year = self._extract_year(citation_text)

                # Extract case name for ECHR v. patterns
                case_name = None
                if ctype == "echr_case_v":
                    try:
                        case_name = f"{m.group(1)} v. {m.group(2)}"
                    except (IndexError, AttributeError):
                        pass

                hit = CitationHit(
                    citation=citation_text,
                    start_char=start,
                    end_char=end,
                    citation_type=ctype,
                    case_name=case_name,
                    case_year=year,
                )
                raw_hits.append(hit)

        # Deduplicate overlapping spans (keep longest)
        return self._dedupe_overlapping(raw_hits)

    def build_prompt_manifest(self, hits: List[CitationHit], max_items: int = 50) -> str:
        """Format citation hits as a prompt block for the LLM precedents pass.

        The LLM only needs to determine treatment/relevance/proposition — not
        *find* the citation text or compute offsets.
        """
        if not hits:
            return ""

        lines = [
            "PRE-DETECTED CITATIONS (regex, with verified char offsets):",
            "Use these as anchors. You may add citations the regex missed, but",
            "for the ones below, USE THE PROVIDED start_char/end_char exactly.",
            ""
        ]
        for i, h in enumerate(hits[:max_items]):
            lines.append(
                f"  [{i+1}] \"{h.citation}\" "
                f"(start_char={h.start_char}, end_char={h.end_char}, type={h.citation_type}"
                + (f", case_name=\"{h.case_name}\"" if h.case_name else "")
                + (f", year={h.case_year}" if h.case_year else "")
                + ")"
            )

        if len(hits) > max_items:
            lines.append(f"  ... and {len(hits) - max_items} more")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Try to pull a 4-digit year from a citation string."""
        m = re.search(r'\b((?:19|20)\d{2})\b', text)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _dedupe_overlapping(hits: List[CitationHit]) -> List[CitationHit]:
        """Remove overlapping spans, keeping the longer match."""
        if not hits:
            return []

        # Sort by start, then by -length (prefer longer)
        sorted_hits = sorted(hits, key=lambda h: (h.start_char, -(h.end_char - h.start_char)))

        result: List[CitationHit] = []
        last_end = -1

        for h in sorted_hits:
            if h.start_char >= last_end:
                result.append(h)
                last_end = h.end_char
            else:
                # Overlapping — keep existing (it's longer due to sort)
                pass

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def preprocess_citations(text: str, jurisdiction: str = "in") -> List[CitationHit]:
    """One-shot convenience wrapper."""
    return CitationPreprocessor(jurisdiction).extract(text)


def build_citation_manifest(text: str, jurisdiction: str = "in") -> str:
    """One-shot: extract + format for prompt injection."""
    cpp = CitationPreprocessor(jurisdiction)
    hits = cpp.extract(text)
    return cpp.build_prompt_manifest(hits)