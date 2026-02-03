#!/usr/bin/env python3
"""
test_extraction.py

Test script for the Legal Reasoning Graph extraction system.
"""

import asyncio
import json
import re
from extractor import (
    LegalReasoningExtractor,
    MockLLMClient,
    ExtractionConfig,
    segment_document
)

# Sample judgment text for testing
SAMPLE_JUDGMENT = """
IN THE SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION

CIVIL APPEAL NO. 1234 OF 2024

XYZ Corporation Ltd.                         ... Appellant
vs.
Union of India & Ors.                        ... Respondents

JUDGMENT

Delivered by: Hon'ble Justice A. Kumar

1. This appeal arises from a decision of the High Court of Delhi dismissing the writ petition filed by the appellant challenging the order dated 15.03.2024 passed by the respondent authorities under Section 10 of the Environmental Protection Act, 1986.

FACTUAL BACKGROUND

2. The appellant is a company engaged in manufacturing of chemicals at its factory situated in Noida, Uttar Pradesh. The appellant has been operating the said factory since 2015 under valid licenses and permissions from the State Pollution Control Board.

3. On 10.01.2024, officials from the Central Pollution Control Board conducted an inspection of the appellant's factory. The inspection report dated 15.01.2024 noted certain alleged violations of emission norms.

4. Pursuant to the said inspection, the respondent No. 2 issued a show cause notice dated 20.01.2024 to the appellant calling upon it to show cause why the factory should not be shut down for violation of environmental norms.

5. The appellant submitted a detailed reply dated 05.02.2024 disputing the findings of the inspection report and providing technical data demonstrating compliance with prescribed emission standards.

6. Without considering the appellant's reply or granting an opportunity of personal hearing, the respondent No. 2 passed the impugned order dated 15.03.2024 directing immediate closure of the appellant's factory.

PROCEEDINGS BEFORE HIGH COURT

7. Aggrieved by the said order, the appellant filed Writ Petition (Civil) No. 5678 of 2024 before the High Court of Delhi under Article 226 of the Constitution.

8. The appellant's primary contention before the High Court was that the closure order was passed in violation of principles of natural justice, inasmuch as no opportunity of personal hearing was granted before passing such a drastic order affecting the appellant's fundamental right to carry on business.

9. The respondents contended that the Environmental Protection Act does not mandate a personal hearing and that the statutory scheme provides a complete procedure that was duly followed.

10. The High Court, by its judgment dated 25.05.2024, dismissed the writ petition holding that:
    (a) The statutory scheme under the Environmental Protection Act is self-contained;
    (b) The Act does not expressly require a personal hearing;
    (c) The written reply submitted by the appellant was sufficient compliance with natural justice.

ISSUES FOR DETERMINATION

11. The following issues arise for our consideration:
    (i) Whether the right to carry on business is a facet of personal liberty under Article 21 of the Constitution?
    (ii) Whether the closure order passed without granting personal hearing violates principles of natural justice?
    (iii) Whether the Environmental Protection Act excludes the requirement of natural justice by necessary implication?

ANALYSIS AND DISCUSSION

Issue (i): Article 21 and Right to Business

12. Article 21 of the Constitution guarantees that no person shall be deprived of his life or personal liberty except according to procedure established by law. This Court has consistently held that the right to livelihood is an integral part of the right to life.

13. In Olga Tellis v. Bombay Municipal Corporation, (1985) 3 SCC 545, this Court held:
    "The sweep of the right to life conferred by Article 21 is wide and far reaching. It does not mean merely that life cannot be extinguished or taken away as, for example, by the imposition and execution of the death sentence, except according to procedure established by law. That is but one aspect of the right to life. An equally important facet of that right is the right to livelihood because, no person can live without the means of living, that is, the means of livelihood."

14. In Board of Trustees of the Port of Bombay v. Dilipkumar Raghavendranath Nadkarni, (1983) 1 SCC 124, this Court extended this principle to include the right to carry on trade or business as part of personal liberty.

15. We hold that the right to carry on business or trade is undoubtedly a facet of personal liberty under Article 21. Any action of the State affecting such right must conform to the requirements of Article 21, including adherence to fair procedure.

Issue (ii): Violation of Natural Justice

16. The principles of natural justice, particularly audi alteram partem (hear the other side), are fundamental to any civilized jurisprudence. These principles require that no person should be condemned unheard.

17. In Maneka Gandhi v. Union of India, AIR 1978 SC 597, this Court revolutionized the understanding of Article 21 by holding that the procedure established by law must be right, just and fair, and not arbitrary, fanciful or oppressive. The principles of natural justice were held to be implicit in Article 21.

18. The Court observed:
    "The procedure prescribed by law has to be fair, just and reasonable, not fanciful, oppressive or arbitrary. When the procedure prescribed by a statute for depriving a person of his fundamental right is examined from this standpoint, it has to pass the test laid down in Article 14 also."

19. In the present case, the impugned order directs closure of a factory which has been operating for nearly a decade. Such closure affects not only the appellant company but also hundreds of workers employed therein. An order of this magnitude and consequence cannot be passed without affording an opportunity of personal hearing.

20. The written reply submitted by the appellant cannot be treated as sufficient compliance with the rule of audi alteram partem in a case involving drastic consequences such as factory closure. A personal hearing would have enabled the appellant to explain the technical aspects of its reply and clarify any doubts the authorities may have had.

21. We find that the closure order dated 15.03.2024 was passed in clear violation of the principles of natural justice.

Issue (iii): Statutory Scheme and Natural Justice

22. The respondents have relied upon the principle that where a statute provides a complete procedure, principles of natural justice may be excluded by necessary implication.

23. In State of U.P. v. Mohammad Nooh, AIR 1958 SC 86, this Court held that procedural requirements prescribed by statute should be strictly followed, and in some circumstances, express statutory procedure may exclude additional requirements of natural justice.

24. However, this principle has significant limitations. In S.L. Kapoor v. Jagmohan, (1980) 4 SCC 379, this Court clarified:
    "The rules of natural justice are not rules embodied always expressly in a statute or in rules framed thereunder. They may be implied from the nature of the duty to be performed and cannot be excluded by implication. They operate as implied mandatory requirements, non-observance of which invalidates the order."

25. The Environmental Protection Act, 1986 does not contain any provision expressly excluding the requirement of personal hearing before passing a closure order. The mere absence of an express provision requiring personal hearing cannot be construed as exclusion of natural justice by necessary implication.

26. We distinguish the decision of this Court in State of U.P. v. Mohammad Nooh (supra) on the ground that in that case, the statute expressly prescribed a different procedure which was held to be sufficient. In the present case, there is no such express alternative procedure that would justify exclusion of personal hearing.

27. We hold that the Environmental Protection Act does not exclude principles of natural justice by necessary implication. The requirement of personal hearing before passing an order of factory closure remains applicable.

CONCLUSION

28. For the reasons stated above, we hold that:
    (a) The right to carry on business is a facet of personal liberty under Article 21;
    (b) The closure order dated 15.03.2024 was passed in violation of principles of natural justice;
    (c) The Environmental Protection Act does not exclude the requirement of personal hearing before factory closure.

29. Accordingly, the appeal is allowed. The judgment of the High Court dated 25.05.2024 is set aside. The closure order dated 15.03.2024 passed by respondent No. 2 is quashed.

30. The matter is remanded to respondent No. 2 to pass a fresh order in accordance with law after granting opportunity of personal hearing to the appellant. The respondent authorities shall complete this exercise within a period of three months from today.

31. There shall be no order as to costs.

32. All pending applications are disposed of.

.........................J.
(A. Kumar)

New Delhi
July 15, 2024
"""


class SmartMockClient(MockLLMClient):
    """A smarter mock client that returns realistic extraction results."""

    async def complete(self, prompt: str, system: str, temperature: float = 0.1,
                       max_tokens: int = 4096, json_mode: bool = True) -> str:
        self.call_log.append({"prompt": prompt[:500], "system": system[:200]})

        # Check PROMPT for keywords - use unique starting phrases to avoid overlap
        prompt_lower = prompt.lower()

        # Use more specific patterns based on the unique START of each prompt
        # Order matters - check more specific patterns first

        # Holdings prompt starts with "Extract HOLDINGS (legal determinations)"
        if prompt_lower.startswith("extract all holdings") or prompt_lower.startswith("extract holdings"):
            return json.dumps({
                "holdings": [
                    {
                        "id": "h1",
                        "text": "The right to carry on business is a facet of personal liberty under Article 21",
                        "start_char": 4000,
                        "end_char": 4150,
                        "surface_text": "We hold that the right to carry on business or trade is undoubtedly a facet of personal liberty...",
                        "resolves_issue": "i1",
                        "is_ratio": True,
                        "novel": False,
                        "reasoning_summary": "Following Olga Tellis and Board of Trustees precedents",
                        "schemes": ["precedent_following"],
                        "confidence": "high"
                    },
                    {
                        "id": "h2",
                        "text": "The closure order was passed in violation of principles of natural justice",
                        "start_char": 5200,
                        "end_char": 5400,
                        "surface_text": "We find that the closure order dated 15.03.2024 was passed in clear violation of the principles of natural justice",
                        "resolves_issue": "i2",
                        "is_ratio": True,
                        "novel": False,
                        "reasoning_summary": "Following Maneka Gandhi, personal hearing required for drastic orders",
                        "schemes": ["precedent_following", "procedural"],
                        "confidence": "high"
                    }
                ]
            })

        # Facts prompt starts with "Extract FACTS"
        elif prompt_lower.startswith("extract all facts") or prompt_lower.startswith("extract facts"):
            return json.dumps({
                "facts": [
                    {
                        "id": "f1",
                        "text": "The appellant is a company engaged in manufacturing of chemicals at its factory in Noida",
                        "start_char": 500,
                        "end_char": 620,
                        "surface_text": "The appellant is a company engaged in manufacturing...",
                        "fact_type": "material",
                        "actor_source": None,
                        "date": None,
                        "confidence": "high"
                    },
                    {
                        "id": "f2",
                        "text": "Officials conducted an inspection on 10.01.2024 noting alleged violations of emission norms",
                        "start_char": 780,
                        "end_char": 950,
                        "surface_text": "On 10.01.2024, officials from the Central Pollution Control Board...",
                        "fact_type": "material",
                        "date": "2024-01-10",
                        "confidence": "high"
                    },
                    {
                        "id": "f3",
                        "text": "Closure order dated 15.03.2024 was passed without granting personal hearing",
                        "start_char": 1500,
                        "end_char": 1680,
                        "surface_text": "Without considering the appellant's reply or granting an opportunity...",
                        "fact_type": "material",
                        "date": "2024-03-15",
                        "confidence": "high"
                    },
                    {
                        "id": "f4",
                        "text": "Appellant submitted detailed reply dated 05.02.2024",
                        "start_char": 1300,
                        "end_char": 1450,
                        "surface_text": "The appellant submitted a detailed reply dated 05.02.2024...",
                        "fact_type": "procedural",
                        "date": "2024-02-05",
                        "confidence": "high"
                    }
                ]
            })

        # Concepts prompt starts with "Extract LEGAL CONCEPTS"
        elif prompt_lower.startswith("extract all legal concepts") or prompt_lower.startswith("extract legal concepts"):
            return json.dumps({
                "concepts": [
                    {
                        "id": "c1",
                        "concept_id": "CONST_ART21",
                        "start_char": 3200,
                        "end_char": 3400,
                        "surface_text": "Article 21 of the Constitution guarantees that no person shall be deprived...",
                        "relevance": "central",
                        "kind": "statute_article",
                        "interpretation": "Right to livelihood and business is part of personal liberty",
                        "interpretation_start_char": 3500,
                        "interpretation_end_char": 3700,
                        "confidence": "high"
                    },
                    {
                        "id": "c2",
                        "concept_id": "DOCTRINE_NATURAL_JUSTICE",
                        "start_char": 4200,
                        "end_char": 4450,
                        "surface_text": "The principles of natural justice, particularly audi alteram partem...",
                        "relevance": "central",
                        "kind": "doctrine",
                        "interpretation": "No person should be condemned unheard",
                        "confidence": "high"
                    },
                    {
                        "id": "c3",
                        "concept_id": "CONST_ART226",
                        "start_char": 2100,
                        "end_char": 2250,
                        "surface_text": "the appellant filed Writ Petition (Civil) No. 5678 of 2024 before the High Court of Delhi under Article 226",
                        "relevance": "supporting",
                        "kind": "statute_article",
                        "confidence": "high"
                    }
                ]
            })

        # Issues prompt starts with "Extract LEGAL ISSUES"
        elif prompt_lower.startswith("extract all legal issues") or prompt_lower.startswith("extract legal issues"):
            return json.dumps({
                "issues": [
                    {
                        "id": "i1",
                        "text": "Whether the right to carry on business is a facet of personal liberty under Article 21?",
                        "start_char": 2800,
                        "end_char": 2950,
                        "surface_text": "(i) Whether the right to carry on business is a facet...",
                        "issue_number": 1,
                        "framed_by": "court",
                        "primary_concepts": ["c1"],
                        "answer": "yes",
                        "confidence": "high"
                    },
                    {
                        "id": "i2",
                        "text": "Whether the closure order passed without granting personal hearing violates principles of natural justice?",
                        "start_char": 2960,
                        "end_char": 3100,
                        "surface_text": "(ii) Whether the closure order passed without granting personal hearing...",
                        "issue_number": 2,
                        "framed_by": "court",
                        "primary_concepts": ["c2"],
                        "answer": "yes",
                        "confidence": "high"
                    }
                ]
            })

        # Arguments prompt starts with "Extract ARGUMENTS"
        elif prompt_lower.startswith("extract all arguments") or prompt_lower.startswith("extract arguments"):
            return json.dumps({
                "arguments": [
                    {
                        "id": "a1",
                        "claim": "Closure order violated natural justice as no personal hearing was granted",
                        "start_char": 2200,
                        "end_char": 2400,
                        "surface_text": "The appellant's primary contention before the High Court was that the closure order was passed in violation...",
                        "actor": "petitioner",
                        "schemes": ["procedural", "rule_application"],
                        "court_response": "accepted",
                        "court_response_start_char": 5200,
                        "court_response_end_char": 5400,
                        "court_reasoning": "Order of such drastic consequence cannot be passed without personal hearing",
                        "confidence": "high"
                    },
                    {
                        "id": "a2",
                        "claim": "Statutory scheme under Environmental Protection Act is self-contained and does not mandate personal hearing",
                        "start_char": 2450,
                        "end_char": 2650,
                        "surface_text": "The respondents contended that the Environmental Protection Act does not mandate a personal hearing...",
                        "actor": "respondent",
                        "schemes": ["rule_application"],
                        "court_response": "rejected",
                        "court_reasoning": "Act does not expressly exclude natural justice",
                        "confidence": "high"
                    }
                ]
            })

        # Precedents prompt starts with "Extract PRECEDENT"
        elif prompt_lower.startswith("extract all precedent") or prompt_lower.startswith("extract precedent"):
            return json.dumps({
                "precedents": [
                    {
                        "id": "p1",
                        "citation": "Maneka Gandhi v. Union of India, AIR 1978 SC 597",
                        "start_char": 4500,
                        "end_char": 4600,
                        "surface_text": "In Maneka Gandhi v. Union of India, AIR 1978 SC 597, this Court revolutionized...",
                        "case_name": "Maneka Gandhi v. Union of India",
                        "case_year": 1978,
                        "cited_proposition": "Procedure established by law must be right, just and fair",
                        "treatment": "followed",
                        "relevance": "central",
                        "confidence": "high"
                    },
                    {
                        "id": "p2",
                        "citation": "Olga Tellis v. Bombay Municipal Corporation, (1985) 3 SCC 545",
                        "start_char": 3600,
                        "end_char": 3750,
                        "surface_text": "In Olga Tellis v. Bombay Municipal Corporation, (1985) 3 SCC 545, this Court held...",
                        "case_name": "Olga Tellis v. Bombay Municipal Corporation",
                        "case_year": 1985,
                        "cited_proposition": "Right to livelihood is integral part of right to life",
                        "treatment": "followed",
                        "relevance": "central",
                        "confidence": "high"
                    }
                ]
            })

        # Outcome prompt starts with "Extract the OUTCOME"
        elif prompt_lower.startswith("extract the outcome"):
            return json.dumps({
                "outcome": {
                    "disposition": "allowed",
                    "start_char": 7200,
                    "end_char": 7400,
                    "surface_text": "Accordingly, the appeal is allowed. The judgment of the High Court dated 25.05.2024 is set aside...",
                    "binary": "accepted",
                    "relief_summary": "Appeal allowed; HC judgment set aside; closure order quashed; matter remanded for fresh hearing",
                    "costs": "none",
                    "directions": [
                        "Pass fresh order after granting personal hearing",
                        "Complete exercise within three months"
                    ]
                }
            })

        # v4 Intra-cluster edges prompt starts with "Extract INTRA-CLUSTER REASONING EDGES"
        elif prompt_lower.startswith("extract intra-cluster reasoning edges"):
            # Determine which concept cluster we are in
            m = re.search(r"concept_id:\s*([a-z0-9_]+)", prompt_lower)
            cid = m.group(1) if m else ""

            # Natural Justice cluster (ontology-backed)
            if "doctrine_natural_justice" in cid:
                return json.dumps({
                    "edges": [
                        {"source": "f3", "target": "c2", "relation": "triggers",
                         "start_char": 1500, "end_char": 1680,
                         "explanation": "No personal hearing triggers natural justice analysis",
                         "confidence": "high", "strength": "strong", "is_critical": True},
                        {"source": "c2", "target": "h2", "relation": "grounds",
                         "start_char": 5200, "end_char": 5400,
                         "explanation": "Natural justice doctrine grounds the violation holding",
                         "confidence": "high", "strength": "strong", "is_critical": True},
                        {"source": "p1", "target": "c2", "relation": "supports",
                         "start_char": 4500, "end_char": 4600,
                         "explanation": "Maneka Gandhi supports incorporation of natural justice into Article 21",
                         "confidence": "high", "strength": "strong"},
                        {"source": "a1", "target": "h2", "relation": "supports",
                         "start_char": 2200, "end_char": 2400,
                         "explanation": "Appellant argument supports the natural justice violation holding",
                         "confidence": "high", "strength": "moderate"},
                        {"source": "a2", "target": "i2", "relation": "addresses",
                         "confidence": "inferred",
                         "explanation": "Respondent argument addresses whether personal hearing is required",
                         "strength": "moderate"}
                    ]
                })

            # Article 21 (pseudo cluster in this test)
            if "const_art21" in cid or "art21" in cid:
                return json.dumps({
                    "edges": [
                        {"source": "c1", "target": "h1", "relation": "grounds",
                         "start_char": 4000, "end_char": 4150,
                         "explanation": "Article 21 jurisprudence grounds the right-to-business holding",
                         "confidence": "high", "strength": "strong", "is_critical": True}
                    ]
                })

            return json.dumps({"edges": []})

        return json.dumps({})


async def test_segmentation():
    """Test document segmentation."""
    print("=" * 60)
    print("TEST: Document Segmentation")
    print("=" * 60)

    doc = segment_document(SAMPLE_JUDGMENT, "test_doc")

    print(f"Document ID: {doc.doc_id}")
    print(f"Total characters: {doc.char_count}")
    print(f"Paragraphs: {doc.para_count}")
    print(f"Sentences: {doc.sent_count}")

    print("\nFirst 5 paragraphs:")
    for para in doc.paragraphs[:5]:
        preview = para.text[:80].replace('\n', ' ')
        print(f"  Para {para.para_index} [{para.start_char}-{para.end_char}]: {preview}...")

    print("\n✓ Segmentation test passed")
    return doc


async def test_extraction():
    """Test the full extraction pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Full Extraction Pipeline")
    print("=" * 60)

    # Use smart mock client
    client = SmartMockClient()
    config = ExtractionConfig(
        model_id="mock-claude-sonnet-4",
        temperature=0.1,
        max_retries=1,
        pipeline_version="v4",
        ontology_path=str((__import__("pathlib").Path(__file__).parent / "ontology_compiled.json").resolve())
    )

    extractor = LegalReasoningExtractor(client, config)

    graph = await extractor.extract(
        text=SAMPLE_JUDGMENT,
        case_id="2024_SC_1234",
        case_name="XYZ Corporation Ltd. v. Union of India",
        case_year=2024,
        court="Supreme Court of India",
        judges=["A. Kumar J"]
    )

    print(f"\n✓ Extraction complete!")
    print(f"  Quality tier: {graph.quality_tier}")
    print(f"  Validation warnings: {len(graph.validation_warnings)}")

    return graph


async def test_graph_operations(graph):
    """Test graph query and analysis operations."""
    print("\n" + "=" * 60)
    print("TEST: Graph Operations")
    print("=" * 60)

    # Test concept retrieval
    print("\nCentral concepts:")
    for cid in graph.get_central_concepts():
        print(f"  - {cid}")

    # Test edges
    print(f"\nEdges from c2 (Natural Justice):")
    for edge in graph.get_edges_from("c2"):
        print(f"  - {edge.id}: {edge.source} --{edge.relation.value}--> {edge.target}")

    # Test Toulmin structure
    if graph.arguments:
        arg = graph.arguments[0]
        print(f"\nToulmin structure for argument {arg.id}:")
        toulmin = graph.get_toulmin_structure(arg.id)
        print(f"  Grounds (facts): {toulmin['grounds']}")
        print(f"  Warrants (concepts): {toulmin['warrants']}")

    # Test holding support
    if graph.holdings:
        holding = graph.holdings[0]
        print(f"\nSupport for holding {holding.id}:")
        support = graph.get_holding_support(holding.id)
        print(f"  Concepts: {support['all_concepts']}")
        print(f"  Justification sets: {len(support['justification_sets'])}")

    print("\n✓ Graph operations test passed")


async def test_counterfactual(graph):
    """Test counterfactual analysis."""
    print("\n" + "=" * 60)
    print("TEST: Counterfactual Analysis")
    print("=" * 60)

    # What if we removed the Natural Justice doctrine?
    if "c2" in [c.id for c in graph.concepts]:
        print("\nCounterfactual: Remove c2 (Natural Justice)")
        result = graph.counterfactual_remove_concept("c2")
        print(f"  Affected holdings: {[h['holding_id'] for h in result['affected_holdings']]}")
        print(f"  Unaffected holdings: {result['unaffected_holdings']}")
        print(f"  Outcome affected: {result['outcome_affected']}")

    # What if we removed Article 21?
    if "c1" in [c.id for c in graph.concepts]:
        print("\nCounterfactual: Remove c1 (Article 21)")
        result = graph.counterfactual_remove_concept("c1")
        print(f"  Affected holdings: {[h['holding_id'] for h in result['affected_holdings']]}")
        print(f"  Outcome affected: {result['outcome_affected']}")

    print("\n✓ Counterfactual test passed")


async def test_json_output(graph):
    """Test JSON serialization."""
    print("\n" + "=" * 60)
    print("TEST: JSON Serialization")
    print("=" * 60)

    json_output = graph.to_json()

    # Verify it's valid JSON
    parsed = json.loads(json_output)

    print(f"JSON size: {len(json_output)} characters")
    print(f"Schema version: {parsed['_meta']['schema_version']}")
    print(f"Quality tier: {parsed['_meta']['quality_tier']}")

    # Print summary
    print("\nGraph summary:")
    print(f"  Facts: {len(parsed['facts'])}")
    print(f"  Concepts: {len(parsed['concepts'])}")
    print(f"  Issues: {len(parsed['issues'])}")
    print(f"  Arguments: {len(parsed['arguments'])}")
    print(f"  Holdings: {len(parsed['holdings'])}")
    print(f"  Precedents: {len(parsed['precedents'])}")
    print(f"  Edges: {len(parsed['edges'])}")
    print(f"  Justification Sets: {len(parsed['justification_sets'])}")

    print("\n✓ JSON serialization test passed")

    return json_output


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LEGAL REASONING GRAPH EXTRACTION - TEST SUITE")
    print("=" * 60)

    # Test 1: Segmentation
    doc = await test_segmentation()

    # Test 2: Full extraction
    graph = await test_extraction()

    # Test 3: Graph operations
    await test_graph_operations(graph)

    # Test 4: Counterfactual
    await test_counterfactual(graph)

    # Test 5: JSON output
    json_output = await test_json_output(graph)

    # Save output
    print("\n" + "=" * 60)
    print("SAVING OUTPUT")
    print("=" * 60)

    # Use relative path that works in any environment
    from pathlib import Path
    output_path = Path(__file__).parent / "test_output.json"
    output_path.write_text(json_output, encoding="utf-8")
    print(f"✓ Graph saved to: {output_path}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())