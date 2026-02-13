"""
Generate External Correctness Annotation PDF for 30 ILTUR Cases.

Reads each case's JSON for extracted data, combines with manual verification
notes, and produces a clean A4 PDF report.
"""

import json
import os
from fpdf import FPDF

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iltur_graphs")

# ── 30 case annotations from external verification research ──────────────────

ANNOTATIONS = [
    # ─── 1947-1955 (4 cases) ─────────────────────────────────────────────────
    {
        "file": "1951_11.json",
        "verified_name": "Legal Representatives of Haji Ali Mohamed Haji Cassum v. State of Bombay",
        "verified_year": 1951,
        "year_note": "Correct. Judgment delivered February 5, 1951. Civil Appeal No. 28 of 1950.",
        "judges_note": "MAHAJAN J. is correct. Justice Mehr Chand Mahajan (later CJI 1954) delivered the judgment. Only the opinion author is listed, not the full bench.",
        "description": (
            "A civil appeal about land revenue assessment in Bombay. The plaintiff owned lands "
            "converted from agricultural to non-agricultural use. He sued claiming the survey officer "
            "and Collector had unlawfully refused to levy non-agricultural assessment. The Court held "
            "that Rule 92 imposes a mandatory duty on the Collector to alter assessment when land use "
            "changes, and the Government's confirmation of the refusal does not constitute a 'direction "
            "otherwise' under Rule 92."
        ),
        "disposition_note": "Appeal allowed. High Court's refusal of relief for Schedule II lands reversed.",
        "precedents_note": "No precedents captured in the data (empty array). This is a gap in extraction, as the judgment likely referenced some authorities.",
        "issues_match": True,
        "issues_note": "The extracted issues about Revenue Jurisdiction Act and survey officer refusing non-agricultural assessment match the case exactly.",
        "accuracy": 8,
        "accuracy_rationale": "Case name, year, judge, issues, and disposition are all correct. Deducted for missing precedents and single-judge listing (likely a multi-judge bench).",
    },
    {
        "file": "1952_105.json",
        "verified_name": "Bajrang Bahadur v. Bakhtraj Kuer",
        "verified_year": 1952,
        "year_note": "Correct. Civil Appeal No. 147 of 1951, from Chief Court of Oudh.",
        "judges_note": "Mukherjea J. is correct. Justice B.K. Mukherjea (later CJI 1954-1956).",
        "description": (
            "A civil succession dispute over a taluqdari estate in Oudh. Raja Bisheshwar Bux Singh "
            "died leaving two sons. The elder son claimed the will gave the younger son (Dhuj Singh) "
            "only a life estate. On Dhuj Singh's death in 1940 without issue, properties should revert. "
            "The Court held Dhuj Singh had only a life interest, not an absolute estate. The will's "
            "dominant intention was to provide for his line without power of alienation."
        ),
        "disposition_note": "Appeal dismissed. The Chief Court of Oudh's affirmation of trial court dismissal was upheld.",
        "precedents_note": "Tagore v. Tagore confirmed as a real Privy Council decision on gifts to unborn persons under Hindu law. Accurately characterized in the data.",
        "issues_match": True,
        "issues_note": "Issues about absolute estate vs. life estate exactly match the real case.",
        "accuracy": 8,
        "accuracy_rationale": "All core facts are correct. Minor: data alternates between 'Avadh' and 'Oudh' spelling.",
    },
    {
        "file": "1953_62.json",
        "verified_name": "Central National Bank Ltd. v. United Commercial Bank Ltd.",
        "verified_year": 1953,
        "year_note": "Correct. Civil Appeal No. 32 of 1953, from Calcutta High Court.",
        "judges_note": "MUKHERJEA J. is correct. Same Justice B.K. Mukherjea.",
        "description": (
            "A commercial law case about the Sale of Goods Act, Section 30(2). A person named Mukherjee "
            "physically obtained possession of shares from a bank. The question was whether he got "
            "possession 'with the consent of' the owner bank, which would protect a bona fide pledgee. "
            "The Court held that consent means agreeing on the same thing in the same sense (Section 13, "
            "Contract Act). On the facts, Mukherjee walked out with the shares without consent."
        ),
        "disposition_note": "Appeal dismissed. Plaintiff bank not entitled to protection under Section 30(2).",
        "precedents_note": "All three cited English precedents confirmed: Cahn v. Pocketts Bristol Channel (1899), Folkes v. King (1923), Pearson v. Rose & Young (1950).",
        "issues_match": True,
        "issues_note": "Issues about Section 30(2) Sale of Goods Act and consent match perfectly.",
        "accuracy": 6,
        "accuracy_rationale": "Legal substance is excellent, but the respondent name is recorded as 'the defendant bank' -- a clear extraction artifact, not a proper party name. Significant data quality problem.",
    },
    {
        "file": "1954_78.json",
        "verified_name": "Karnail Singh and Malkiat Singh v. State of Punjab",
        "verified_year": 1954,
        "year_note": "Year is NULL in data -- critical extraction failure. Should be 1954 based on the case ID (1954_78 from the SCR volume).",
        "judges_note": "Venkatarama Ayyar is correct. There is a minor timing question (appointed January 1955) suggesting the decision may be early 1955, with the SCR volume year being 1954.",
        "description": (
            "A criminal appeal for murder. Karnail Singh and Malkiat Singh were convicted under "
            "Section 302 read with Section 34 IPC for the murder of Gurbaksh Singh. Originally charged "
            "under Section 302/149 (common object), they were convicted under Section 302/34 (common "
            "intention). The Court held that substitution of Section 34 for Section 149 is permissible "
            "where facts overlap and no prejudice results."
        ),
        "disposition_note": "Appeal dismissed. Conviction confirmed.",
        "precedents_note": "All three cited precedents confirmed: Hanumant v. State of MP (1952), Lachhman Singh v. State (1952), Dalip Singh v. State of Punjab. The data correctly shows Dalip Singh was distinguished.",
        "issues_match": True,
        "issues_note": "Issues about Section 302/34 and common intention match exactly.",
        "accuracy": 7,
        "accuracy_rationale": "Excellent legal substance. Null year and empty facts array are significant data quality problems.",
    },
    # ─── 1956-1960 (5 cases) ─────────────────────────────────────────────────
    {
        "file": "1956_21.json",
        "verified_name": "The Mill v. Poona Girni Kamagar Union",
        "verified_year": 1956,
        "year_note": "Correct as 1956 (SCR volume year). Civil Appeal No. 323 of 1955.",
        "judges_note": "Govinda Menon J. listed but uncertain -- he was appointed January 1957. If the case was decided in 1956, there may be a judge attribution issue. Alternatively the SCR volume year may not match the exact decision date.",
        "description": (
            "A labour law case under the Bombay Industrial Relations Act, 1946. A Poona cotton textile "
            "mill was running 580 looms at 2 looms per weaver. Management issued notice for a four-loom "
            "experiment. Workers went on strike. The Labour Court held the strike was not illegal, the "
            "Labour Appellate Tribunal reversed (declaring it illegal), and the High Court reversed the "
            "LAT. The Supreme Court restored the LAT's order."
        ),
        "disposition_note": "Appeal allowed. High Court order set aside, LAT order declaring strike illegal restored.",
        "precedents_note": "No precedents captured (empty array).",
        "issues_match": True,
        "issues_note": "Issues about illegal strike and lockout match the case. The data shows sections 78, 97, 98 of BIRA -- accurately captured.",
        "accuracy": 5,
        "accuracy_rationale": "Vague case name ('The Mill' is the court's abbreviation, not the actual company name). Potential judge attribution timing conflict. No precedents captured.",
    },
    {
        "file": "1957_131.json",
        "verified_name": "Bakshish Singh v. State of Punjab",
        "verified_year": 1957,
        "year_note": "Correct. Criminal Appeal No. 205 of 1956, from Punjab High Court.",
        "judges_note": "Kapur J. is correct. Justice J.L. Kapur (SC 1957-1962).",
        "description": (
            "A criminal appeal about a murder conviction reversed by the High Court from an acquittal. "
            "The central issue was the reliability of a dying declaration. Head Constable Maya Ram "
            "recorded the deceased Bachhinder Singh's dying declaration in Urdu, though the deceased "
            "spoke Punjabi. The Court held that recording in Urdu was standard practice in Punjab courts "
            "and not a ground for rejection."
        ),
        "disposition_note": "Appeal dismissed. High Court's reversal of acquittal upheld; conviction maintained.",
        "precedents_note": "Both cited precedents confirmed: Abdul Mohammad v. AG Palestine (AIR 1945 PC 42) on prosecution witness obligations, and Stephen Seneviratne v. The King (AIR 1936 PC 289).",
        "issues_match": True,
        "issues_note": "Dying declaration in Urdu vs. Punjabi is exactly the central issue. Accurately captured.",
        "accuracy": 9,
        "accuracy_rationale": "One of the most accurately captured cases. Case name, year, judge, issues, precedents, and disposition all correct.",
    },
    {
        "file": "1958_43.json",
        "verified_name": "K.S. Srinivasan v. Director General, All India Radio",
        "verified_year": 1958,
        "year_note": "Correct.",
        "judges_note": "Five-judge Constitution Bench correctly listed: Das CJ, Venkatarama Aiyar, S.K. Das, Sarkar, Bose J. One of the few cases with a complete bench listed.",
        "description": (
            "A service law/constitutional law case. Srinivasan was appointed Liaison Officer at AIR "
            "on a temporary basis. The key question was whether his termination constituted 'punishment' "
            "under Article 311. Applying Parshotam Lal Dhingra v. Union of India, the Court held that "
            "termination of a person who has no right to the post is not punishment. The Court also held "
            "that Article 320(3)(c) consultation with PSC is directory, not mandatory."
        ),
        "disposition_note": "Appeal dismissed and Article 32 petition also dismissed.",
        "precedents_note": "Parshotam Lal Dhingra v. Union of India confirmed as the landmark 1958 case on Article 311(2). Accurately characterized in the data.",
        "issues_match": True,
        "issues_note": "Service law issues about right to post match correctly.",
        "accuracy": 9,
        "accuracy_rationale": "Case name, year, full bench composition, issues, key precedent, and disposition all correct.",
    },
    {
        "file": "1959_43.json",
        "verified_name": "Benares Ice Factory, Ltd. v. New Bheerbhum Coal Co. Ltd.",
        "verified_year": 1959,
        "year_note": "Correct. Civil Appeal No. 342 of 1959, from Calcutta High Court.",
        "judges_note": "Gajendragadkar J. is correct. Justice P.B. Gajendragadkar (SC 1957-1966, CJI 1964-1966).",
        "description": (
            "A civil case about execution proceedings. A consent decree created a first charge on "
            "plant and machinery. When the judgment-debtor defaulted, a receiver was appointed who sold "
            "the immovable property. The question was whether CPC Order 21 Rule 89 (allowing judgment-"
            "debtors to set aside execution sales by depositing purchase price plus 5%) applied to "
            "receiver sales. Held: it does not."
        ),
        "disposition_note": "Appeal dismissed. Order 21 Rule 89 does not apply to receiver sales.",
        "precedents_note": "S.M. Sudevi Devi v. Sovaram Agarwallah confirmed. Correctly shown as distinguished.",
        "issues_match": True,
        "issues_note": "CPC Order 21 Rule 89 and receiver selling immovable property match exactly.",
        "accuracy": 9,
        "accuracy_rationale": "All metadata and legal content correct. Well-captured case.",
    },
    {
        "file": "1960_305.json",
        "verified_name": "State of Bombay v. Ratilal Vadilal Bros.",
        "verified_year": 1960,
        "year_note": "Correct. Civil Appeal No. 429 of 1959.",
        "judges_note": "Hidayatullah J. is correct. Justice M. Hidayatullah (SC 1958-1970, CJI 1968-1970).",
        "description": (
            "A sales tax case under the Bombay Sales Tax Act, 1953. The State treated Ratilal Vadilal "
            "Bros. as 'dealers' selling coal. The Tribunal found they were agents/intermediaries who "
            "facilitated purchases between a colliery and a consumer under the Coal Control Order's "
            "priority certificate system. There was only one sale (colliery to consumer), not two."
        ),
        "disposition_note": "Appeal dismissed. Respondents were not dealers carrying on business of selling coal.",
        "precedents_note": "No precedents captured (empty array). May reflect the judgment not citing cases extensively, or an extraction gap.",
        "issues_match": True,
        "issues_note": "Issues about dealer status under the Act and coal business match exactly.",
        "accuracy": 8,
        "accuracy_rationale": "Case name, year, judge, issues, and disposition all correct. Minor gap: no precedents captured.",
    },
    # ─── 1961-1965 (5 cases) ─────────────────────────────────────────────────
    {
        "file": "1962_60.json",
        "verified_name": "Narain Singh v. State of Punjab",
        "verified_year": 1962,
        "year_note": "Correct. Criminal Appeal No. 218 of 1959.",
        "judges_note": "SHAH J. is correct. Justice J.C. Shah (SC 1959-1971, CJI 1970-1971).",
        "description": (
            "A criminal appeal about self-defence. Narain Singh was convicted under Section 304 "
            "Part II IPC (culpable homicide, not murder) for stabbing Bachan Singh with a kirpan "
            "during a strangulation attempt. The Court held he had NOT exceeded his right of "
            "self-defence -- a strangulation attempt created reasonable apprehension of death "
            "justifying lethal force. The entire Section 342 CrPC statement must be considered "
            "as a whole, not dissected."
        ),
        "disposition_note": "Conviction and sentence set aside. Narain Singh acquitted outright.",
        "precedents_note": "No precedents captured (empty array).",
        "issues_match": True,
        "issues_note": "Self-defence and exceeding right of self-defence exactly match.",
        "accuracy": 8,
        "accuracy_rationale": "All correct. Case name slightly incomplete ('State' without 'of Punjab'). Note: case involves s.304 Part II, not s.302 as some extracted issues suggest.",
    },
    {
        "file": "1962_224.json",
        "verified_name": "Anant Prasad Lakshminivas Generiwal v. State of Andhra Pradesh",
        "verified_year": 1962,
        "year_note": "Correct. Judgment delivered November 2, 1962.",
        "judges_note": "WANCHOO J. is correct. Justice K.N. Wanchoo (later CJI 1968). Only opinion author listed, not full bench.",
        "description": (
            "The appellant was a trustee of a Hindu temple in former Hyderabad territory. He registered "
            "the trust under the MP Public Trusts Act. The State of AP sought to apply the Hyderabad "
            "Endowments Regulations. The Court held that registration under the MP Act does not exclude "
            "the Hyderabad Regulations because situs (temple location) determines jurisdiction. The "
            "Regulations were not repealed by the Part B States Laws Act."
        ),
        "disposition_note": "Partly allowed. Appeal against HC order dismissed, but orders dated June 1960 set aside as ultra vires.",
        "precedents_note": "State of Bihar v. Smt. Charusila Dasi confirmed as a real 1959 SC decision on religious endowment legislation.",
        "issues_match": True,
        "issues_note": "Trust registration, temple endowments, Regulations vs. Art 14 all match.",
        "accuracy": 8,
        "accuracy_rationale": "Case name, year, judge, issues, cited precedent, and disposition all correct. Minor: single-judge listing.",
    },
    {
        "file": "1963_151.json",
        "verified_name": "Jang Singh v. Brij Lal",
        "verified_year": 1963,
        "year_note": "Correct. Judgment delivered February 20, 1963.",
        "judges_note": "Hidayatullah J. is correct. Justice M. Hidayatullah.",
        "description": (
            "A pre-emption case from Punjab. Jang Singh obtained a pre-emption decree and had to "
            "deposit the decretal amount within a specified time. Due to a court officer's miscalculation, "
            "the deposit was short by one rupee. The Court applied the maxim 'Actus curiae neminem "
            "gravabit' (an act of the court shall prejudice no one) and held the court must restore "
            "a person harmed by a court's mistake."
        ),
        "disposition_note": "Appeal allowed. High Court judgment set aside, appellant directed to deposit the additional Re. 1.",
        "precedents_note": "No precedents captured in the data. May reflect the judgment relying on a general legal maxim rather than specific case citations.",
        "issues_match": True,
        "issues_note": "Pre-emption decree and extending time for deposit match exactly.",
        "accuracy": 8,
        "accuracy_rationale": "All core metadata correct. No precedents listed is a minor gap.",
    },
    {
        "file": "1963_175.json",
        "verified_name": "Appellant v. Respondent (anonymized)",
        "verified_year": 1963,
        "year_note": "Data says 1960 but case ID is 1963_175. The year 1960 may refer to the lower court proceedings or filing date. The SC judgment was likely delivered between 1960-1963.",
        "judges_note": "Sinha CJ is correct. Justice Bhuvneshwar Prasad Sinha served as CJI from October 1959 to January 1964.",
        "description": (
            "A criminal case involving IPC sections 493 (cohabitation by deceit about marriage) and 495 "
            "(concealment of former marriage). The substantive legal question before the Supreme Court "
            "was whether Section 5 of the Limitation Act applies to condone delay in filing for special "
            "leave under Section 417(3) CrPC. The Court held that Section 417(4) of the Code is a "
            "'special law' with a different limitation period, and Section 5 does NOT apply."
        ),
        "disposition_note": "Appeal dismissed. High Court's view that s.5 Limitation Act does not apply upheld.",
        "precedents_note": "S.M. Thakur v. State of Bihar and Canara Bank Ltd. v. The Warden Insurance Co. both confirmed as real cases.",
        "issues_match": True,
        "issues_note": "Issues about s.5 Limitation Act and s.417(4) CrPC as special law match exactly.",
        "accuracy": 5,
        "accuracy_rationale": "Legal content is solid, but 'Appellant v. Respondent' is a placeholder -- the case cannot be independently cited or verified by name. Year discrepancy between data (1960) and case ID (1963) is also problematic.",
    },
    {
        "file": "1963_254.json",
        "verified_name": "Dr. Yash Pal Sahi v. Delhi Administration",
        "verified_year": 1963,
        "year_note": "Correct. Judgment delivered November 29, 1963.",
        "judges_note": "GAJENDRAGADKAR J. is correct. At the time a puisne judge (became CJI February 1964).",
        "description": (
            "Dr. Yash Pal Sahi and his wife ran a homoeopathic hospital and journal at Jangpura, "
            "New Delhi. They were prosecuted under the Drugs and Magic Remedies (Objectionable "
            "Advertisements) Act, 1954, for sending a packet containing prohibited advertisements for "
            "homoeopathic medicines. The Court held that 'sending' an advertisement within India "
            "constitutes 'taking part in publication' under s.2(d), and a single contravention is "
            "sufficient to attract penalty under s.7."
        ),
        "disposition_note": "Appeal dismissed. Conviction under s.3 read with s.7 confirmed. Fine of Rs. 500.",
        "precedents_note": "No precedents captured. Case was decided on statutory interpretation of the 1954 Act.",
        "issues_match": True,
        "issues_note": "Issues about s.3/s.7, s.14, and sending advertisements all match.",
        "accuracy": 8,
        "accuracy_rationale": "Case name, year, judge, legal issues, and disposition all correct. Only minor gaps (single-judge listing, no precedents).",
    },
    # ─── 1966-1970 (5 cases) ─────────────────────────────────────────────────
    {
        "file": "1966_100.json",
        "verified_name": "U.P. Co-operative Federation Ltd. v. Sunder Brothers",
        "verified_year": 1966,
        "year_note": "Data says 1964, but case ID is 1966_100. The 1964 value appears to be the civil appeal filing year, not the judgment year. Systematic extraction error.",
        "judges_note": "Ramaswami J. is correct.",
        "description": (
            "A case about the interplay between arbitration clauses and civil suits. The UP Co-operative "
            "Federation's agreement with Sunder Brothers contained an arbitration clause (clause 28). "
            "The Federation sought to stay suit proceedings under s.34 of the Indian Arbitration Act. "
            "The Court held the clause created an arbitration agreement under s.47 (not statutory "
            "arbitration under s.46), and the High Court's discretion in refusing stay was proper given "
            "12 years of delay."
        ),
        "disposition_note": "Appeal dismissed. High Court's refusal to stay suit proceedings upheld.",
        "precedents_note": "Bristol Corporation v. John & Co. and Charles Osenton & Co. v. Johnston both confirmed as real English cases.",
        "issues_match": True,
        "issues_note": "Statutory arbitration under clause 28, s.34 Indian Arbitration Act all match.",
        "accuracy": 7,
        "accuracy_rationale": "Legal substance is correct. Year discrepancy (1964 vs. ~1966) is a systematic pipeline issue. Precedents confirmed.",
    },
    {
        "file": "1967_177.json",
        "verified_name": "National Iron and Steel Co. Ltd. v. NISCO Karmachari Sangha",
        "verified_year": 1967,
        "year_note": "Data says 1965, but case ID is 1967_177. Same systematic extraction error -- 1965 is the appeal filing year.",
        "judges_note": "Mitter J. is correct. Justice G.K. Mitter (SC 1964-1973).",
        "description": (
            "An industrial disputes case involving four interrelated companies operating from the same "
            "premises at Belur, Howrah. The government made one order of reference covering all four. "
            "The Court upheld the single reference as valid due to 'sufficient functional integrality' "
            "between the companies. The Tribunal's gratuity scheme, retrenchment findings, and abolition "
            "of contract labour were all upheld."
        ),
        "disposition_note": "Appeal dismissed. Tribunal's award upheld on all four contentions.",
        "precedents_note": "Wenger & Co. v. Their Workmen and Workmen of Dimakuchi Tea Estate confirmed as real cases.",
        "issues_match": True,
        "issues_note": "Industrial Disputes Act s.10, gratuity scheme, retrenchment all match.",
        "accuracy": 7,
        "accuracy_rationale": "Legal content is strong. Year discrepancy (1965 vs. ~1967) and the functional integrality test well-captured.",
    },
    {
        "file": "1967_265.json",
        "verified_name": "M/s. Balwant Singh Santok Singh v. Income-tax Officer",
        "verified_year": 1967,
        "year_note": "Year is NULL in data -- significant extraction failure. Should be approximately 1967 based on case ID.",
        "judges_note": "Shelat J. is correct. Justice J.C. Shelat (SC 1963-1973, later CJI).",
        "description": (
            "A partnership firm registered under s.26A of the Income-tax Act, 1922. The ITO filed a "
            "criminal complaint alleging offences under IPC ss.193 and 196. The firm challenged whether "
            "the ITO acting under s.26A was a 'court' requiring s.476/479A CrPC procedure. The Court "
            "held that ITO proceedings are 'proceedings in court' for s.195(1)(b) CrPC, but the ITO is "
            "NOT a 'revenue court', so ss.476/479A do not apply."
        ),
        "disposition_note": "Appeal dismissed. Complaint not quashed; jurisdictional issue to be raised before Magistrate.",
        "precedents_note": "In re Poonamchand Maneklal, State v. Nemchand Pashvir Patel, and Jagannath Prasad v. State of UP all confirmed.",
        "issues_match": True,
        "issues_note": "Issues about s.26A Income-tax Act and ss.476/479A CrPC match exactly.",
        "accuracy": 6,
        "accuracy_rationale": "Legal substance is nuanced and well-captured. But null year AND null court are significant metadata gaps.",
    },
    {
        "file": "1967_295.json",
        "verified_name": "Bhanwar Singh v. State of Rajasthan",
        "verified_year": 1967,
        "year_note": "Correct.",
        "judges_note": "Vaidialingam J. is correct. Justice S. Vaidialingam (SC 1966-1973).",
        "description": (
            "A criminal conspiracy case involving cheating of banks and post offices through forgery. "
            "Two appellants were convicted under IPC ss.120B, 420, 467/471, 380 and others. The key "
            "legal question was whether sanction under s.196A CrPC was needed. The Court held no -- the "
            "object of the conspiracy was cheating (a cognizable offence), while forgery was merely the "
            "means. The distinction between 'object' and 'means' of conspiracy is significant."
        ),
        "disposition_note": "Appeal dismissed. Convictions upheld.",
        "precedents_note": "State of AP v. Kandimalla Subbaiah confirmed -- a key precedent on s.196A CrPC.",
        "issues_match": True,
        "issues_note": "Sanction under s.196A, evidence assessment, and conspiracy charges all match.",
        "accuracy": 9,
        "accuracy_rationale": "Best case in this batch. All metadata correct. Object vs. means distinction in conspiracy law is well-captured.",
    },
    {
        "file": "1969_132.json",
        "verified_name": "M/s. Munshi Lal Beni Ram Glass Works v. Lal Khan",
        "verified_year": 1969,
        "year_note": "Data says 1968, but case ID is 1969_132. The 1968 value is the appeal filing year. Same systematic error.",
        "judges_note": "Dua J. is correct. Justice I.D. Dua (SC 1968-1972).",
        "description": (
            "A case about the U.P. Industrial Disputes Act amendments. The core questions involved s.6A "
            "as amended by U.P. Act 1 of 1957. The Court held that s.16 of the Act refers to s.6A as "
            "amended (not the prior version), proceedings pending before the adjudicator attract s.6A, "
            "and resort to s.17 is NOT essential for enforcement of the award."
        ),
        "disposition_note": "Appeals dismissed with costs.",
        "precedents_note": "Central Distillery and Chemical Works Ltd., Meerut v. State of UP confirmed. State of Punjab v. Mohar Singh also confirmed.",
        "issues_match": True,
        "issues_note": "S.6A of UP Industrial Disputes Amendment and enforcement of award match.",
        "accuracy": 7,
        "accuracy_rationale": "Legal content correct. Year discrepancy (1968 vs. ~1969) is the established systematic issue.",
    },
    # ─── 1971-1974 (5 cases) ─────────────────────────────────────────────────
    {
        "file": "1971_100.json",
        "verified_name": "Messrs Alloy Steel Project v. Workmen",
        "verified_year": 1971,
        "year_note": "Data says 1969, but case ID is 1971_100. Again the appeal filing year (1969) was captured instead of the judgment year (~1970-71).",
        "judges_note": "Bhargava J. is correct. Justice Vishishtha Bhargava (SC 1966-1973).",
        "description": (
            "The Alloy Steel Project was an undertaking of Hindustan Steel Ltd. (a Government company). "
            "The dispute was about bonus for 1965-66 under the Payment of Bonus Act. The Court held that "
            "an 'establishment' is distinct from a 'company' under the Act. Under the proviso to s.3, "
            "the Alloy Steel Project IS a separate establishment because separate accounts are maintained. "
            "Since the separate establishment showed no surplus, no bonus was payable."
        ),
        "disposition_note": "Appeal allowed. Tribunal order set aside. No bonus payable for 1965-66.",
        "precedents_note": "No precedents captured. Case decided primarily on statutory interpretation of the Payment of Bonus Act.",
        "issues_match": True,
        "issues_note": "Whether Alloy Steel Project was a separate establishment matches exactly.",
        "accuracy": 7,
        "accuracy_rationale": "Legal substance is correct. Year discrepancy is the systematic issue. No precedents captured.",
    },
    {
        "file": "1971_578.json",
        "verified_name": "Union of India v. Ram Kishan",
        "verified_year": 1971,
        "year_note": "Correct. Judgment delivered in 1971.",
        "judges_note": "Sikri CJ is correct. S.M. Sikri served as CJI from January 1971 to April 1973.",
        "description": (
            "Ram Kishan, a Foot Constable in Delhi Police, was dismissed by the SP (Traffic). He "
            "challenged the dismissal on grounds that (a) the SP Traffic was not competent, and (b) "
            "Punjab Police Rules 16.38 were not complied with. The Court held that Rule 16.38(1) was "
            "not complied with at all, the departmental inquiry was vitiated, and the dismissal was "
            "illegal."
        ),
        "disposition_note": "Appeal dismissed with costs. Dismissal order declared illegal.",
        "precedents_note": "Union of India v. Jagjit Singh and Delhi Admn. v. Chanan Shah confirmed as real and relevant cases. One self-citation to an earlier stage of the same case noted.",
        "issues_match": True,
        "issues_note": "SP Traffic competence, Punjab Police Rules 16.38/16.24, mandatory vs. directory all match.",
        "accuracy": 8,
        "accuracy_rationale": "All core metadata correct. Only deduction: single-judge listing when bench likely had multiple judges.",
    },
    {
        "file": "1972_479.json",
        "verified_name": "Oriental Mercantile Agency v. Presiding Officer, Labour Court, Madras",
        "verified_year": 1972,
        "year_note": "Data says 1971 but this is demonstrably wrong. Justice Chandrachud was not appointed to the Supreme Court until July 24, 1972. The case must have been decided in late 1972 or later.",
        "judges_note": "CHANDRACHUD J. is correct for the case but inconsistent with the year 1971 in the data.",
        "description": (
            "A labour law case with a tortuous procedural history. In 1961, Oriental Mercantile Agency "
            "retrenched six workmen. The Labour Court gave an award in 1963 finding retrenchment "
            "justified. This was challenged through multiple stages: a single judge set aside the award "
            "(1967), a Division Bench issued an order (1967), then a clarificatory order was issued "
            "(1967), and a second Labour Court award came (1968). The Supreme Court found violations "
            "of natural justice at multiple stages."
        ),
        "disposition_note": "Remanded. Four separate orders set aside; Writ Appeal revived for disposal on merits.",
        "precedents_note": "No precedents captured in the data.",
        "issues_match": True,
        "issues_note": "Labour court jurisdiction and Division Bench judgment correctness match the case.",
        "accuracy": 6,
        "accuracy_rationale": "Legal substance and disposition are accurate. The year is verifiably wrong (1971 vs. 1972+) since Chandrachud was not on the SC until July 1972.",
    },
    {
        "file": "1974_85.json",
        "verified_name": "Mohd. Yunus Saleem v. Shiv Kumar Shastri",
        "verified_year": 1974,
        "year_note": "Data says 1972, which is the appeal filing year. The SC judgment was delivered approximately 1973-1974 (reported in 1974 SCR volume).",
        "judges_note": "GOSWAMI J. is correct. Justice P.K. Goswami (SC 1971-1978).",
        "description": (
            "An election law case from the Aligarh parliamentary constituency. Mohd. Yunus Saleem, the "
            "defeated Congress(R) candidate, challenged the election of BKD candidate Shiv Kumar Shastri. "
            "The petition raised issues about Election Commission's power to adjourn polls, withdrawal "
            "of candidature, and corrupt practices under sections 123(1)-(4) of the Representation of "
            "the People Act, 1951. The Allahabad High Court dismissed the petition."
        ),
        "disposition_note": "Appeal dismissed with costs. No corrupt practice proved.",
        "precedents_note": "No precedents captured in the data.",
        "issues_match": True,
        "issues_note": "Election Commission adjourning poll, withdrawal from election, and corrupt practices all match.",
        "accuracy": 7,
        "accuracy_rationale": "Core facts are accurate. Year discrepancy is the filing-year vs. decision-year pattern. Missing precedents is a minor gap.",
    },
    {
        "file": "1974_129.json",
        "verified_name": "Anil Kumar Bose and Raghunath Prasad v. State of Bihar",
        "verified_year": 1974,
        "year_note": "Data says 1970, which is the appeal filing year. SC judgment was closer to 1974.",
        "judges_note": "Goswami J is correct. Same Justice P.K. Goswami.",
        "description": (
            "An accountant and cashier were convicted under Section 420/34 IPC (cheating with common "
            "intention) by the Patna High Court. The key question was whether they acted with mens rea "
            "(criminal intent) or whether the irregularities were merely an administrative lapse. The "
            "Supreme Court found the evidence insufficient to establish dishonest intention."
        ),
        "disposition_note": "Appeal allowed. Appellants acquitted and discharged from bail bonds.",
        "precedents_note": "No precedents captured.",
        "issues_match": True,
        "issues_note": "Section 420/34 IPC, mens rea vs. administrative lapse match exactly.",
        "accuracy": 7,
        "accuracy_rationale": "Legal substance correct. 'v. State' could more precisely be 'v. State of Bihar'. Year discrepancy follows the pattern.",
    },
    # ─── 2000-2019 (6 cases) ─────────────────────────────────────────────────
    {
        "file": "2002_580.json",
        "verified_name": "Pannalal Jaysoara & Mohd. Gulzar v. State",
        "verified_year": 2002,
        "year_note": "Plausible for the SC judgment. TADA charges (repealed 1995) could still be adjudicated in 2002 for cases already filed.",
        "judges_note": "ARIJIT PASAYAT is correct. Justice Pasayat (SC August 2001 - May 2009).",
        "description": (
            "A major criminal conspiracy case involving bomb manufacturing in the Bow Bazar area of "
            "Calcutta with intent to strike terror and affect communal harmony. Charges included IPC "
            "ss.120B, 302/34, 436/34, 307, 326, Explosive Substances Act, and TADA. The Court held "
            "that post-arrest confessional statements do NOT fall within Section 10 of the Evidence Act "
            "as the agency of conspiracy ceases after arrest."
        ),
        "disposition_note": "All appeals dismissed. Convictions under 120B IPC, TADA, and Explosive Substances Act upheld. State's appeal against acquittal on ss.302/34 and 436/34 also dismissed.",
        "precedents_note": "No precedents captured in the data. The facts array is also empty -- a significant extraction gap.",
        "issues_match": True,
        "issues_note": "Criminal conspiracy 120B IPC, bomb manufacturing, confessional statements under s.164 CrPC all match.",
        "accuracy": 5,
        "accuracy_rationale": "Judge, year, and legal issues correct. But 'Jaysoara' is likely a garbled surname (possibly 'Jaisiwal'). Empty facts array is a major data gap. The case involves much more serious charges (TADA, communal terrorism) than the summary suggested.",
    },
    {
        "file": "2003_78.json",
        "verified_name": "Appellant v. State of Rajasthan (name anonymized/not extracted)",
        "verified_year": 2003,
        "year_note": "Data says 2002, case ID is 2003_78. Minor discrepancy -- likely filing vs. reporting year.",
        "judges_note": "Arijit Pasayat is correct.",
        "description": (
            "A service law case. The appellant, a lady doctor appointed on a purely temporary basis in "
            "1974 under the Local Self-Government Department, Government of Rajasthan, sought "
            "regularization. Her temporary term kept being extended. The Court upheld the Division Bench "
            "view that there was no right to regularization and the termination was valid."
        ),
        "disposition_note": "Appeal dismissed. Termination upheld; no regularization.",
        "precedents_note": "Director, Institute of Management Development, U.P. vs. Pushpa Srivastava (1992) 4 SCC 33 confirmed as a real case on regularization of temporary appointments.",
        "issues_match": True,
        "issues_note": "Termination from service and regularization of temporary appointment match.",
        "accuracy": 6,
        "accuracy_rationale": "Legal substance, judge, and precedent correct. But 'Appellant v. State of Rajasthan' fails to identify the actual party -- a data extraction failure.",
    },
    {
        "file": "2004_764.json",
        "verified_name": "Deepa Jain and Hari Om Maheshwari v. the respondent (name not extracted)",
        "verified_year": 2004,
        "year_note": "Correct.",
        "judges_note": "Santosh Hegde is correct. Justice N. Santosh Hegde (SC December 1999 - June 2005).",
        "description": (
            "An arbitration case under the Arbitration Act, 1940. During proceedings, the respondent "
            "failed to appear on May 10, 1999 (arbitrators waited until 4:40 PM). Arbitrators declined "
            "to grant further adjournment and delivered awards in November 1999. The respondent challenged "
            "under Section 30, and the High Court set aside the awards. The Supreme Court reversed, "
            "holding that refusal of adjournment by arbitrators does not constitute misconduct under "
            "Section 30(a)."
        ),
        "disposition_note": "Appeal allowed. High Court erred in setting aside the awards to give opportunity to a defaulting party.",
        "precedents_note": "Arosan Enterprises Ltd. v. Union of India (1999) 9 SCC 449 and State of U.P. vs. Allied Constructions (2003) 7 SCC 396 both confirmed.",
        "issues_match": True,
        "issues_note": "Arbitration, High Court interference with arbitrator discretion, refusal of further evidence all match.",
        "accuracy": 7,
        "accuracy_rationale": "Legal substance, year, judge, and precedents all correct. Respondent name missing ('v. the respondent').",
    },
    {
        "file": "2007_946.json",
        "verified_name": "Aravali Golf Club v. Chander Hass & Anr.",
        "verified_year": 2007,
        "year_note": "Correct. SLP (C) No. 3358 of 2007.",
        "judges_note": "No judges listed -- empty array in data. This is a significant extraction failure. The case was likely decided by a bench including Justice Markandey Katju, known for its strong statements about judicial restraint.",
        "description": (
            "The respondents were hired as Mali (gardeners) at the Aravali Golf Club (run by Haryana "
            "Tourism Corporation) on daily wages in 1988-89. They were later assigned tractor driver "
            "duties without that post existing. In 2001 they sued for regularization as tractor drivers. "
            "The Supreme Court held courts cannot direct employers to create posts that do not exist in "
            "the sanctioned establishment."
        ),
        "disposition_note": "Appeal allowed. Trial Court judgment (dismissing the suit) restored.",
        "precedents_note": "Indian Drugs Pharmaceuticals Ltd. v. Workman (2007) 1 SCC 408 and S.C. Chandra v. State of Jharkhand confirmed.",
        "issues_match": True,
        "issues_note": "Regularization of tractor driver posts and court jurisdiction to create posts match.",
        "accuracy": 6,
        "accuracy_rationale": "Legal substance and precedents correct. Missing judges is significant. 'v. the plaintiffs/respondents' is not a proper citation.",
    },
    {
        "file": "2008_1460.json",
        "verified_name": "Nikhil Merchant v. Central Bureau of Investigation",
        "verified_year": 2008,
        "year_note": "Correct. Judgment dated October 16, 2008. Reported at (2008) 9 SCC 677.",
        "judges_note": "Altamas Kabir is correct. Justice Kabir (SC September 2005, later CJI September 2012 - July 2013).",
        "description": (
            "A landmark case on quashing criminal proceedings on compromise. FIR under Sections "
            "420/468/471/34/120-B IPC for a vehicle financing dispute (Maruti Van financed for Rs. 30,000). "
            "After compromise (Rs. 45,000 each), the appellant sought quashing. The Court held that the "
            "power under Section 482 CrPC is NOT limited by Section 320 CrPC, even for non-compoundable "
            "offences. In private disputes where parties have settled, a pragmatic approach should be taken."
        ),
        "disposition_note": "Appeal allowed. Criminal proceedings quashed.",
        "precedents_note": "B.S. Joshi v. State of Haryana (2003) 4 SCC 675 confirmed as the seminal precedent. Pepsi Foods Ltd. v. Special Judicial Magistrate (1998) 5 SCC 749 also confirmed.",
        "issues_match": True,
        "issues_note": "FIR under ss.420/468/471/34/120-B, power under s.482 CrPC, quashing on compromise all match.",
        "accuracy": 9,
        "accuracy_rationale": "Excellent data quality. Case name, year, judge, legal issues, both precedents, and disposition all verified.",
    },
    {
        "file": "2010_721.json",
        "verified_name": "D. Velusamy v. D. Patchaiammal",
        "verified_year": 2010,
        "year_note": "Correct. Reported at (2010) 10 SCC 469.",
        "judges_note": "Markandey Katju is correct. Justice Katju (SC November 2006 - September 2011).",
        "description": (
            "A landmark case on live-in relationships and the Protection of Women from Domestic Violence "
            "Act, 2005 (PWDVA). The appellant claimed marriage to Lakshmi (1980), while respondent "
            "Patchaiammal claimed to be his wife. The Court held that the lower court findings about "
            "Lakshmi's status were null and void for violating natural justice (she was never heard). "
            "The Court also laid down five conditions for 'relationship in the nature of marriage' under "
            "the PWDVA: (1) hold out as spouses, (2) legal age, (3) qualified to marry, (4) voluntary "
            "cohabitation, (5) significant period of time."
        ),
        "disposition_note": "Remanded. Lower court findings set aside. Case remanded to Family Court to hear Lakshmi.",
        "precedents_note": "Six precedents confirmed including S. Khushboo v. Kanniammal (2010), Savitaben v. State of Gujarat (2005), and notably the American cases Marvin v. Marvin (1976), Taylor v. Fields (1986), and Devaney v. L'Esperance (2008).",
        "issues_match": True,
        "issues_note": "Marriage law, live-in relationships, PWDVA, and second wife validity all match.",
        "accuracy": 9,
        "accuracy_rationale": "Excellent data quality. All metadata, legal substance, six precedents (including foreign cases), and disposition verified.",
    },
]


def load_case(filename):
    """Load and return JSON data for a case."""
    path = os.path.join(BASE, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AnnotationPDF(FPDF):
    """Custom PDF with headers and footers."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "External Correctness Annotations -- ILTUR Graph Data", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, text)
        self.ln(7)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def label_value(self, label, value):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(40, 40, 40)
        self.cell(38, 5, label + ":")
        self.set_font("Helvetica", "", 9.5)
        # Handle long values with multi_cell
        x = self.get_x()
        y = self.get_y()
        self.multi_cell(0, 5, value)
        self.ln(1)

    def rating_badge(self, score):
        """Draw a visual accuracy rating."""
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        if score >= 8:
            self.set_text_color(34, 139, 34)  # green
        elif score >= 6:
            self.set_text_color(200, 150, 0)  # amber
        else:
            self.set_text_color(200, 50, 50)  # red
        self.cell(0, 8, f"Accuracy Rating: {score}/10")
        self.set_text_color(40, 40, 40)
        self.ln(10)

    def horizontal_rule(self):
        y = self.get_y()
        self.set_draw_color(180, 180, 180)
        self.line(10, y, 200, y)
        self.ln(5)


def extract_summary(data):
    """Extract a brief summary from case JSON data."""
    parts = []
    name = data.get("case_name") or "N/A"
    year = data.get("case_year") or "N/A"
    court = data.get("court") or "N/A"
    judges = data.get("judges") or []
    parts.append(f"Case Name: {name}")
    parts.append(f"Year: {year}")
    parts.append(f"Court: {court}")
    parts.append(f"Judges: {', '.join(judges[:5]) if judges else 'N/A'}")

    # Issues
    issues = data.get("legal_issues") or data.get("issues") or []
    if issues:
        issue_texts = [i.get("text", "")[:120] for i in issues[:3]]
        parts.append(f"Key Issues: {'; '.join(issue_texts)}")

    # Precedents
    cited = data.get("cited_cases") or data.get("precedents") or []
    if cited:
        cited_names = []
        for c in cited[:4]:
            cn = c.get("case_name") or c.get("name") or ""
            if cn:
                cited_names.append(cn[:80])
        if cited_names:
            parts.append(f"Cited Precedents: {'; '.join(cited_names)}")

    # Outcome
    outcome = data.get("outcome") or {}
    if isinstance(outcome, dict):
        disp = outcome.get("disposition") or ""
        if disp:
            parts.append(f"Disposition: {disp}")

    return "\n".join(parts)


def build_pdf():
    pdf = AnnotationPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ────────────────────────────────────────────────────────────
    W = 190  # effective page width (A4 210mm - 10mm left - 10mm right)
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(10)
    pdf.multi_cell(W, 12, "External Correctness Annotations", align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(10)
    pdf.multi_cell(W, 10, "ILTUR Graph Data", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(10)
    pdf.multi_cell(
        W, 7,
        "A verification of 30 Indian Supreme Court cases extracted by Grok\n"
        "from the ILTUR dataset (1947-2019), checked against external sources.",
        align="C",
    )
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    pdf.multi_cell(
        W, 6,
        "30 cases sampled across six time periods:\n"
        "1947-1955 (4) | 1956-1960 (5) | 1961-1965 (5)\n"
        "1966-1970 (5) | 1971-1974 (5) | 2000-2019 (6)",
        align="C",
    )
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(10)
    pdf.multi_cell(
        W, 6,
        "February 2026",
        align="C",
    )

    # ── Methodology Page ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Methodology")
    pdf.ln(14)
    pdf.body_text(
        "This report examines 30 cases drawn from 2,518 structured JSON files in the "
        "iltur_graphs/ directory. Cases were selected using stratified random sampling "
        "across six year ranges, preferring cases with proper adversarial party names "
        "(containing 'v.') for easier verification."
    )
    pdf.body_text(
        "For each case, we checked the following against web sources and legal databases:"
    )
    pdf.body_text(
        "  1. Case name -- Is it the correct, citable name?\n"
        "  2. Year -- Does the recorded year match the judgment date?\n"
        "  3. Judges -- Are the listed judges correct for this case?\n"
        "  4. Legal issues -- Do the extracted issues reflect what the case is about?\n"
        "  5. Precedents -- Are the cited cases real and accurately characterized?\n"
        "  6. Disposition -- Does the outcome (allowed/dismissed) match?"
    )
    pdf.body_text(
        "Each case receives an accuracy rating from 1 to 10, where 10 means all "
        "extracted data is correct with no gaps, and lower scores reflect errors in "
        "metadata, missing fields, or incorrect attributions."
    )
    pdf.body_text(
        "Verification was performed using web searches against Indian Kanoon, SCC Online, "
        "and general legal databases, supplemented by knowledge of Indian Supreme Court "
        "jurisprudence and judicial appointment records."
    )

    # ── Individual Case Pages ─────────────────────────────────────────────────
    for idx, ann in enumerate(ANNOTATIONS, 1):
        data = load_case(ann["file"])
        pdf.add_page()

        # Case header
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        case_id = ann["file"].replace(".json", "")
        pdf.multi_cell(0, 7, f"Case {idx}/30: {ann['verified_name']}")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"File: {ann['file']}  |  Case ID: {case_id}")
        pdf.ln(8)

        pdf.horizontal_rule()

        # Extracted Data Summary
        pdf.section_title("Extracted Data (from JSON)")
        summary = extract_summary(data)
        pdf.body_text(summary)

        pdf.horizontal_rule()

        # Verification Notes
        pdf.section_title("Verification Notes")

        pdf.label_value("Year", ann["year_note"])
        pdf.label_value("Judges", ann["judges_note"])
        pdf.label_value("Description", ann["description"])
        pdf.label_value("Disposition", ann["disposition_note"])
        pdf.label_value("Precedents", ann["precedents_note"])
        pdf.label_value("Issues Match", ann["issues_note"])

        pdf.horizontal_rule()

        # Accuracy
        pdf.section_title("Assessment")
        pdf.body_text(ann["accuracy_rationale"])
        pdf.rating_badge(ann["accuracy"])

    # ── Summary / Statistics Page ─────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Summary Statistics")
    pdf.ln(14)

    scores = [a["accuracy"] for a in ANNOTATIONS]
    avg = sum(scores) / len(scores)
    high = [s for s in scores if s >= 8]
    mid = [s for s in scores if 6 <= s < 8]
    low = [s for s in scores if s < 6]

    pdf.section_title("Overall Results")
    pdf.body_text(f"Total cases verified: {len(ANNOTATIONS)}")
    pdf.body_text(f"Average accuracy score: {avg:.1f}/10")
    pdf.body_text(f"High accuracy (8-10): {len(high)} cases ({len(high)/len(scores)*100:.0f}%)")
    pdf.body_text(f"Medium accuracy (6-7): {len(mid)} cases ({len(mid)/len(scores)*100:.0f}%)")
    pdf.body_text(f"Low accuracy (<6): {len(low)} cases ({len(low)/len(scores)*100:.0f}%)")

    pdf.ln(3)
    pdf.horizontal_rule()

    pdf.section_title("Score Distribution")
    for score in range(10, 0, -1):
        count = scores.count(score)
        if count > 0:
            bar = "|" * (count * 4)
            pdf.body_text(f"  {score}/10:  {bar}  ({count} case{'s' if count != 1 else ''})")

    pdf.ln(3)
    pdf.horizontal_rule()

    pdf.section_title("By Time Period")
    periods = [
        ("1947-1955", ANNOTATIONS[0:4]),
        ("1956-1960", ANNOTATIONS[4:9]),
        ("1961-1965", ANNOTATIONS[9:14]),
        ("1966-1970", ANNOTATIONS[14:19]),
        ("1971-1974", ANNOTATIONS[19:24]),
        ("2000-2019", ANNOTATIONS[24:30]),
    ]
    for label, group in periods:
        period_scores = [a["accuracy"] for a in group]
        period_avg = sum(period_scores) / len(period_scores)
        pdf.body_text(f"  {label}: avg {period_avg:.1f}/10  (scores: {', '.join(str(s) for s in period_scores)})")

    pdf.ln(3)
    pdf.horizontal_rule()

    pdf.section_title("Systemic Issues Identified")
    pdf.body_text(
        "1. Year extraction error (systematic): In many cases (especially 1960s-1970s), "
        "the case_year field captures the appeal filing year rather than the Supreme Court "
        "judgment delivery year. The case ID (derived from the SCR volume number) is more "
        "reliable. This appears to be a pipeline bug in the Grok-based extraction."
    )
    pdf.body_text(
        "2. Incomplete case names: Several cases have placeholder names like 'Appellant v. "
        "Respondent', 'The Mill', or 'v. the respondent' instead of actual party names. This "
        "is a significant data quality issue for citation and verification purposes."
    )
    pdf.body_text(
        "3. Single-judge attribution: Most cases list only the opinion author, not the full "
        "bench composition. Supreme Court benches in this era typically had 2-5 judges."
    )
    pdf.body_text(
        "4. Missing precedents: Several cases have empty precedent arrays despite the judgment "
        "likely citing prior authorities. This may reflect extraction gaps for older judgments."
    )
    pdf.body_text(
        "5. OCR artifacts: Some surface_text fields contain systematic OCR errors from Indian "
        "Kanoon (e.g., 'companymitted' for 'committed'). These propagate into anchor texts."
    )
    pdf.body_text(
        "6. Empty facts arrays: A few cases have no facts extracted at all, which is a "
        "significant data completeness gap."
    )

    pdf.ln(3)
    pdf.horizontal_rule()

    pdf.section_title("Strengths of the Dataset")
    pdf.body_text(
        "1. Legal content quality is generally strong -- holdings, issues, and arguments "
        "are accurately extracted and capture key legal principles."
    )
    pdf.body_text(
        "2. Cited precedents that could be verified (including English and American cases) "
        "all check out as real cases with correct citations."
    )
    pdf.body_text(
        "3. Dispositions are correctly classified in all cases examined."
    )
    pdf.body_text(
        "4. Post-2000 cases show significantly better data quality than older cases, "
        "likely due to better digitization and clearer judgment formatting."
    )
    pdf.body_text(
        "5. The Toulmin argument structure and legal reasoning chains appear well-captured."
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iltur_annotations.pdf")
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
    print(f"Total pages: {pdf.page_no()}")
    print(f"Cases annotated: {len(ANNOTATIONS)}")
    print(f"Average accuracy: {avg:.1f}/10")


if __name__ == "__main__":
    build_pdf()
