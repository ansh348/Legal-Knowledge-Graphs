# Computational Legal Reasoning Patterns in Indian Supreme Court Judgments: Implementation Specifications

This reference manual provides directly implementable extraction rules for converting Indian Supreme Court judgments into structured reasoning graphs. All patterns include actual linguistic signals from landmark judgments, edge type mappings, and confidence scores for NLP extraction.

---

## 1. Linguistic Patterns for Fact-Concept-Holding Extraction

### 1.1 Fact-to-Doctrine Triggering Patterns

**Pattern: doctrine_invocation**
```
Signals: ["This brings into play", "attracts the doctrine of", "invokes the principle of", "falls within the ambit of", "brings into operation", "triggers the application of", "squarely falls under", "this makes it necessary to analyse", "takes in", "is a facet of", "is an essential element of", "is intrinsic to", "emanates from", "flows from"]
Creates edge: (Fact) --[TRIGGERS]--> (Doctrine)
Confidence: high
Example: "This makes it necessary to analyse the origins of privacy and to trace its evolution" - K.S. Puttaswamy (2017)
```

**Pattern: constitutional_nexus**
```
Signals: ["is a part of", "is implicit in", "is recognized as dimensions of", "is guaranteed under", "forms part of", "is traceable to", "can be traced to Articles", "is a fundamental right which can be traced to"]
Creates edge: (Right) --[DERIVED_FROM]--> (Constitutional_Provision)
Confidence: high
Example: "right to privacy is a part of fundamental rights which can be traced to Articles 14, 19 and 21" - K.S. Puttaswamy (2017)
```

**Pattern: requirement_failure**
```
Signals: ["does not satisfy", "fails to meet", "is violative of", "falls foul of", "cannot be sustained", "is not saved under", "is hit by", "does not come within", "cannot pass muster", "suffers from the vice of", "does not answer the test of", "is disproportionate to", "fails the test of", "is inconsistent with"]
Creates edge: (Fact_Pattern) --[FAILS]--> (Legal_Requirement)
Confidence: high
Example: "Section 66A cannot possibly be said to create an offence which falls within the expression 'decency' or 'morality'" - Shreya Singhal (2015)
```

**Pattern: conjunctive_satisfaction (AND logic)**
```
Signals: ["taken together", "read with", "in conjunction with", "coupled with", "read together", "considered in totality", "cumulatively", "all the three requirements", "each of the ingredients must be satisfied", "must satisfy all conditions", "sine qua non", "the twin conditions"]
Creates edge: (Multiple_Facts) --[JOINTLY_SATISFY]--> (Test)
Confidence: high
Example: "three-fold criteria of legality, legitimacy of aims, and proportionality" - K.S. Puttaswamy (2017)
```

**Pattern: disjunctive_sufficiency (OR logic)**
```
Signals: ["either...or", "any one of the following", "independently of", "alternatively", "or else", "in any of these cases", "on any of the grounds", "even one of", "any ground", "by itself sufficient"]
Creates edge: (Single_Fact) --[SUFFICES_FOR]--> (Test)
Confidence: medium
```

### 1.2 Holding and Ratio Markers

**Pattern: primary_holding**
```
Signals: ["We hold that", "We are of the considered view that", "We are therefore of the view that", "We accordingly hold", "In our considered view", "For the foregoing reasons, we hold", "We have no hesitation in holding", "We declare", "We direct", "We affirm", "We uphold", "We strike down", "We quash", "It must therefore be held"]
Creates edge: (Analysis) --[CONCLUDES_WITH]--> (Holding)
Confidence: high
Example: "we hold that bail covers both-release on one's own bond, with or without sureties" - Moti Ram v. State of M.P. (1978)
```

**Pattern: ratio_decidendi_marker**
```
Signals: ["the ratio of the judgment", "the principle upon which the case is decided", "the ratio decidendi of this case", "we lay down", "we authoritatively hold", "the law declared by this Court", "this proposition is binding", "forms the basis of the decision"]
Creates edge: (Holding) --[CREATES]--> (Binding_Precedent)
Confidence: high
Example: "The only thing binding in a judge's decision is the principle upon which the case is decided" - Union of India v. Dhanwanti Devi
```

**Pattern: obiter_dicta_marker**
```
Signals: ["We may observe", "We may note", "We hasten to add", "It may be mentioned", "In passing, we note", "merely an observation", "not necessary for the decision", "by way of observation", "we need not express any opinion", "we leave open", "we do not decide", "does not arise for decision"]
Creates edge: (Observation) --[MARKED_AS]--> (Obiter_Dictum)
Confidence: high
Example: "obiter dictum is a mere observation or remark made by the Court, by way of aid, while deciding the actual issue" - Arun Kumar Agrawal v. State of MP
```

**Strength Gradation Signals (certainty levels):**
- **STRONGEST**: "We hold", "We declare", "We affirm", "is unconstitutional", "is void"
- **STRONG**: "We are of the considered view", "We are satisfied", "We are clearly of the opinion"
- **MODERATE**: "We are of the opinion", "In our view", "We are persuaded"
- **TENTATIVE**: "We are inclined to think", "prima facie", "it appears", "would seem"
- **WEAK**: "We note", "We observe", "We may mention", "arguably"

### 1.3 Multi-Part Test Application Signals

**Pattern: proportionality_structuring**
```
Signals: ["legitimate aim", "legitimate goal", "of sufficient importance to warrant overriding", "rationally connected to", "suitable means of furthering this goal", "suitability or rational connection stage", "necessary in that there are no alternative measures", "least restrictive means", "necessity stage", "proportionality stricto sensu", "balancing stage", "proper relation between"]
Creates edge: (Impugned_Measure) --[EVALUATED_UNDER]--> (Proportionality_Prong)
Confidence: high
Example: "a four-fold test: (a) legitimate goal stage, (b) suitability or rational connection stage, (c) necessity stage, (d) balancing stage" - K.S. Puttaswamy/Aadhaar (2018)
```

**Pattern: test_prong_navigation**
```
Signals: ["Coming to the first limb", "On the first prong", "As regards the first requirement", "turning to the second test", "on the question of", "insofar as the requirement of...is concerned", "having dealt with...we now turn to", "satisfies the first condition", "fails on the second prong", "stands satisfied", "does not stand the scrutiny of"]
Creates edge: (Test_Prong) --[RESULT]--> (Pass/Fail)
Confidence: high
```

**Pattern: alternative_sufficiency (arguendo)**
```
Signals: ["Even assuming", "Even if we accept", "Assuming without conceding", "Even otherwise", "For the sake of argument, even if", "arguendo", "Even on the assumption that", "Granting that", "Be that as it may", "Even taking the case at its highest"]
Creates edge: (Alternative_Argument) --[FAILS_INDEPENDENTLY]--> (Separate_Ground)
Confidence: high
```

---

## 2. Precedent Treatment Patterns

### 2.1 Core Precedent Relations

**Pattern: following_precedent**
```
Signals: ["As held in", "Following the ratio in", "We respectfully agree with the view taken in", "The law laid down by this Court in", "This view has been followed and reiterated by", "The decision is binding on this Court", "We approve the decision in", "Judicial discipline obliges them to follow it", "The principle laid down in [case] governs the present case"]
Creates edge: (Precedent) --[FOLLOWS]--> (Current_Holding)
Confidence: high
Example: "judicial discipline obliges them to follow it, regardless of their doubts about its correctness" - Central Board of Dawoodi Bohra Community (2005)
```

**Pattern: distinguishing_precedent**
```
Signals: ["The facts in the present case are distinguishable", "That case turned on", "The said decision is not applicable in the present case because", "That case was concerned with [X], whereas the present case involves [Y]", "The ratio of that decision has no application to the facts of this case", "[Case] stands on a different footing", "The said ruling is not squarely applicable"]
Creates edge: (Precedent) --[DISTINGUISHED]--> (Current_Holding)
Confidence: high
Example: "The facts of the present case is clearly distinguishable from the facts of the case dealt with by the Hon'ble Supreme Court" - A.Radhika vs Wilson Sundararaj
```

**Pattern: doubting_precedent**
```
Signals: ["We have reservations about the correctness of", "The correctness of... needs reconsideration", "We express our doubt about the view taken in", "We are not inclined to follow", "With great respect, we are unable to agree with", "The matter may be placed for hearing before a Bench consisting of a quorum larger", "We direct that the matter be referred to a larger Bench"]
Creates edge: (Precedent) --[DOUBTED]--> (Current_Holding)
Confidence: high
```

**Pattern: overruling_precedent**
```
Signals: ["We overrule the decision in", "The law laid down in... is no longer good law", "The earlier decision stands overruled", "We depart from the previous decision", "We disapprove the decision in", "The decision in [case] cannot be treated as good law", "The majority view was erroneous", "The law stated in... is hereby overruled"]
Creates edge: (Precedent) --[OVERRULED]--> (Current_Holding)
Confidence: high
Example: "There is nothing in our Constitution which prevents us from departing from a previous decision if we are convinced of its error" - Bengal Immunity Co. v. State of Bihar (1955)
```

**Pattern: explaining_precedent**
```
Signals: ["The true ratio of that decision is", "That case must be understood as", "The ratio decidendi of the judgment is", "What the Court meant to say in that case was", "The decision is only an authority for what it actually decides", "A decision is an authority for what it decides and not what can logically be deduced therefrom"]
Creates edge: (Precedent) --[EXPLAINED]--> (Current_Holding)
Confidence: high
```

### 2.2 Binding vs. Persuasive Authority

**Pattern: binding_larger_bench**
```
Signals: ["The law laid down by this Court in a decision delivered by a Bench of larger strength is binding on any subsequent Bench of lesser or coequal strength", "A Bench of lesser quorum cannot disagree", "The decision of a Constitution Bench binds", "The Bench strength is determinative of the binding nature"]
Creates edge: (Larger_Bench_Decision) --[BINDS]--> (Smaller_Bench)
Confidence: high
```

**Pattern: article_141_binding**
```
Signals: ["The law declared by the Supreme Court shall be binding on all courts within the territory of India", "Article 141 of the Constitution mandates", "Article 141 which lays down that the law declared by this Court shall be binding"]
Creates edge: (Supreme_Court_Decision) --[BINDING_Art141]--> (All_Courts_India)
Confidence: high
```

**Pattern: persuasive_only**
```
Signals: ["The decision has only persuasive value", "None of these decisions are binding upon Supreme Court but they are authorities of high persuasive value", "Foreign judgments and obiter dicta are not binding", "This can be accepted as a persuasive precedent", "A persuasive precedent is one which the judges are under no obligation to follow"]
Creates edge: (Foreign_Judgment/High_Court/Obiter) --[PERSUASIVE]--> (Current_Court)
Confidence: high
```

**Pattern: per_incuriam**
```
Signals: ["A decision should be treated as given per incuriam when it is given in ignorance of the terms of a statute", "The decision is per incuriam", "A decision rendered by ignorance of a previous binding decision", "Such a decision would not be binding as a judicial precedent"]
Creates edge: (Per_Incuriam_Decision) --[NOT_BINDING]--> (Current_Court)
Confidence: high
```

**Pattern: sub_silentio**
```
Signals: ["A decision passes sub silentio, in the technical sense, when the particular point of law involved in the decision is not perceived by the court", "Precedents sub silentio are not regarded as authoritative", "A decision which is not express and is not founded on reasons"]
Creates edge: (Sub_Silentio_Decision) --[NOT_AUTHORITATIVE]--> (Current_Court)
Confidence: high
```

---

## 3. Doctrine Dependency Structures

### 3.1 Constitutional Doctrines

**Doctrine: Basic Structure**
```
Type: constitutional_principle
Parent_doctrine: None (sui generis)
Establishing_cases: Kesavananda Bharati v. State of Kerala (1973), Minerva Mills v. UOI (1980), I.R. Coelho v. State of TN (2007)
Requires: [AND]
  - Constitutional amendment under Article 368
  - Amendment "damages/emasculates/destroys/abrogates" basic features
Alternatives: [OR - any basic feature violation suffices]
  - Judicial review power violated
  - Federalism destroyed
  - Secularism abrogated
  - Democratic structure removed
  - Rule of law eliminated
Defeaters:
  - Amendment does not alter basic features
  - Incidental effect without substantial damage
  - Amendment pre-dates April 24, 1973 (per Waman Rao)
Typical_edge_pattern: (Constitutional_Amendment) --[challenges]--> (Basic_Structure) --[if violated]--> (Amendment_Void)
Key_phrases: "emasculate basic features", "destroy identity of Constitution", "beyond Parliament's constituent power"
```

**Doctrine: Proportionality Test**
```
Type: judicial_review_standard
Parent_doctrine: Due Process/Maneka Gandhi Framework
Establishing_cases: Modern Dental College v. State of MP (2016), K.S. Puttaswamy I (2017), K.S. Puttaswamy II/Aadhaar (2018)
Requires: [AND - four prongs]
  1. Legality: Law authorizing restriction exists
  2. Legitimate State Aim: Proper governmental purpose
  3. Rational Connection/Suitability: Measure rationally connected to aim
  4. Necessity: Least restrictive means available
  5. Balancing: Benefits exceed costs to rights
Alternatives: None - all prongs must pass
Defeaters:
  - No legitimate aim
  - Less restrictive alternatives available
  - Disproportionate impact on rights
  - No rational nexus
Typical_edge_pattern: (State_Action) --[tested against]--> (Proportionality_Test) --[if failed]--> (Unconstitutional)
Key_phrases: "legitimate aim", "rational nexus", "least restrictive means", "proportionality stricto sensu"
```

**Doctrine: Manifest Arbitrariness**
```
Type: judicial_review_standard
Parent_doctrine: Article 14 (supplements Reasonable Classification)
Establishing_cases: Shayara Bano v. UOI (2017), Navtej Singh Johar v. UOI (2018), Joseph Shine v. UOI (2018)
Requires: [OR - any one suffices]
  - Legislation is "capricious"
  - Legislation is "irrational"
  - Legislation "without adequate determining principle"
  - Legislation is "excessive and disproportionate"
  - Non-classification: failure to recognize degrees of harm
Defeaters:
  - Rational basis and determining principle present
  - Policy decision within legislative wisdom
  - Public interest justification
Typical_edge_pattern: (Legislation) --[challenged as]--> (Manifestly_Arbitrary) --[if established]--> (Violates_Art14)
Key_phrases: "manifestly arbitrary", "capriciously", "irrationally", "without determining principle"
```

**Doctrine: Maneka Gandhi Framework (Due Process)**
```
Type: constitutional_principle
Parent_doctrine: Golden Triangle (Articles 14, 19, 21 interlinked)
Establishing_cases: Maneka Gandhi v. UOI (1978), R.C. Cooper v. UOI (1970)
Requires: [AND - Golden Triangle test]
  - Procedure established by law exists
  - Procedure must be "right, just and fair"
  - Procedure must not be "arbitrary, fanciful or oppressive"
  - Must satisfy Article 14 (non-arbitrariness)
  - Must satisfy Article 19 (reasonableness if applicable)
  - Must conform to principles of natural justice
Defeaters:
  - Express statutory exclusion of natural justice
  - Emergency situations
  - Clear legislative intent excluding procedural requirements
Typical_edge_pattern: (State_Action) --[must satisfy]--> (Maneka_Gandhi_Test) --[if failed]--> (Unconstitutional)
Key_phrases: "just, fair and reasonable", "right and just and fair", "not arbitrary, fanciful or oppressive", "Golden Triangle"
```

**Doctrine: Reasonable Classification (Article 14 Twin Test)**
```
Type: judicial_review_standard
Parent_doctrine: Article 14 equality
Establishing_cases: State of WB v. Anwar Ali Sarkar (1952), Ram Krishna Dalmia v. Tendolkar (1958)
Requires: [AND - twin test]
  1. Intelligible Differentia: Clear distinguishing criteria
  2. Rational Nexus: Differentia relates to legislative object
Alternatives: None - both required
Defeaters:
  - No intelligible differentia (arbitrary grouping)
  - No rational nexus to legislative object
  - Over-inclusion/under-inclusion without justification
Typical_edge_pattern: (Classification) --[twin test]--> (Valid if both satisfied) OR (Violates Art14 if fails)
Key_phrases: "intelligible differentia", "rational nexus", "reasonable classification"
```

### 3.2 Administrative Law Doctrines

**Doctrine: Wednesbury Unreasonableness**
```
Type: judicial_review_standard (secondary review)
Parent_doctrine: Common law; supplements Article 14
Establishing_cases: Om Kumar v. UOI (2000), Tata Cellular v. UOI (1994)
Requires: [OR - any one suffices]
  - Decision based on irrelevant considerations
  - Decision ignores relevant considerations
  - Decision "so outrageous that no sensible person could have arrived at it"
Defeaters:
  - Fundamental rights involved (proportionality applies instead)
  - Decision within reasonable range of outcomes
  - Policy matter within executive domain
Typical_edge_pattern: (Administrative_Decision) --[challenged]--> (Wednesbury_Test) --[if failed]--> (Quashed)
Key_phrases: "so unreasonable that no reasonable authority", "outrageous in defiance of logic", "secondary review"
```

**Doctrine: Legitimate Expectation**
```
Type: administrative_principle (procedural fairness)
Parent_doctrine: Natural Justice; Article 14 fairness
Establishing_cases: State of Kerala v. K.G. Madhavan Pillai (1988), Food Corporation v. Kamdhenu (1993)
Requires: [OR - source of expectation]
  - Express promise or assurance by public authority
  - Consistent past practice/established pattern
  - Representation creating reasonable expectation
AND:
  - Expectation must be "legitimate" (reasonable, logical, valid)
Defeaters:
  - Overriding public interest
  - Change in statutory provisions
  - Misrepresentation by claimant
  - Expectation contrary to law
Typical_edge_pattern: (Promise/Practice) --[creates]--> (Legitimate_Expectation) --[if breached]--> (Action_Vitiated)
Key_phrases: "reasonable expectation", "consistent past practice", "express promise", "fair play in action"
```

**Doctrine: Natural Justice - Audi Alteram Partem**
```
Type: natural_justice_principle
Parent_doctrine: Articles 14, 21
Establishing_cases: Maneka Gandhi v. UOI (1978), A.K. Kraipak v. UOI (1969)
Requires: [AND]
  - Notice of proposed action
  - Reasonable opportunity to be heard
  - Fair hearing before adverse decision
  - Speaking order/reasoned decision
Defeaters:
  - Urgency/emergency (post-decision hearing may suffice)
  - Statutory exclusion (subject to Art 14 challenge)
  - Purely administrative (non-quasi-judicial) actions
  - Conclusion obvious and hearing could not make difference
Typical_edge_pattern: (Adverse_Action) --[requires]--> (Notice + Hearing) --[if omitted]--> (Order_Void)
Key_phrases: "no one shall be condemned unheard", "fair opportunity to answer", "speaking order"
```

### 3.3 Interpretive Canons

**Doctrine: Harmonious Construction**
```
Type: interpretation_canon
Establishing_cases: CIT v. Hindustan Bulk Carriers (2003) - five principles
Requires: [AND - five principles]
  1. Avoid head-on clash between provisions
  2. Construe provisions harmoniously
  3. One provision cannot defeat another unless irreconcilable
  4. Give effect to both as much as possible
  5. Construction rendering one provision dead is not harmonious
Defeaters:
  - Irreconcilable conflict
  - Clear legislative intent for one to prevail
  - Later enactment expressly overrides earlier
Typical_edge_pattern: (Conflicting_Provisions) --[harmonious construction]--> (Both_Given_Effect)
Key_phrases: "avoid head-on clash", "construe harmoniously", "neither provision rendered otiose"
```

**Doctrine: Pith and Substance**
```
Type: constitutional_principle (federalism)
Parent_doctrine: Article 246 and Seventh Schedule
Establishing_cases: State of Bombay v. F.N. Balsara (1951), Kartar Singh v. State of Punjab (1994)
Requires: [AND]
  - Challenge to legislative competence based on List entries
  - Court determines "true nature and character" of legislation
  - Dominant character must fall within enacting body's list
Defeaters:
  - Legislation in pith and substance belongs to another list
  - Cannot validate legislation if substance clearly outside jurisdiction
Typical_edge_pattern: (Legislation) --[pith and substance]--> (Valid if dominant character in correct list)
Key_phrases: "true nature and character", "dominant purpose", "incidental encroachment"
```

**Doctrine: Reading Down**
```
Type: interpretation_canon
Parent_doctrine: Presumption of constitutionality
Establishing_cases: Kedar Nath Singh v. State of Bihar (1962), State of Bombay v. FN Balsara (1951)
Requires: [AND]
  - Provision capable of two interpretations
  - One interpretation constitutional, other unconstitutional
  - Reading down possible without rewriting
  - Legislative intent not frustrated by narrower reading
Defeaters:
  - Only one reasonable interpretation possible
  - Reading down would rewrite statute
  - Core of provision is unconstitutional
Typical_edge_pattern: (Overbroad_Provision) --[read down]--> (Narrower_Constitutional_Interpretation)
Key_phrases: "read down to save constitutionality", "presumption of constitutionality"
```

---

## 4. IPC/CrPC/IEA → BNS/BNSS/BSA Mapping

### 4.1 IPC to BNS (Bharatiya Nyaya Sanhita) Key Mappings

**Homicide Offenses:**
| Old | New | Concept | Changes |
|-----|-----|---------|---------|
| IPC 302 | BNS 103 | MURDER | Renumbered |
| IPC 304 | BNS 105 | CULPABLE_HOMICIDE | Renumbered |
| IPC 304A | BNS 106 | DEATH_BY_NEGLIGENCE | **Punishment 2yr→5yr; medical practitioner provisions added** |
| IPC 304B | BNS 80 | DOWRY_DEATH | Renumbered |
| IPC 307 | BNS 109 | ATTEMPT_MURDER | Life = "remainder of natural life" clarified |

**Sexual Offenses:**
| Old | New | Concept | Changes |
|-----|-----|---------|---------|
| IPC 375 | BNS 63 | RAPE_DEFINITION | **Age of consent for married woman 15→18 (implements Independent Thought v. UOI)** |
| IPC 376 | BNS 64 | RAPE_PUNISHMENT | Renumbered |
| IPC 376D | BNS 70(1) | GANG_RAPE | Consolidated |
| NEW | BNS 69 | SEXUAL_INTERCOURSE_BY_DECEIT | **NEW - false promise of marriage, up to 10 years** |

**Property Offenses:**
| Old | New | Concept | Changes |
|-----|-----|---------|---------|
| IPC 378-379 | BNS 303/305(a) | THEFT | Renumbered |
| NEW | BNS 304 | SNATCHING | **NEW offense - "suddenly or quickly or forcibly seizes" - up to 3 years** |
| IPC 390-391 | BNS 292/310(2) | ROBBERY/DACOITY | Consolidated |
| IPC 405-409 | BNS 316(1)-(4) | CRIMINAL_BREACH_TRUST | **All CBT consolidated; punishment 3yr→5yr** |
| IPC 415-420 | BNS 318 | CHEATING | Combined |

**Sedition Replacement (MAJOR CHANGE):**
| Old | New | Concept | Changes |
|-----|-----|---------|---------|
| IPC 124A | BNS 152 | SEDITION→SOVEREIGNTY | **"Sedition" replaced with "Acts endangering sovereignty, unity and integrity." Subject: "Government"→"India." Punishment: 3yr→7yr + mandatory fine. Scope expanded: includes secession, armed rebellion, subversive activities.** |

**Other Key Mappings:**
| Old | New | Concept | Changes |
|-----|-----|---------|---------|
| IPC 34 | BNS 3(5) | COMMON_INTENTION | Now in definitions |
| IPC 499-500 | BNS 356 | DEFAMATION | **Community service added as punishment option** |
| IPC 503-506 | BNS 351 | CRIMINAL_INTIMIDATION | Consolidated |
| IPC 511 | BNS 62 | ATTEMPT | Renumbered |
| IPC 377 | REPEALED | - | Per Navtej Singh Johar (2018) |
| IPC 497 | REPEALED | - | Per Joseph Shine (2018) |

**New Offenses in BNS:**
- **Section 111**: Organized Crime (NEW)
- **Section 112**: Petty Organized Crime (NEW)
- **Section 113**: Terrorism (NEW explicit definition)
- **Section 103(2)**: Mob Lynching (NEW)

### 4.2 CrPC to BNSS Key Mappings

| Old CrPC | New BNSS | Concept | Changes |
|----------|----------|---------|---------|
| 41 | 35 | ARREST_WITHOUT_WARRANT | **35(7): No arrest without SP-level approval for elderly (60+) or infirm for offences <3 years** |
| 125 | 144 | MAINTENANCE | Renumbered |
| 154 | 173 | FIR | **Electronic filing explicit; Zero FIR statutory recognition; 173(3): preliminary inquiry for 3-7yr offences with DSP approval** |
| 161 | 180 | POLICE_EXAMINATION | Renumbered with enhancements |
| 164 | 183 | CONFESSION_RECORDING | **Audio-video recording preferred** |
| 313 | 278 | ACCUSED_EXAMINATION | Renumbered |
| 319 | 283 | ADD_ACCUSED | Renumbered |
| 378 | 419 | APPEAL_ACQUITTAL | Renumbered |
| 437 | 480 | REGULAR_BAIL | Renumbered |
| 436A | 481 | UNDERTRIAL_DETENTION | **First-time offender: bail after 1/3 of maximum sentence; 481(2): multiple pending cases = ineligible** |
| 438 | 482 | ANTICIPATORY_BAIL | **438(1A)/(1B) OMITTED; 482(4): No anticipatory bail for rape of woman under 18 (was 16)** |
| 439 | 483 | BAIL_HC_SESSIONS | Renumbered |
| 482 | 528 | INHERENT_POWERS_HC | Renumbered |

### 4.3 IEA to BSA Key Mappings

| Old IEA | New BSA | Concept | Changes |
|---------|---------|---------|---------|
| 3 | 2 | DEFINITIONS | **"Document" explicitly includes electronic/digital records** |
| 6 | 4 | RES_GESTAE | "or relevant facts" added |
| 24-29 | 22 | CONFESSION_ADMISSIBILITY | **CONSOLIDATED: "Coercion" explicitly added; clearer timeline** |
| 25-26 | 23 | CONFESSION_POLICE | Combined |
| 27 | 23 (proviso) | DISCOVERY_CONFESSION | Included as proviso |
| 30 | 24 | CO_ACCUSED_CONFESSION | **Trial-in-absentia = joint trial for this section** |
| 32 | 26 | DYING_DECLARATION | **Restructured into 8 subsections** |
| 45 | 39 | EXPERT_OPINION | **"Any other field" added; Examiner of Electronic Evidence (IT Act) recognized** |
| 65B | 63 | ELECTRONIC_EVIDENCE | **MAJOR: "Communication device" explicit; MANDATORY certificate in prescribed format; HASH value required; signed by person in charge AND expert** |
| 101-104 | 104-107 | BURDEN_OF_PROOF | Renumbered |
| 114 | 119 | COURT_PRESUMPTION | Renumbered |

---

## 5. Reasoning Chain Templates

### 5.1 Fundamental Rights Challenge (Articles 14/19/21)

```
Case_type: Fundamental_Rights_Challenge

Entry_facts:
- State action/law affecting life, liberty, equality
- Executive order impinging on personal liberty
- Legislation imposing restrictions on freedoms
- Government policy creating discriminatory classifications
- Law appearing arbitrary or without rational basis

Required_concepts: [in order]
1. Maneka Gandhi doctrine - Golden Triangle interconnection
2. "Right, just, fair and reasonable" procedure test
3. Reasonable Classification (intelligible differentia + rational nexus)
4. Manifest Arbitrariness (capricious, irrational, without determining principle)
5. Proportionality (legitimate aim → suitability → necessity → balancing)

Typical_issues_framed:
- "Whether the impugned law/action violates Article 14 by being manifestly arbitrary?"
- "Whether the procedure satisfies the triple test under Articles 14, 19 and 21?"
- "Whether the classification has intelligible differentia with rational nexus?"
- "Whether the restriction on fundamental right is reasonable and proportionate?"

Decision_factors:
PETITIONER WINS IF: No intelligible differentia | Arbitrary procedure | Fails proportionality | Manifestly arbitrary
PETITIONER LOSES IF: Reasonable basis | Fair procedure | Proportionate means | Legislative wisdom

Precedent_cluster: Maneka Gandhi (1978), E.P. Royappa (1974), Shayara Bano (2017), Navtej Singh Johar (2018), K.S. Puttaswamy (2017), Joseph Shine (2018)

Reasoning_flow:
Step 1: Identify FR allegedly violated → Step 2: Apply Maneka Gandhi (read together) → Step 3: Art 14: twin test + manifest arbitrariness → Step 4: Art 21: right/just/fair procedure → Step 5: Art 19: proportionality of restrictions → Step 6: Balance individual rights vs. state interest → Step 7: Determine validity
```

### 5.2 Administrative Action Review

```
Case_type: Administrative_Action_Review

Entry_facts:
- Rejection of bid/tender
- Contract cancellation without hearing
- Non-renewal of license/permit
- Transfer/posting orders challenged
- Disciplinary action in service matters

Required_concepts: [in order]
1. Tata Cellular principles - Limited scope of judicial review
2. Wednesbury unreasonableness (illegality, irrationality, procedural impropriety)
3. Principles of natural justice
4. Legitimate expectation
5. Jagdish Mandal two-prong test

Typical_issues_framed:
- "Whether the decision-making process was vitiated by arbitrariness or procedural impropriety?"
- "Whether principles of natural justice were violated?"
- "What is the scope of judicial review in contractual/tender matters?"

Decision_factors:
PETITIONER WINS IF: Audi alteram partem violated | Bias/mala fides | Wednesbury unreasonable | Procedural norms violated
PETITIONER LOSES IF: Decision within discretion | Mere dissatisfaction | Technical evaluation correct | Public interest justifies

Precedent_cluster: Tata Cellular v. UOI (1994), Jagdish Mandal v. State of Orissa (2007), Ramana Dayaram Shetty (1979)

Reasoning_flow:
Step 1: Identify nature of action → Step 2: Apply Tata Cellular (judicial restraint) → Step 3: Test PROCESS not decision: illegality/irrationality/procedural impropriety → Step 4: Check natural justice compliance → Step 5: Jagdish Mandal test → Step 6: Balance private grievance vs. public interest
```

### 5.3 Criminal Appeal (Conviction/Sentencing)

```
Case_type: Criminal_Appeal

Entry_facts:
- Appeal against conviction by Sessions/High Court
- State challenge to acquittal
- Death sentence confirmation/commutation
- Evidence appreciation challenge

Required_concepts: [in order]
1. Proof beyond reasonable doubt
2. Evidence appreciation (separation of grain from chaff)
3. Circumstantial evidence (chain must be complete)
4. Bachan Singh framework (aggravating vs. mitigating)
5. Rarest of rare doctrine

Typical_issues_framed:
- "Whether the prosecution has proved its case beyond reasonable doubt?"
- "Whether the circumstantial evidence forms a complete chain pointing to guilt?"
- "Whether the case falls within the 'rarest of rare' category?"

Decision_factors:
CONVICTION UPHELD IF: Proof beyond reasonable doubt | Credible eye-witness | Complete circumstantial chain | Medical/forensic support
CONVICTION OVERTURNED IF: Reasonable doubt exists | Unreliable witnesses | Incomplete chain | Two views possible (benefit of doubt)
DEATH PENALTY UPHELD IF: Rarest of rare | Aggravating > mitigating | No reformation possible
DEATH PENALTY COMMUTED IF: Mitigating factors | Delay | Reformation possible

Precedent_cluster: Bachan Singh (1980), Machhi Singh (1983), Shatrughan Chauhan (2014), Chandrappa (2007)
```

### 5.4 Statutory Interpretation

```
Case_type: Statutory_Interpretation

Entry_facts:
- Ambiguity in statutory language
- Conflict between provisions
- Question of legislative intent
- Tax/fiscal statute interpretation
- Beneficial legislation interpretation

Required_concepts: [in order]
1. Literal rule (plain meaning)
2. Golden rule (avoid absurdity)
3. Mischief rule/Purposive construction
4. Harmonious construction (five principles)
5. Ejusdem generis, noscitur a sociis

Decision_factors:
LITERAL APPLIES IF: Clear language | No absurdity | No conflict
PURPOSIVE APPLIES IF: Ambiguous language | Absurd result | Two meanings possible | Beneficial legislation

Precedent_cluster: CIT v. Hindustan Bulk Carriers (2003), Bengal Immunity (1955), Venkataramana Devaru (1958)

Reasoning_flow:
Step 1: Identify interpretive question → Step 2: Literal rule first → Step 3: If ambiguous, contextual interpretation → Step 4: If absurd, purposive construction → Step 5: If conflict, harmonious construction → Step 6: Consider statute type (penal=strict, beneficial=liberal)
```

### 5.5 Federalism/Legislative Competence

```
Case_type: Legislative_Competence_Challenge

Entry_facts:
- State law encroaching on Union List
- Central law encroaching on State List
- Concurrent List conflict
- Presidential assent under Article 254(2) at issue

Required_concepts: [in order]
1. Article 246, Seventh Schedule
2. Pith and Substance doctrine
3. Colorable Legislation
4. Repugnancy doctrine (Article 254)
5. Occupied Field doctrine

Typical_issues_framed:
- "Whether the impugned legislation falls within legislative competence?"
- "What is the pith and substance of the impugned law?"
- "Whether there is repugnancy between State and Central law?"

Decision_factors:
LAW UPHELD IF: Pith and substance within competence | Incidental encroachment minor | Field not occupied
LAW STRUCK DOWN IF: Substance outside competence | Colorable exercise | Irreconcilable repugnancy | Field fully occupied

THREE TESTS FOR REPUGNANCY (Deep Chand):
1. Direct conflict - impossible to obey both
2. Parliament intended exhaustive code
3. Both laws occupy same field

Precedent_cluster: State of Bombay v. F.N. Balsara (1951), M. Karunanidhi v. UOI (1979), Deep Chand v. State of UP (1959)
```

---

## 6. Long-Distance and Counterfactual Reasoning Patterns

### 6.1 Deferred Application Patterns

**Pattern: deferred_application**
```
Signals: ["We shall deal with this contention later", "We shall presently see", "We shall revert to this aspect", "This issue shall be addressed presently", "We shall advert to this aspect subsequently", "as will be discussed presently"]
Resolution_logic: Search FORWARD for resumption markers ("Reverting to...", "Coming now to...", "We now address...") or topic-matching paragraphs
Graph_implication: Creates DEFERRED_EDGE; marks node with "awaits_resolution" property
Example: "As we shall presently see, there is nothing said in either case from which it may be possible to hold that the law laid down... is 'no longer good law'" - Supreme Court Bar Association v. UOI (1998)
```

### 6.2 Back-Reference Patterns

**Pattern: noted_above**
```
Signals: ["As noted above", "As stated above", "As already observed", "As discussed above", "As indicated earlier", "From the foregoing discussion"]
Resolution_logic: Search BACKWARD for paragraphs containing explicit legal holdings, factual findings, or referenced precedent; semantic similarity + entity matching
Graph_implication: Creates BACK_REFERENCE edge; current reasoning DEPENDS on earlier established facts/law
```

**Pattern: aforesaid_facts**
```
Signals: ["The aforesaid facts", "The aforesaid judgment", "The above discussion", "From the admitted position", "The admitted facts", "The undisputed facts"]
Resolution_logic: Resolve to Facts section or most recent factual narrative; "aforesaid" = immediately preceding material
Graph_implication: Creates FACTUAL_DEPENDENCY edge; holdings CONDITIONAL on factual findings
```

### 6.3 Doctrinal Chain Patterns

**Pattern: sequential_doctrine**
```
Signals: ["Having held that... the question of... assumes importance", "Once it is established that... the consequence is", "Having so held... we proceed to consider"]
Resolution_logic: Doctrine A → Doctrine B dependency; track "having held" to identify primary doctrine
Graph_implication: Creates DOCTRINAL_CHAIN edge (transitive support); CRITICAL - if Doctrine A fails, chain collapses
Example: "Having held that Article 21 is violated, the question of proportionality assumes importance"
```

**Pattern: golden_triangle_interconnection**
```
Signals: ["The trinity exists between Article 14, Article 19 and Article 21", "read together", "taken together", "interplay between"]
Resolution_logic: Multiple provisions form interconnected doctrine; violation of one triggers all
Graph_implication: Creates CONJUNCTIVE_DEPENDENCY edges; mutually reinforcing doctrinal base
Example: "A seven-Judge Bench held that a trinity exists between Article 14, Article 19 and Article 21. All these articles have to be read together" - Maneka Gandhi
```

### 6.4 Counterfactual Markers

**Pattern: hypothetical_concession**
```
Signals: ["Even if we were to accept", "Even assuming", "Even granting", "Assuming but not deciding", "Assuming without deciding", "Proceeding on the assumption that", "For argument's sake", "Granting for the moment"]
Resolution_logic: Rhetorical concession; judge provisionally accepts premise but shows conclusion fails anyway
Graph_implication: Creates HYPOTHETICAL_CONCESSION edge; ALTERNATIVE_REASONING - holding STRENGTHENED not dependent on this path
Example: "Even assuming the petitioner's contention to be correct, the relief cannot be granted for procedural reasons"
```

**Pattern: but_for_causation**
```
Signals: ["Had the authority granted a hearing", "If the procedure had been followed", "But for the defect in", "Had the notice been properly served", "Were it not for the procedural irregularity"]
Resolution_logic: Identifies causal factor - if X happened differently, outcome Y would follow; indicates what was DETERMINATIVE
Graph_implication: Creates DETERMINATIVE_FACTOR edge; CRITICAL causal link
Example: "Had this last opportunity to produce the accused been afforded, the order was legal and valid" - Narata Ram v. State of H.P.
```

**Pattern: distinguishing_counterfactual**
```
Signals: ["The position would have been different if", "Had the facts been", "Unlike the present case", "The distinguishing feature", "In contradistinction to"]
Resolution_logic: Judge identifies why cited precedent doesn't apply; track precedent + factual difference + why material
Graph_implication: Creates DISTINGUISHING_EDGE; marks precedent as INAPPLICABLE with reason_node attached
```

**Pattern: single_factor_determinism**
```
Signals: ["On this ground alone", "This alone is sufficient to", "This by itself is fatal to", "This single circumstance", "For this reason alone", "This ground independently suffices"]
Resolution_logic: ONE factor is SUFFICIENT for outcome; remaining factors are surplus
Graph_implication: Creates DETERMINATIVE edge marked as SUFFICIENT; other edges marked as SURPLUS_TO_REQUIREMENTS - **HIGH VALUE for identifying CRITICAL edges**
Example: "Therefore, it is on this ground alone the impugned order deserves to be set aside" - Prapbakaran (2010)
```

**Pattern: sufficient_conditions_disjunctive**
```
Signals: ["Any one of these grounds suffices", "Each of these is independently sufficient", "On either of these grounds"]
Resolution_logic: Multiple independent grounds - holding survives if ANY one stands
Graph_implication: Creates DISJUNCTIVE_SUPPORT structure; OR-connected edges where holding survives if any single edge valid
```

**Pattern: sufficient_conditions_conjunctive**
```
Signals: ["These grounds cumulatively establish", "Taken together, these factors", "The combined effect of these", "All these factors point to"]
Resolution_logic: Multiple factors together create sufficient grounds - no single factor sufficient alone
Graph_implication: Creates CONJUNCTIVE_SUPPORT structure; AND-connected edges where ALL must hold
```

---

## 7. Edge Type Summary for Graph Schema

| Relation | Source_Type | Target_Type | Trigger Pattern Category |
|----------|-------------|-------------|--------------------------|
| TRIGGERS | Fact | Doctrine | doctrine_invocation |
| DERIVED_FROM | Right | Constitutional_Provision | constitutional_nexus |
| FAILS | Fact_Pattern | Legal_Requirement | requirement_failure |
| JOINTLY_SATISFY | Multiple_Facts | Test | conjunctive_satisfaction |
| SUFFICES_FOR | Single_Fact | Test | disjunctive_sufficiency |
| CONCLUDES_WITH | Analysis | Holding | primary_holding |
| CREATES | Holding | Binding_Precedent | ratio_marker |
| MARKED_AS | Observation | Obiter_Dictum | obiter_marker |
| EVALUATED_UNDER | Measure | Proportionality_Prong | proportionality_structuring |
| FOLLOWS | Precedent | Current_Holding | following_precedent |
| DISTINGUISHED | Precedent | Current_Holding | distinguishing_precedent |
| DOUBTED | Precedent | Current_Holding | doubting_precedent |
| OVERRULED | Precedent | Current_Holding | overruling_precedent |
| EXPLAINED | Precedent | Current_Holding | explaining_precedent |
| BINDS | Larger_Bench | Smaller_Bench | binding_larger_bench |
| PERSUASIVE | Foreign/HC/Obiter | Current_Court | persuasive_only |
| NOT_BINDING | Per_Incuriam_Decision | Current_Court | per_incuriam |
| DEFERRED | Current_Paragraph | Future_Paragraph | deferred_application |
| BACK_REFERENCE | Current_Paragraph | Earlier_Paragraph | noted_above/aforesaid |
| DOCTRINAL_CHAIN | Doctrine_A | Doctrine_B | sequential_doctrine |
| HYPOTHETICAL | Alternative_Argument | Holding | hypothetical_concession |
| DETERMINATIVE | Causal_Factor | Outcome | but_for/single_factor |
| DISTINGUISHING | Cited_Precedent | Inapplicability | distinguishing_counterfactual |

---

## Implementation Notes

**Priority for Extraction:**
1. **Primary holding markers** ("We hold") → highest confidence for holding extraction
2. **Single-factor determinism** phrases → identify CRITICAL edges
3. **Hypothetical concession** markers → identify robust/alternative reasoning (not dependencies)
4. **Doctrinal chain** signals → reconstruct transitive support structure

**Context Sensitivity:**
- "In our view" may be obiter if not followed by dispositional language
- Always check for negators ("We do not hold", "cannot be said to")
- Multi-judge opinions require attribution tracking ("concurring", "dissenting", "speaking for myself")

**Proportionality Analysis:**
- Four-prong test often appears sequentially - track progression through stages
- Each prong creates separate edge to test component

**Criminal Code Mapping:**
- Effective July 1, 2024 - cases registered before continue under old codes
- Dual framework during transition period

This reference manual provides the extraction rules, edge mappings, and linguistic patterns needed to convert Indian Supreme Court judgments into structured reasoning graphs supporting counterfactual analysis.