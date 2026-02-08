# Evaluation Protocol

Formal documentation of the evaluation methodology used in this multi-agent research system.

## Metric Definitions

### 1. Claim-to-Source Grounding Accuracy

**Definition**: Percentage of sentence-level factual claims that can be traced to at least one retrieved document chunk.

**Measurement Method**:
1. Extract factual claims from generated text (sentences with assertive statements, statistics, or citations)
2. Compare each claim against retrieved source chunks
3. Use semantic similarity (embedding cosine similarity ≥ 0.5) or keyword overlap as matching criteria
4. Calculate: `grounded_claims / total_claims`

**Scope**: Internal evaluation, not an external benchmark.

---

### 2. Hallucinated Citation Rate

**Definition**: Percentage of cited references that do not correspond to real retrieved documents.

**Measurement Method**:
1. Extract all citations from generated text (URLs, markdown links, academic-style citations)
2. Validate each citation against the retrieval log
3. Citations are "valid" if they match a source URL, title, or are numbered references with sources present
4. Calculate: `hallucinated_citations / total_citations`

**Scope**: Automated validation against retrieval logs.

---

### 3. Structural Coherence Score

**Definition**: Normalized internal rubric evaluating logical section ordering, topic adherence, and cross-section consistency.

**Measurement Method**:
1. Score individual sections on quality heuristics (length, structure, title relevance)
2. Evaluate section-to-section flow using semantic similarity
3. Check for topic drift between introduction and conclusion
4. Validate structure against the generated outline (if available)
5. Combine into weighted score: `0.3×sections + 0.25×flow + 0.25×(1-redundancy) + 0.1×(1-drift) + 0.1×structure`

**Range**: 0 to 1, where 1 indicates perfect coherence.

**Note**: This is NOT a standardized NLP benchmark metric (e.g., BLEU, ROUGE).

---

### 4. Redundancy Rate

**Definition**: Percentage of section pairs with high semantic overlap indicating repeated content.

**Measurement Method**:
1. Compute semantic embeddings for each section
2. Calculate pairwise cosine similarities
3. Flag pairs with similarity > 0.7 as redundant
4. Calculate: `redundant_pairs / total_pairs`

---

### 5. Generation Time

**Definition**: Wall-clock time from workflow start to final report completion.

**Measurement**: Recorded per run by the metrics logging system.

---

## Evaluation Conditions

- Topics sampled from technical and scientific domains
- Report length: 8–12 pages (approximately 4,000–6,000 words)
- Evaluation conducted on sampled outputs, not exhaustive corpus-wide claims
- Hardware: RTX 2080 Ti (11GB VRAM), Intel Core i9, 32GB RAM

---

## Interpretation Guidelines

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Grounding Accuracy | <70% | 70-85% | 85-95% | >95% |
| Hallucination Rate | >10% | 5-10% | 2-5% | <2% |
| Coherence Score | <0.6 | 0.6-0.75 | 0.75-0.9 | >0.9 |
| Redundancy Rate | >20% | 10-20% | 5-10% | <5% |

---

## What This Evaluation Does NOT Claim

- **State-of-the-art performance** on standard benchmarks
- **Factual accuracy** beyond retrieved sources
- **Generalization** to domains not covered by retrieval
- **Replacement** of human peer review

This transparency strengthens credibility by clearly scoping what the system can and cannot guarantee.
