A reliable ModernBERT implementation begins with **data governance, leakage-resistant splits, hierarchical labels, and calibrated abstention**—not just a fine-tuning script. Synthetic examples can help sparse subclasses, but only as provenance-tracked, independently filtered supplements to authentic documents; they must never contaminate validation or test sets. ModernBERT-base is suitable for this task because it is a 149M-parameter encoder with native support for sequences up to 8,192 tokens, substantially reducing the number of windows needed for long legal and business documents.[1][2]

## Objectives and design

The training system should optimize for the pipeline’s actual operational goal:

1. Correctly identify the **primary document class**.
2. Predict a subclass only when the parent class is credible.
3. Recognize uncertainty and route difficult, messy, novel, or weakly supported documents to the existing LLM intake/sorter path.
4. Preserve the pipeline’s mandatory deterministic normalization and no-truncation doctrine.
5. Improve minority-class performance without teaching the model synthetic artifacts, copied templates, or misleading filename shortcuts.[2][3]

The recommended production design is a **shared ModernBERT encoder with a hierarchical multitask classifier**, not a flat 40+ label classifier and not six independently hosted full model copies.

\[
\text{Document} \rightarrow \text{ModernBERT encoder}
\rightarrow
\begin{cases}
\text{primary document-type head}\\
\text{conditional subclass head}
\end{cases}
\rightarrow
\text{calibration + abstention}
\rightarrow
\text{fast path or LLM fallback}
\]

This aligns with the mailroom taxonomy: document type is a broad, data-rich decision; subclasses are meaningful only inside their primary class. For example, a contract subtype should never compete in the same output layer with an insurance-claim subtype.[2]

## Dataset preparation

### Immutable dataset snapshot

Create a reproducible training manifest before writing any training code. The ML working copy is `Lucius-Morningstar/mailroom-finetune` @ `19720ceb4e29bc3134a88507aa57cdfac7a64a1b` (byte-schema identical to the canonical `mailroom-dataset` eval corpus); pin the exact revision rather than training from a mutable `main` branch. The prior proposal identifies 2,979 train rows and a separate 323-row test split; retain that test split as a fully untouched final evaluation set.[2][3]

Each training run should write a manifest like:

```json
{
  "dataset_repo": "Lucius-Morningstar/mailroom-finetune",
  "dataset_revision": "<immutable-commit-sha>",
  "dataset_config": "ground_truth",
  "train_source_split": "train",
  "final_test_source_split": "test",
  "label_schema_version": "mailroom-taxonomy-v1",
  "preprocessing_version": "modernbert-prep-v1",
  "split_seed": 42,
  "synthetic_policy_version": "synthetic-v1"
}
```

Store this manifest alongside the model artifact, tokenizer, calibration parameters, label maps, synthetic-data ledger, and training configuration.

### Canonical record schema

Create a normalized parquet or JSONL training table with one row per original document. Do not make windows the primary records until after document-level splitting is complete.

```json
{
  "document_id": "stable-content-or-source-id",
  "source_split": "train",
  "source_origin": "authentic",
  "filename": "example_agreement.pdf",
  "title": "Agreement and Plan of Merger",
  "text_raw": "...",
  "text_normalized": "...",
  "primary_doc_class": "merger_agreement",
  "doc_subclass": "all_cash",
  "content_sha256": "...",
  "family_group_id": "...",
  "char_count": 81840,
  "synthetic_parent_ids": [],
  "dataset_revision": "<sha>"
}
```

For generated examples, `source_origin` must be `synthetic`, with generator, model, prompt-template, reviewer, filter outcomes, and acceptance timestamp recorded separately.

### Label normalization

Use the same taxonomy and normalization code that production uses. Do not duplicate label aliases in a training notebook.

Normalize:

- Case: `Service` → `service`.
- Separator differences: `Co_Branding` → `co_branding`.
- Whitespace and punctuation variants.
- Deprecated aliases.
- Missing subclass labels.
- Invalid parent/subclass combinations.

Create and version a parent-to-subclass map:

```python
SUBCLASSES_BY_PARENT = {
    "contract": [...],
    "merger_agreement": [...],
    "corporate_record": [...],
    "correspondence": [...],
    "insurance_claim": [...],
}
```

A record with an invalid subclass must be quarantined for review; it should not be silently relabeled. The goal is for the training-time labels and live pipeline taxonomy to be mathematically identical.[2]

### Deterministic normalization

Apply the existing deterministic intake normalization before tokenization:

- Whitespace normalization.
- Non-breaking-space normalization.
- Hyphen and character cleanup.
- OCR artifact counters.
- Repeated header/footer indicators.
- Input emptiness checks.

The classifier should train on what it will receive in production: normalized text plus filename/title context. The deterministic clerk remains mandatory and never gets bypassed.[3]

Keep both `text_raw` and `text_normalized`. Never overwrite raw source text. This preserves traceability, allows preprocessing revision comparisons, and lets you audit whether normalization accidentally removes class-specific cues.

### Data quality audit

Before splitting or augmentation, generate a data-quality report containing:

| Audit | Required check |
| --- | --- |
| Class counts | Primary-class and subclass count distribution |
| Missing labels | Null/unknown type and subclass rates |
| Lengths | Character and token quantiles by class |
| Duplicates | Exact hash duplicates and near duplicates |
| Label consistency | Same content with conflicting labels |
| Filename leakage | Label predictability from filename alone |
| Title leakage | Label predictability from title alone |
| Parent/subclass validity | Every subclass belongs to its assigned parent |
| Text quality | Empty, OCR-heavy, garbled, or malformed documents |
| Family leakage | Related documents, amendments, exhibits, templates, transactions |

The filename/title audit is especially important because the mailroom corpus intentionally follows a “title-wins doctrine.” Titles are valid predictive information in production, but a benchmark can become unrealistic if near-identical names, template families, or folders appear in both train and test.[2]

### Grouped, leakage-resistant splits

Do **not** randomly split document rows first. A legal-mailroom corpus is likely to contain highly related documents: amendments, exhibits, template variants, correspondence threads, repeated forms, and transaction families.

Build a `family_group_id` from available metadata and similarity signals:

- Normalized filename stem.
- Parent folder or dataset source group.
- Text fingerprint / MinHash similarity.
- Near-duplicate title.
- Shared transaction, company, or filing identifier if present.
- Repeated boilerplate/template cluster.

Then split by group, not row.

Recommended split strategy inside the existing training set:

| Partition | Role | Approximate share | Contains synthetic data? |
| --- | --- | ---: | --- |
| Fit/train | Parameter learning | 80% | Yes, after filtering |
| Validation | Early stopping/model selection | 10% | No |
| Calibration | Temperature fitting and routing thresholds | 10% | No |
| Official test | Final, locked report only | Existing 323 rows | No |

Use a grouped, stratified split by primary class. For rare subclasses that cannot be represented cleanly in every partition, document the constraint rather than breaking grouping to force an artificial balance.

No item derived from, paraphrased from, or semantically near-duplicated from validation or test material may appear in synthetic data.

### Long-document representation

ModernBERT supports up to 8,192 tokens natively. It was pretrained initially at 1,024 tokens and extended to 8,192 tokens, so it is materially better suited to this corpus than conventional 512-token encoder workflows, although it still cannot ingest documents with hundreds of thousands of characters in one pass.[1][4]

Use a structured input prefix in every window:

```text
[FILE_NAME] acquisition_agreement_2025.pdf
[TITLE] Agreement and Plan of Merger
[DOCUMENT_CLASS_CANDIDATE] unknown
[WINDOW] 2/5
[CHAR_OFFSET] 28400

[TEXT]
...
```

Do not insert a label-derived candidate in real training/inference inputs; the example label field above should remain `unknown` or be omitted. The key fields are filename, title, window position, and normalized text.

Use the repository’s paragraph-aware `sliding_windows()` approach:

- Split at paragraph boundaries where possible.
- Use hard splits only for pathological paragraphs larger than the budget.
- Retain all text; never truncate the source document.
- Use approximately 10–15% overlap so a clause at a window boundary appears in both windows.
- Include title/filename in every window.
- Retain each window’s absolute source offset.
- Split documents into train/validation/test **before** creating windows.

For training, avoid allowing extremely long documents to dominate gradients. Use a deterministic window policy:

- Always include the title + head window.
- Include one or more middle windows based on length.
- Include a tail/signature window for documents where closings, execution pages, or exhibit labels matter.
- Include every window for rare labels or difficult examples, with document-level weighting.
- Keep the full coverage behavior at inference; training can use carefully sampled windows to control computational cost.

Long-document classification research commonly uses segment-then-aggregate or hierarchical designs because no fixed context window covers every long document. Legal-document research likewise shows that long-context approaches and segmented strategies can outperform simple TF-IDF baselines, with training typically using early stopping on held-out development performance.[5][6][7]

## Model and training recipe

### Architecture

Use `answerdotai/ModernBERT-base` as the shared encoder:

- 22 layers.
- 149 million parameters.
- Native maximum context of 8,192 tokens.
- Encoder-only architecture appropriate for fixed-taxonomy classification.
- Apache-2.0 licensing according to the model card.[1][2]

Attach the following heads to one encoder:

| Head | Output labels | Applies when |
| --- | --- | --- |
| `doc_type` | 5 primary classes, optionally an abstention/unknown policy | Always |
| `contract_subclass` | CUAD family labels + `other` | Contract is selected/plausible |
| `merger_subclass` | All-cash, all-stock, mixed, other | Merger agreement is selected/plausible |
| `corporate_record_subclass` | Bylaws, articles, powers, etc. | Corporate record is selected/plausible |
| `correspondence_subclass` | Email, letter, notice, memo, etc. | Correspondence is selected/plausible |
| `insurance_claim_subclass` | Relevant claim categories | Insurance claim is selected/plausible |

During training, route subclass loss using the **true parent class**, so a weak early parent prediction does not prevent learning a subclass head. During inference, run the subclass head for the top parent candidate; optionally evaluate the second parent when parent-class margin is small and route uncertain cases to the LLM.

The multitask loss can be expressed as:

\[
\mathcal{L} =
\lambda_{\text{type}}\mathcal{L}_{\text{type}}
+
\lambda_{\text{sub}}\mathcal{L}_{\text{sub}}
+
\lambda_{\text{reg}}\mathcal{L}_{\text{reg}}
\]

where the subclass loss is masked for documents without a valid subclass label. Start with:

\[
\lambda_{\text{type}} = 1.0,\qquad
\lambda_{\text{sub}} = 1.0
\]

Then tune only if validation indicates that a parent head is overpowering the subclass objective.

### Baseline sequence

Build the system in evidence-producing stages:

1. **Title/filename-only baseline**  
   Logistic regression or lightweight linear classifier. This estimates how much label signal comes from metadata alone.

2. **Text-only baseline**  
   ModernBERT or a simpler TF-IDF baseline using normalized body text.

3. **Metadata + body baseline**  
   ModernBERT with title/filename prefix and first window.

4. **Hierarchical long-document ModernBERT**  
   Windowed classification plus document-level aggregation.

5. **Class-weighting and sampling**  
   Train real-data-only hierarchical model with imbalance controls.

6. **Synthetic augmentation**  
   Add only filtered synthetic examples to demonstrated weak subclasses.

This order is crucial. It lets you determine whether improved results actually come from better modeling or simply leakage, metadata memorization, synthetic duplication, or an easier split.

### Initial hyperparameters

Use these as a controlled starting configuration, not immutable doctrine:

| Parameter | Initial setting | Notes |
| --- | ---: | --- |
| Base checkpoint | `answerdotai/ModernBERT-base` | Shared encoder |
| Maximum sequence length | 8,192 tokens | Reduce only if GPU capacity requires it |
| Precision | bf16 preferred | fp16 fallback if required |
| Optimizer | AdamW fused where supported | Standard transformer optimizer |
| Learning rate | \(1 \times 10^{-5}\) to \(2 \times 10^{-5}\) | Begin at \(2 \times 10^{-5}\) |
| Weight decay | 0.01 | Standard starting point |
| LR schedule | Linear decay | With warmup |
| Warmup proportion | 0.05–0.06 | 5–6% of training steps |
| Epochs | 3–8 | Use early stopping |
| Early stopping | 2 evaluations without improvement | Monitor macro-F1 and calibration |
| Physical batch size | Hardware-dependent | Likely 1–4 at 8K context |
| Effective batch size | 16–32 | Use gradient accumulation |
| Gradient clip norm | 1.0 | Reduces instability |
| Checkpoint metric | Validation macro-F1 | Track parent and subclass separately |
| Random seeds | 42, 43, 44 | Use multiple seeds for final model |
| Training backend | PyTorch + Transformers | Standard `AutoModel` workflow |

ModernBERT can be fine-tuned with Hugging Face’s `AutoModelForSequenceClassification` and `Trainer` conventions, but the mailroom’s hierarchical structure likely requires a custom model wrapper or custom `Trainer.compute_loss`. Public ModernBERT fine-tuning examples commonly use `AutoTokenizer`, `AutoModelForSequenceClassification`, tokenized datasets, AdamW-style optimization, and a validation metric such as F1.[8][9]

### Imbalance handling before synthesis

Use authentic data first. The primary classes are materially imbalanced in the current plan—for example, merger agreements are much rarer than insurance claims and correspondence.[2]

Apply these controls before generating data:

- Inverse-frequency or effective-number class weights per head.
- Weighted random sampling at the document level.
- Balanced batch sampling for minority parent classes.
- Macro-F1, per-class recall, and per-class precision as core metrics.
- A separate model-selection metric for rare subclasses.
- `other` and LLM fallback policy for labels that lack enough real support.

For each head, estimate weights using a capped scheme rather than raw inverse frequency:

\[
w_c = \min\left(w_{\max}, \left(\frac{N}{K n_c}\right)^\alpha\right)
\]

where:

- \(N\) is total examples for the relevant head;
- \(K\) is the number of labels;
- \(n_c\) is the count for class \(c\);
- \(\alpha\) is typically between 0.5 and 1.0;
- \(w_{\max}\) limits rare-label instability.

This avoids allowing a one- or two-document class to dominate optimization.

### Window aggregation

At inference, predict each window independently and aggregate at the document level. Do not simply average every probability: generic boilerplate windows can dilute decisive title, heading, and clause evidence.

Recommended aggregation:

1. Predict calibrated primary-class probabilities for every window.
2. Compute a confidence-weighted plurality vote.
3. Select the primary label using:
   - support count;
   - mean calibrated confidence;
   - title/head-window evidence;
   - winner-versus-runner-up margin.
4. Run the matching subclass head on windows supporting the selected parent.
5. Aggregate subclass predictions similarly.
6. Record window agreement, alternate labels, margin, and all per-window outputs.

A useful routing score is:

\[
S =
p_{\mathrm{cal}}
\times
a_{\mathrm{window}}
\times
m_{\mathrm{margin}}
\]

where \(p_{\mathrm{cal}}\) is calibrated winning probability, \(a_{\mathrm{window}}\) is agreement across windows, and \(m_{\mathrm{margin}}\) reflects separation from the nearest alternate label.

The model should abstain to the LLM if the raw probability is high but windows disagree, if the title and body conflict, or if the subclass prediction is weak.

### Calibration

Confidence must not be treated as a raw softmax score. Modern neural networks are frequently overconfident; temperature scaling learns a single positive temperature on a held-out calibration partition to rescale logits without changing the predicted label or classification accuracy.[10][11]

For each head:

\[
p_i =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
\]

where \(z_i\) is an uncalibrated logit and \(T\) is fitted on the authentic calibration partition by minimizing negative log likelihood.

Fit separate temperatures for:

- Primary document-type head.
- Each parent-specific subclass head.

Track:

- Expected calibration error.
- Brier score.
- Reliability diagrams.
- Selective risk at every candidate routing threshold.
- Per-class calibration, especially for low-support labels.

Do not train or calibrate on synthetic samples. The calibration set must contain authentic, held-out documents only.

## Synthetic supplementation

### Core policy

Synthetic data is a **controlled augmentation source**, not a replacement for missing authentic examples. The literature supports generation plus filtering, but also emphasizes that LLM-generated samples are not guaranteed to be valid and should be subjected to quality assurance, filtering, and human review.[12][13][14]

The project should use synthetic data only to:

- Expand sparse but real subclasses.
- Add vocabulary and formatting variation.
- Generate targeted contrasts between confused sibling labels.
- Improve robustness to harmless layout and phrasing variation.
- Produce training material for underrepresented document structures.

It should not be used to:

- Create a production fast-path label with zero or near-zero authentic support.
- Balance every class mechanically.
- Alter validation, calibration, or test distributions.
- Substitute invented legal facts for real corpora.
- Copy or lightly paraphrase existing documents.
- Generate fake OCR noise without verifying it represents real intake conditions.

### Eligibility tiers

| Authentic examples per subclass | Synthetic policy | Production policy |
| ---: | --- | --- |
| 0–4 | Do not train a fast-path subclass model from synthetic-only data | Map to `other` or LLM fallback |
| 5–14 | Generate up to 3 accepted synthetic items per authentic item | Require high threshold and LLM fallback |
| 15–29 | Generate up to 2 accepted synthetic items per authentic item | Eligible after validation evidence |
| 30–74 | Up to 1 accepted synthetic item per authentic item, only for measured confusion | Usually eligible |
| 75+ | No routine augmentation | Use only targeted error reduction |

The actual cap is on **accepted**, not generated, items. If the quality filters reject 80% of a batch, do not relax filters to reach a quota.

### Label cards

Every synthetic target needs a version-controlled label card that restricts generation and makes review auditable.

```yaml
label: co_branding
parent_class: contract
canonical_token: co_branding

positive_cues:
  - Shared brand use or joint promotional activity
  - Approval process for marketing materials
  - Brand guidelines or trademark-use limitations
  - Collaborative campaign, product, or service presentation

negative_cues:
  - Standalone trademark license
  - Distribution-only relationship
  - Joint venture formation
  - Employment or consulting arrangement

expected_structure:
  - Title
  - Parties
  - Scope of collaboration
  - Brand-use rules
  - Approval process
  - Term and termination
  - Signatures

title_patterns:
  - Co-Branding Agreement
  - Brand Collaboration Agreement
  - Joint Marketing and Brand Use Agreement

target_length_chars:
  min: 3500
  max: 18000

nearest_confusions:
  - trademark_license
  - distribution
  - joint_venture
```

Create label cards from the taxonomy, authentic examples, and subject-matter review. They should never embed verbatim source documents or sensitive source entities.

### Generation approaches

Use multiple controlled forms of synthesis rather than one generic prompt.

| Method | Best use | Key control |
| --- | --- | --- |
| Schema-first drafting | Rare formal document subclasses | Generate a structure, then draft sections |
| Contrastive generation | Confused sibling labels | Generate minimal pairs differing in decisive cues |
| Style/format variation | Robustness to headings, layout, wording | Preserve label semantics |
| Counterfactual rewrite | Remove a misleading cue while retaining class | Human/LLM verification required |
| Retrieval-guided synthesis | Domain realism without copying | Use high-level label cards, not direct source excerpts |
| OCR/layout perturbation | Messy-document resilience | Only if based on observed error patterns |

For a distinction like `trademark_license` versus `co_branding`, generate contrastive pairs where shared marketing obligations and approval provisions exist in only the co-branding item, while a limited standalone trademark grant exists in only the trademark-license item. This teaches the classifier boundaries, not merely label-associated keywords.

### Mandatory quality pipeline

Every generated candidate must pass all stages:

1. **Metadata/schema check**
   - Correct parent class and subclass.
   - Required generator provenance.
   - Correct length range.
   - Complete title/body fields.
   - No invalid taxonomy token.

2. **Content-rule check**
   - Includes label-card positive cues.
   - Does not include forbidden neighboring-label cues.
   - Fits expected document structure.
   - Contains no prompt leakage or irrelevant explanation.

3. **Similarity and contamination check**
   - Exact deduplication by hash.
   - Near-duplicate detection with MinHash and embeddings.
   - N-gram overlap thresholds.
   - Entity overlap screening.
   - Reject material too close to source examples, especially any validation/test record.

4. **Independent label adjudication**
   - A different model, prompt, or reviewer classifies the candidate blindly.
   - Retain only candidates where independent adjudication agrees with the assigned label at a high standard.
   - Do not let the generator certify its own output.

5. **Quality and realism scoring**
   - Is the sample internally consistent?
   - Is it structurally plausible?
   - Does it look like a document rather than an instruction-following artifact?
   - Does it add diversity relative to retained examples?

6. **Human audit**
   - Review every rare-class accepted example initially.
   - Audit a random, stratified sample of larger batches.
   - Record rejection reason categories.

7. **Final acceptance ledger**
   - Immutable identifier.
   - Source label card version.
   - Generator model/version.
   - Prompt template hash.
   - Filter metrics.
   - Independent adjudication output.
   - Human reviewer decision when applicable.

Research on LLM text augmentation specifically recommends post-generation filtering, including duplicate/prompt-leakage checks, similarity-based filters, independent critics/classifiers, and human revision when quality is important.[12][13][14]

### Synthetic-data mixture

Keep authentic data dominant.

Recommended training constraints:

- At least 60–70% authentic examples in every epoch.
- At most 30–40% synthetic documents globally.
- At most 50% synthetic examples in a minority subclass’s sampled batch.
- Give synthetic rows a lower loss weight, e.g. 0.4–0.7 versus 1.0 for authentic rows.
- Keep synthetic records out of validation, calibration, test, and final deployment-gate analyses.
- Report results separately for authentic-only and authentic-plus-synthetic training.

The training loss can incorporate source weighting:

\[
\mathcal{L}_{\mathrm{weighted}} =
\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\cdot
s_i
\cdot
\ell_i
\]

where \(w_{y_i}\) is the class weight and \(s_i = 1.0\) for authentic examples, while \(s_i < 1.0\) for synthetic examples.

This makes synthetic material a diversity signal rather than a substitute ground truth.

## Validation and release gates

### Required metrics

Report parent and subclass performance separately:

| Metric | Why it matters |
| --- | --- |
| Primary-class accuracy | Basic routing quality |
| Primary-class macro-F1 | Protects minority classes |
| Subclass macro-F1 by parent | Avoids dominance by frequent subclasses |
| Per-class precision/recall | Reveals dangerous confusion patterns |
| Confusion matrix | Shows which labels need data or rules |
| ECE and Brier score | Validates confidence-based routing |
| Selective risk | Error rate among fast-pathed documents |
| Coverage | Share of documents allowed to bypass LLM |
| Window agreement | Indicates long-document consistency |
| OOD/abstention rate | Verifies deferral behavior |
| Authentic-only test score | Prevents synthetic benchmark inflation |

### Ablation matrix

The evidence for synthetic augmentation should come from controlled ablations:

| Run | Authentic data | Class weighting | Window aggregation | Synthetic data |
| --- | --- | --- | --- | --- |
| A | Yes | No | Basic | No |
| B | Yes | Yes | Basic | No |
| C | Yes | Yes | Full | No |
| D | Yes | Yes | Full | Filtered, targeted |
| E | Yes | Yes | Full | Synthetic removed for one label family |

Deploy synthetic augmentation only when Run D improves authentic-test macro-F1 or selective-risk performance over Run C, without producing unacceptable calibration degradation or new class-specific regressions.

### Deployment conditions

Enable classifier fast-path routing only if all conditions hold:

- Model artifact, tokenizer, labels, calibration parameters, and preprocessing versions match.
- Deterministic intake clerk says the document is not messy.
- Parent-class confidence is above the calibrated threshold.
- Required subclass confidence is above its separate threshold.
- Window agreement and margin meet minimums.
- Label has enough authentic support.
- No OOD or title/body contradiction signal appears.
- Failure to load or run the classifier defaults to the existing LLM/deterministic route.

Start in shadow mode: run ModernBERT on every document, log its route recommendation, but continue calling the LLM sorter. Compare the classifier result against downstream outcomes and manually audit disagreements. Then enable skipping only for clean, high-confidence, well-supported labels.

## Implementation sequence

1. Pin the Hugging Face dataset revision and create the normalized document-level training table.
2. Run quality, duplicate, title/filename leakage, and class-distribution audits.
3. Build grouped, stratified authentic train/validation/calibration partitions; lock the official test split.
4. Implement the shared ModernBERT hierarchical model and real-data-only baseline.
5. Add long-document windows and document-level aggregation.
6. Add class-weighted loss and balanced sampling.
7. Fit temperature calibration on authentic calibration data only.
8. Build label cards for sparse subclasses and generate candidates.
9. Filter, adjudicate, audit, and ledger accepted synthetic examples.
10. Run controlled real-only versus augmented ablations.
11. Package model, tokenizer, preprocessing, label maps, calibration, and artifact manifest together.
12. Integrate into `intake_node` in shadow mode, then conduct a controlled high-confidence rollout.

The key principle is simple: **real documents determine truth, synthetic documents supply carefully filtered variation, and calibration determines whether the model is trusted enough to save an LLM call.** This creates a legally and operationally defensible ModernBERT training program rather than merely a larger—but potentially noisier—dataset.

Sources
[1] answerdotai/ModernBERT-base - Hugging Face <https://huggingface.co/answerdotai/ModernBERT-base>
[2] pasted_text_1789685418.txt <https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12189704/14120326-e590-4e7d-84dc-4d722d0b28a7/pasted_text_1789685418.txt>
[3] pasted_text_1789685510.txt <https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12189704/c7c61dc6-2edc-43c1-88b5-c12f73ffa0a2/pasted_text_1789685510.txt>
[4] config.json · answerdotai/ModernBERT-base at main - Hugging Face <https://huggingface.co/answerdotai/ModernBERT-base/blob/main/config.json>
[5] [PDF] Processing Long Legal Documents with Pre-trained Transformers <https://aclanthology.org/2022.nllp-1.11.pdf>
[6] [PDF] Hierarchical Transformers for Long Document Classification <https://www.semanticscholar.org/paper/Hierarchical-Transformers-for-Long-Document-Pappagari-%C5%BBelasko/46b3ba0f3cb8340bc94f26e0fdf6dc4e38f68948>
[7] coastalcph/trldc: Transformer-based Long Document Classification <https://github.com/coastalcph/trldc>
[8] Fine-tune classifier with ModernBERT in 2025 - Philschmid <https://www.philschmid.de/fine-tune-modern-bert-in-2025>
[9] Fine-tune ModernBERT for text classification using synthetic data <https://huggingface.co/blog/davidberenstein1957/fine-tune-modernbert-on-synthetic-data>
[10] [1706.04599] On Calibration of Modern Neural Networks <https://arxiv.org/abs/1706.04599>
[11] Revisiting the Calibration of Modern Neural Networks <https://proceedings.neurips.cc/paper_files/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf>
[12] Text data augmentation for large language models: a ... <https://link.springer.com/article/10.1007/s10462-025-11405-5>
[13] Text Data Augmentation for Large Language Models: A ... <https://arxiv.org/html/2501.18845v1>
[14] Synthetic Data Generation Using Large Language Models ... <https://arxiv.org/html/2503.14023>
[15] Long Document Classification in the Transformer Era: A Survey on ... <https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70019>
[16] Class VI - Wells used for Geologic Sequestration of Carbon Dioxide <https://www.epa.gov/uic/class-vi-wells-used-geologic-sequestration-carbon-dioxide>
[17] Class: The Next Generation Virtual Classroom | Class <https://www.class.com/>
[18] [PDF] BERT-based Models for Arabic Long Document Classification <https://ceur-ws.org/Vol-3656/paper1.pdf>
[19] It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like ... <https://arxiv.org/html/2502.03793v1>
[20] ModernBERT for Sequence Classification - issues with finetuning <https://github.com/huggingface/transformers/issues/38720>
[21] [D] Finetuning ModernBERT is taking 3hrs (2 epochs) and 35gigs of ... <https://www.reddit.com/r/MachineLearning/comments/1is0q1a/d_finetuning_modernbert_is_taking_3hrs_2_epochs/>
[22] Efficient Methods for Updating a BERT Sequence Classification ... <https://stackoverflow.com/questions/78611808/efficient-methods-for-updating-a-bert-sequence-classification-model-with-new-cla>
[23] Refreshing zero-shot classification with ModernBERT - Medium <https://blog.knowledgator.com/refreshing-zero-shot-classification-with-modernbert-1a7ea9a4a776>
[24] (PDF) Hierarchical Transformers for Long Document Classification <https://www.academia.edu/79734742/Hierarchical_Transformers_for_Long_Document_Classification>
[25] Extending Temperature Scaling with Homogenizing Maps <http://jmlr.org/papers/volume26/24-0700/24-0700.pdf>
[26] On Calibration of Modern Neural Networks <https://www.semanticscholar.org/paper/On-Calibration-of-Modern-Neural-Networks-Guo-Pleiss/d65ce2b8300541414bfe51d03906fca72e93523c>
[27] Network Calibration by Class-based Temperature Scaling <https://www.eng.biu.ac.il/goldbej/files/2022/01/Lior_Frenkel_Eusipco_2021.pdf>
[28] Neural Network Calibration <https://geoffpleiss.com/blog/nn_calibration.html>
[29] Tropic-AI/moBERTo - Hugging Face <https://huggingface.co/Tropic-AI/moBERTo>
[30] artiwise-ai/modernbert-base-tr-uncased - Hugging Face <https://huggingface.co/artiwise-ai/modernbert-base-tr-uncased>
[31] On Calibration of Modern Neural Networks <https://fernandoperezc.github.io/Advanced-Topics-in-Machine-Learning-and-Data-Science/Fluri.pdf>
[32] Calibration Techniques in Deep Neural Networks - Heartbeat <https://heartbeat.comet.ml/calibration-techniques-in-deep-neural-networks-55ad76fea58b>
