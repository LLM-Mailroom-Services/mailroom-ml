# PROVENANCE — GEPA mechanics in this agent file

What this pins: the GEPA framework facts (loop order, acceptance criteria,
selection strategies, frontier types, component selection, merge
preconditions, budget/stop/cache knobs) encoded in
`.opencode/agents/prompt-engineer.md` were extracted from the **upstream
GEPA repository source**, not paraphrased from the paper or blog posts.

- Upstream repo: https://github.com/gepa-ai/gepa (Apache-2.0, Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors)
- Pinned ref: `b265bf9ca77fd8e8d82039d9f74911b8780fe1ce` (default branch, 2026-08-19, "Reduce redundant reflective trace context (#383)")
- Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv 2507.19457
- Adapted (not vendored): this repo ships NO GEPA code; the agent file translates the engine's mechanics into the manual, board-governed iteration protocol. Source files consulted:
  - `src/gepa/core/engine.py` — iteration orchestration (accept+select authority)
  - `src/gepa/core/state.py` — `FrontierType` = instance/objective/hybrid/cartesian; Pareto frontier bookkeeping (`get_pareto_front_mapping`); evaluation cache
  - `src/gepa/strategies/acceptance.py` — `StrictImprovementAcceptance` (default), `ImprovementOrEqualAcceptance`
  - `src/gepa/strategies/candidate_selector.py` — `ParetoCandidateSelector` (default) / `CurrentBestCandidateSelector` / `EpsilonGreedyCandidateSelector` / `TopKParetoCandidateSelector`
  - `src/gepa/strategies/component_selector.py` — `RoundRobinReflectionComponentSelector` (default) / `AllReflectionComponentSelector`
  - `src/gepa/strategies/batch_sampler.py` — `EpochShuffledBatchSampler` (default)
  - `src/gepa/proposer/reflective_mutation/reflective_mutation.py` — proposer flow (before/after subsample evaluation, ASI construction)
  - `src/gepa/proposer/merge.py` — system-aware merge: common-ancestry check, validation-support disjointness (`merge_val_overlap_floor`), accept iff score >= max(parents)
  - `src/gepa/api.py` — public defaults (`candidate_selection="pareto"`, `frontier_type="instance"`, `skip_perfect_score=True`, `module_selector="round_robin"`, `use_merge=False`, `max_merge_invocations=5`, `merge_val_overlap_floor=5`)
- Vendored/adapted date: 2026-08-21 (hermes, KANBAN-066 / issue #31)

Re-sync (diff our claims against a fresh clone):

```bash
rm -rf /tmp/gepa && git clone --depth 1 https://github.com/gepa-ai/gepa.git /tmp/gepa
cd /tmp/gepa && git rev-parse HEAD   # compare to the pin above
grep -n "class .*Acceptance" src/gepa/strategies/acceptance.py
grep -n "class .*CandidateSelector" src/gepa/strategies/candidate_selector.py
grep -nE "FrontierType|frontier_type" src/gepa/core/state.py | head
grep -nE "candidate_selection|frontier_type|skip_perfect_score|module_selector|use_merge|max_merge_invocations|merge_val_overlap_floor" src/gepa/api.py
```

If upstream renames any class/default referenced in the agent file, update
both the agent file and `tests/test_prompt_engineer_gepa.py` in the same
commit, bump this pin, and note the delta in a dated discussion entry.
License note: GEPA is Apache-2.0; we reference names/behavior only — no
source copied — so no license header is required in the agent file itself.
