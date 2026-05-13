# Paper Improvement Log

## 2026-05-13 Night Pass

Scope: improve the course-project workshop draft without launching more experiments.

Completed:

- Reframed the draft from an ATLAS-SOTA story to a constraint-transfer story for offline-to-online RL.
- Added checked related work and BibTeX entries for D4RL, CQL, Cal-QL, TD3, SAC, PPO, TD3+BC, ReBRAC, PRDC, A2PR, SSAR, IQL, AWR, Adaptive BC, PROTO, ENOTO, and SUF.
- Made the ATLAS method boundary explicit: post-cache trusted-action distillation, not a from-scratch cheaper SSAR replacement.
- Added appendix commands for the C-line smoke, label export, selector training, offline ATLAS, and minimal O2O runner.
- Recompiled the LaTeX draft with `latexmk`; final build has no undefined citations/references and no overfull hbox warnings.
- Removed TODO boxes from the abstract so the first page reads like a paper rather than an internal checklist.

## 2026-05-13 Image2 Figure Pass

Scope: replace the rough ATLAS mechanism diagram with a native image-generation paper illustration.

Completed:

- Added a project-local `paper_illustration_image2.py` helper for preflight, finalize, and verify receipts. It does not generate images; rendering is still done by the native `codex-image2` bridge.
- Ran preflight successfully with `ok=true`.
- Generated `figures/ai_generated/figure_v1.png` via `codex-image2`; native image generation was confirmed by the bridge.
- Accepted the first generated figure at score 9/10 after checking component coverage, arrow direction, label readability, and paper readiness.
- Finalized canonical artifacts under `figures/ai_generated/`.
- Copied the accepted figure into `paper/figures/fig2_trusted_constraint_transfer_image2.png`.
- Updated Section 3 to use the new image2 figure and recompiled the paper.
- Saved `main_round2_image2.pdf` as the current image-integrated PDF snapshot.

Still open:

- A-line should fill the value-conservatism row with comparable CQL/Cal-QL results.
- B-line should fill the non-conservative online contrast row with comparable PPO/SAC/vanilla TD3-style results.
- Final abstract and conclusion should be shortened after A/B results arrive.

## 2026-05-14 O2O Constraint-Transfer Pass

Scope: integrate the required P0 offline-to-online eval20 panel, external review feedback, and targeted Q-filtered trust diagnostics.

Completed:

- Added the P0 eval20 O2O panel: TD3+BC, ATLAS, random matched trust subset, and SSAR/IQL-qv teacher labels under 50k offline + 10k online on `hopper-medium-replay-v2`.
- Ran and saved Gemini external review. The review scored the draft 6/10 and recommended reframing around the constraint-transfer gap rather than ATLAS-as-SOTA.
- Implemented a minimal online Q-filtered trust gate and added seed0/seed1 diagnostic results.
- Updated the title, abstract, introduction, method, experiment section, discussion, conclusion, appendix, and figures to match the current evidence boundary.
- Recompiled `main.pdf` with `latexmk`; final build has no undefined citations/references and no overfull hbox warnings.
- Visually checked representative PDF pages after compilation.

Current conclusion:

- C-line is now sufficient as a mechanism study draft: trusted-action labels help offline initialization, but fixed teacher regularization transfers poorly to online fine-tuning.
- Q-filtered trust is a promising hypothesis on seed0, but seed1 shows high Hopper O2O variance and prevents a stable superiority claim.
- The next paper-level blocker is not another C-line baseline sweep; it is comparable A-line and B-line rows if the final report keeps the three-track framing.
