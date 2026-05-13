# LaTeX Workshop Draft

This directory contains the current workshop-style LaTeX draft for the RL course project.
It uses the `neurips_2026` style file in `dblblindworkshop` mode.

Use this when A/B-line teammates send results:

1. Open `main.tex`.
2. Fill TODO boxes in `sections/`.
3. Keep all runs in the same format: method, family, env, seed, offline steps, online steps, eval episodes, final score, best score, curve/log path.
4. Keep citations tied to checked primary sources or paper pages.

Current status:

- C-line results, figures, references, and appendix reproduction commands are prefilled.
- P0 offline-to-online eval20 results, Gemini external-review feedback, Q-filtered trust diagnostics, and seed1 controls are integrated into the draft.
- A-line and B-line rows are TODO placeholders.
- The paper compiles under the NeurIPS 2026 workshop template. `neurips_2026.tex` is a compatibility wrapper for Overleaf projects that still use the original template filename as the main document.
- Replace `NeurIPS Workshop Draft` in `main.tex` once the exact workshop target is known.
- Remaining TODOs are intentional merge points for teammate results, not LaTeX blockers.
- Current claim boundary: the draft supports a C-line constraint-transfer-gap mechanism study, not a robust ATLAS superiority claim.

Suggested compile command:

```bash
cd /Users/robin/Desktop/taoyao/RL/project/paper/latex_template_20260513
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected output: `main.pdf`.
