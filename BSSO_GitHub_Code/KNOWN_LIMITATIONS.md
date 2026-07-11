# Known analytical limitations retained to match the manuscript

The repository corrects reproducibility and implementation errors without changing the manuscript's core analytical design. Several limitations therefore remain:

1. **Small development sample and single 7:3 split.** The internal test set is independent of model selection in this revision, but estimates remain sensitive to the patient split. A repeated nested-CV redesign would be methodologically stronger but would change the reported workflow and likely the numerical results.

2. **Top-50 recurrence across subset sizes.** The primary feature-selection rule ranks subsets of different sizes together. Larger subsets have a greater mechanical opportunity to include each predictor. This is retained because it is the prespecified manuscript strategy.

3. **Micro-average multiclass DeLong comparison.** The code applies paired DeLong to the flattened one-vs-rest micro-average representation. This is an explicit and reproducible implementation, but it should not be interpreted as a universally accepted general multiclass DeLong procedure.

4. **Hosmer-Lemeshow testing in a small internal test set.** The code reproduces class-specific tests but does not automatically label non-significant results as proof of good calibration. Brier scores and calibration curves should remain the primary calibration evidence.

5. **Side-level descriptive group comparisons.** Baseline ANOVA/Kruskal-Wallis and categorical tests treat mandibular sides as observations and do not model within-patient correlation. They are descriptive and should not be used as the main inferential basis for predictor selection.

6. **Exploratory causal analyses.** M1/M4 GEE and g-computation analyses are observational, may be unstable with the available sample size, and cannot establish causal effects. They remain explicitly exploratory.

7. **Conditional prediction rather than direct preoperative observation.** LLBCE and Depth of A represent achieved or candidate osteotomy geometry. The model estimates outcome probabilities conditional on anatomy and proposed parameter values; it does not deterministically control the fracture line.

8. **Deployment domain.** The web application is restricted to the observed training range of MRT and PMBT and class-specific empirical ranges of LLBCE and Depth of A. It does not support extrapolation beyond these domains.
