# ED409 Syllabus Alignment and Presentation Preparation

## Purpose

This project should demonstrate the ED409 syllabus, not become a catalogue of
unrelated statistical methods. Every method used in the report or presentation
must satisfy one of two conditions:

1. It is explicitly part of the syllabus; or
2. It is a small supporting implementation or diagnostic whose purpose can be
   explained in plain language.

Methods in the second category should not become the central claim of the
assignment. This document records what must be understood before presenting.

## Direct syllabus alignment

| Project component | ED409 syllabus connection | Required understanding |
|---|---|---|
| Multiple OLS regression | Module I: classical multiple linear regression | OLS minimizes the sum of all squared residuals. Large residuals can have disproportionate influence. |
| Heteroskedasticity | Module I: models with heteroskedastic errors | Error variance is not constant. In this dataset, residual spread increases with fitted fare. |
| HC3 standard errors | Module I: White's heteroskedasticity-consistent standard error | HC3 belongs to the White heteroskedasticity-consistent family and additionally adjusts strongly for leverage. Coefficients do not change; estimated uncertainty changes. |
| VIF and collinearity | Module II: collinearity, consequences, assessment and strategies | VIF measures how well one predictor is linearly explained by the other predictors. Values near 1 indicate little linear collinearity. |
| Leverage, studentized residuals and Cook's distance | Module II: outlier and influential observations, diagnostics and strategies | Leverage concerns unusual predictors, a residual concerns unusual response behaviour, and Cook's distance concerns influence on the fitted model. |
| Least Trimmed Squares | Module II: robust regression, LTS properties and illustrations | LTS minimizes the sum of the smallest `h` squared residuals, preventing the largest residuals from directly determining the raw fit. |
| Taxi-fare business case | Module V: real-world applications using statistical software | The model is an applied regression case for fare explanation, stability assessment, and audit prioritization. |

The main assignment therefore stays primarily inside Modules I, II, and V.

## Supporting concepts used carefully

These concepts are reasonable extensions of the syllabus wording. They should
be defined whenever they appear.

| Supporting concept | Why it is used | How prominently to present it |
|---|---|---|
| Train-test split | Prevents model choices from being evaluated on the same observations used to make those choices. | Explain once in methodology. It is experimental discipline, not a new regression method. |
| MAE, RMSE and R-squared | Summarize prediction error and explained variation. | Use as model-comparison measures and know the difference between absolute and squared-error penalties. |
| Pearson and Spearman correlation | Compares ordinary linear association with rank-based monotonic association. | Descriptive evidence only; neither proves causation. |
| Breusch-Pagan test | Formal support for the syllabus topic of heteroskedasticity. Its null hypothesis is constant variance. | Keep as supporting evidence alongside the residual plot. Do not rely only on its p-value. |
| Ramsey RESET | Checks whether nonlinear functions of fitted values contain omitted mean structure. | Mention as a model-specification diagnostic and explain that rejection does not identify the correct missing term. |
| Jarque-Bera and Shapiro-Wilk | Check residual normality. | Keep in the technical diagnostics. The Q-Q plot and practical tail behaviour are easier to defend. |
| Durbin-Watson | Screens first-order residual autocorrelation, which is named in Module I. | Use cautiously because the stored rows are randomly ordered rather than a continuous time series. It is not strong evidence here. |
| FAST-LTS | Efficient algorithm for computing the syllabus LTS estimator. | Explain random starting subsets and C-steps; present it as computation, not a different statistical model. |
| Reweighted LTS | Refits least squares to observations compatible with the resistant raw LTS fit to improve efficiency. | Define raw residual scale, cutoff, and binary weights. Always report raw and reweighted results separately. |
| MAD residual scale | Robust alternative to residual standard deviation. | Explain that the median absolute deviation is less affected by extreme residuals. |
| Controlled response contamination | Demonstrates a robustness property by adding exactly $100 to selected training responses. | Clearly separate artificial corruption from genuine taxi records. |

## Concepts deliberately not added

The syllabus also covers ridge, LASSO, dimension reduction, generalized linear
models, logistic regression, kernel regression, lowess, nearest-neighbour
methods, local polynomials, and splines. They are not automatically needed in
this assignment.

- The response is continuous fare, so logistic or other binary-response GLMs
  would answer a different question.
- The predictor set is small and VIF is low, so ridge, LASSO, and dimension
  reduction would not solve a demonstrated problem.
- Nonparametric models could describe curvature, but using them as the primary
  method would distract from the required OLS-versus-LTS comparison.

If a nonlinear sensitivity model becomes necessary, it will be specified using
training evidence, identified clearly as secondary, and kept simpler than the
main comparison.

## Questions to prepare for

### Why use LTS when OLS already has a high R-squared?

R-squared measures in-sample fit, not resistance. OLS minimizes every squared
residual, so a small number of extreme observations can change its coefficients
even when overall R-squared is high. LTS studies whether the dominant
relationship is stable when the largest residuals do not determine the raw fit.

### What are `alpha` and `h`?

`alpha` is the retained fraction and `h` is the number of smallest squared
residuals used by raw LTS. At `alpha = 0.75`, approximately 75% of observations
contribute to the trimmed objective. Smaller alpha gives more resistance but
uses less information.

### What does FAST-LTS change?

It changes how LTS is computed, not the LTS objective. It uses many starting
subsets and concentration steps to find a low trimmed sum of squares without
checking every possible subset.

### Why reweight after raw LTS?

Raw LTS is resistant but can be statistically inefficient. Reweighting uses the
raw resistant fit to identify compatible observations and then refits least
squares to them. Raw LTS remains the robustness anchor.

### What is the difference between White and HC3 standard errors?

Both are heteroskedasticity-consistent estimators. The original White estimator
is commonly labelled HC0. HC3 uses an additional leverage adjustment and is
often more conservative when influential predictor patterns exist. Neither
changes the OLS coefficient estimates.

### Do significant diagnostic tests prove the model is useless?

No. With 40,000 observations, formal tests detect small departures. Decisions
must combine p-values with residual plots, effect sizes, and the purpose of the
model. The linear model can remain a useful interpretable approximation.

### Why not delete the four extreme-duration trips?

Their truth is unknown. Deleting them only because they strongly affect OLS
would prejudge the result. They remain in the data so LTS can assess them, and
any flags are called audit candidates rather than errors or fraud.

### Why must scaling use only training data?

Using test-set means or standard deviations would leak information from the
evaluation sample into model construction. Training-fitted scaling preserves an
honest out-of-sample comparison.

### Why add $100 only to training fares?

The stress test asks how estimation changes under contaminated training data.
The test set stays clean and fixed so prediction degradation is measured
against the same uncontaminated reference.

## Rule for the next phase

The clean OLS and LTS comparison will stay focused on syllabus concepts. Any
additional test or transformation must be documented with its purpose, null
hypothesis or objective, interpretation, and relationship to an ED409 module.
No method will be added merely because a software package provides it.
