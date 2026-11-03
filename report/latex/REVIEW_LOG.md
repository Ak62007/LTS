# LaTeX rewrite and review record

## Pass 1: narrative and initial typesetting

- Rewrote the opening around the audit team's decision, then explained one
  dataset row, the metered-fare response, predictors, source, and sample.
- Replaced disconnected technical notes with a sequence from the business
  questions through evidence and interpretation to the recommendations.
- Added saved distribution, fare-relationship, residual, binned-residual,
  clean-performance, stress-performance, and coefficient-change figures.
- Replaced the crowded actual-point illustration with two separate views:
  an artificial-outlier overview and matching-axis before/after fitted lines.
- Compiled and rendered the first draft. Found the last lines of the business
  questions spilling onto page 3, figures separated from relevant prose, and
  long checksum text extending beyond the margin.

## Pass 2: pagination and technical explanation

- Shortened the final business-case paragraph, restoring the strict two-page
  opening and starting the solution on page 3.
- Kept figures at their interpretation points, enabled natural page-bottom
  spacing, and made the fingerprints wrap safely.
- Restored the full OLS coefficient/standard-error table, distinguishing HC3
  uncertainty from a change in coefficients.
- Corrected the overfull text and visually inspected all 19 pages.
- Checked the opposing MAE/RMSE rankings, nested audit thresholds, and the
  interpretation of artificial-label precision and false positives.

## Pass 3: final editing and reproducibility

- Added the installed RobPy version and exact comparable-display counts.
- Moved the reference-profile explanation before the fitted-line figure to
  avoid an awkward paragraph split. Started the conclusion on its own page.
- Shared the revised plotting function between the analysis command and report
  builder; updated notebook 05 to show the same two views.
- Added a repeatable LaTeX build command and a self-contained source ZIP.
- Kept the fitted-line interpretation together, grouped the references onto
  one page, and replaced the reference-list environment to remove its layout
  warning. Used hanging reference indentation and readable URL wrapping.

## Initial LaTeX edition verification (3 November 2026)

- Final PDF: 19 pages; case, dataset introduction, and four questions occupy
  pages 1-2; the solution starts on page 3.
- Nine analytical figures, ten result tables, and seven genuinely executed
  code images appear in the report. Each code image precedes its result.
- All pages were rendered and visually reviewed. Final unchanged pages were
  also checked by pixel comparison against the reviewed render; revised
  pages were inspected again. No clipping, overlapping content, broken
  equation layout, or unresolved cross-references remained.
- The final compiler log contains no warnings, overfull boxes, or underfull
  boxes. The source ZIP was extracted into a separate temporary directory
  and successfully compiled without access to repository figure paths.
- Primary clean and stressed MAE, median absolute error, RMSE, R-squared,
  and coefficient values were checked against the source CSV tables.
- Notebook 05 was re-executed: all five code cells completed without errors,
  including the new plots and display-design output. Existing notebook 06
  execution evidence remains present and error-free.
- Automated checks: `uv run pytest` reports 17 passed; `uv run ruff check .`
  passes. The report build uses saved estimates and does not refit models.
- Raw data, modelling split, contamination labels, and statistical estimates
  were not changed by this report revision.

## Requested editorial revision (3 November 2026)

- Removed the download-date sentence, passenger/driver contrast, and rate-code
  definition from page 1 as requested.
- Removed both computing-benchmark paragraphs from the sampling section and
  the later sentence referring back to that discussion.
- Removed the entire closing "Reproducibility and preparation for the
  presentation" section. Provenance remains recorded in repository metadata.
- Added "Direct answers to the business questions" immediately before the
  references, with a separate answer for each of the four opening questions.
  The answers retain the MAE/RMSE trade-off and the distinction between audit
  candidates and verified errors or fraud.
- The revised PDF has 20 pages: case/questions on pages 1-2, solution starting
  on page 3, direct answers on page 19, and references on page 20.
- Checked the new answer-section values against the saved model, influence,
  audit, and contamination results. No models or estimates were changed.
- Rendered all 20 pages. Fifteen unchanged pages match the previously reviewed
  render exactly; inspected each revised/new page and confirmed legible layout.
  The final compiler log has no warnings or overfull/underfull boxes.
- Refreshed the portable source ZIP and its existing extracted copy after
  verifying that the extracted copy contained no user changes.

## Section order and repository link revision (3 November 2026)

- Renamed the answers section to "Answers we found for the business questions"
  and moved it before "What the audit team should conclude". Automatic section
  numbering now makes the answers section 8 and the conclusion section 9.
- Added the GitHub repository as an inline hyperlink in the sampling discussion,
  where the preparation steps and executed notebooks are introduced.
- Rebuilt and rendered all 20 pages. The answers occupy page 18, followed by
  the conclusion on page 19. Inspected the three changed pages; all other
  pages match the previously reviewed render exactly. Verified the clickable
  repository URL on page 4 and a compiler log without layout warnings.
