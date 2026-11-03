# Final report: LaTeX edition

`ED409_LTS_Final_Report.tex` is the canonical editable report. The compiled
submission is `report/ED409_LTS_Final_Report.pdf` in the repository. The earlier
DOCX is retained as a previous edition; edits to it do not update the LaTeX PDF.

The first two PDF pages contain the business scenario, dataset introduction,
and four questions. The solution starts on page 3. The remainder explains the
data preparation, estimator, diagnostics, actual-data results, audit queue,
controlled contamination experiment, and limitations.
The section "Direct answers to the business questions" explicitly answers
each opening question using the results, immediately before the references.

## Rebuild within the repository

From the repository root, with the uv environment and Tectonic installed:

```bash
uv run python -m report.build_latex_report
```

This command regenerates the report tables from saved CSV outputs and draws
the two actual-trip illustrations from the fixed split and saved coefficients.
It then compiles the PDF and creates `report/ED409_LTS_LaTeX_Source.zip`.
It does not refit OLS or FAST-LTS. After changing statistical results, rerun the
analysis and executed evidence notebook before building the report.

To regenerate only tables and figures:

```bash
uv run python -m report.build_latex_assets
```

Numeric table fragments live in `generated/`. Figure captions and all prose
live in the main `.tex` file. The original seven code images come from the
genuinely executed `06_final_report_execution_evidence.ipynb`; the images are
placed before their corresponding results.

## Compile the standalone source archive

Extract the ZIP into an empty directory and run:

```bash
tectonic ED409_LTS_Final_Report.tex
```

The ZIP contains every figure, code image, and table fragment needed to
compile. No raw taxi data or Python environment is required for compilation.
The same contents can be uploaded to Overleaf; choose the main `.tex` file
and pdfLaTeX. Tectonic was the compiler used for the verified local PDF.

The archive is a typesetting source package, not a replacement for the full
analysis repository. No illustrative points were invented, and the conditional
lines use the saved full-training-sample coefficients.

## Figure interpretation

Figure 8 shows the artificial outlier band on the full fare scale. Figure 9
uses identical smaller fare scales before and after corruption so that the
ordinary trip cloud and each fitted line are readable. The high red points
are intentionally outside this second view and remain visible in Figure 8.
The comparison uses nearby trip profiles for display, while the models use
all 40,000 training rows. The display metadata are in `generated/plot_metadata.json`.

After editing, render the PDF and inspect every page. Keep the case/questions
within two pages and check all figures and numerical claims against the CSVs.
