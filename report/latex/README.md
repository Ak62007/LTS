# LaTeX report

The editable source is `ED409_LTS_Report.tex`. The final PDF is
`report/ED409_LTS_Report.pdf`, and the standalone source package is
`report/ED409_LTS_LaTeX_Source.zip`.

The case and questions occupy pages 1–2. The solution begins on page 3.
"Answers we found for the business questions" precedes the conclusion.

## Build from the repository

From the repository root, with uv and Tectonic installed:

```bash
uv run python -m report.build_latex_report
```

The build regenerates tables from the saved CSVs, redraws the actual-trip
illustrations from saved coefficients, compiles the PDF, and packages its source.
It does not refit models.

To refresh only tables and figures:

```bash
uv run python -m report.build_latex_assets
```

To rerun the seven report excerpts and regenerate their code images:

```bash
uv run python -m report.build_execution_evidence
```

The evidence is stored in `notebooks/06_final_report_execution_evidence.ipynb`.
Every code image in the report precedes its corresponding result.

## Standalone compilation

Extract the source ZIP into an empty directory, then run:

```bash
tectonic ED409_LTS_Report.tex
```

The archive includes all required figures, code images, and numeric table
fragments. Raw taxi data and Python are not needed for compilation. On
Overleaf, select `ED409_LTS_Report.tex` as the main file and use pdfLaTeX;
the verified local build uses Tectonic.

Figure 8 shows the artificially shifted fares on the full scale. Figure 9
uses matching smaller scales to compare ordinary observations and fitted
lines. Plot restrictions affect the display only; the fits use all 40,000
training rows. Display settings are saved in `generated/plot_metadata.json`.

After editing, render and inspect the PDF, check reported numbers against
the CSVs, and preserve the two-page limit for the opening case and questions.
