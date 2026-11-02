"""Create the training-only exploratory analysis stage report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT = PROJECT_ROOT / "report" / "Training_Data_Exploration_Report.docx"

NAVY = "17365D"
PALE_BLUE = "EAF1F8"
PALE_GRAY = "F4F6F8"
LIGHT_BORDER = "D9D9D9"
BLACK = RGBColor(0, 0, 0)


def set_cell_shading(cell, fill: str) -> None:
    """Apply a solid cell background."""
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_borders(cell, color: str = LIGHT_BORDER, size: str = "5") -> None:
    """Apply visible light borders to every side of a table cell."""
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(
    cell, top: int = 90, start: int = 100, bottom: int = 90, end: int = 100
) -> None:
    """Set table-cell padding in twips."""
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in [
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ]:
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    """Mark a table row to repeat at page breaks."""
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.append(header)


def set_run_font(run, name: str = "Arial", size: float | None = None) -> None:
    """Set both visible and OOXML font declarations."""
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)


def add_page_number(paragraph) -> None:
    """Insert an editable PAGE field."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])
    set_run_font(run, size=9)


def configure_document(document: Document) -> None:
    """Apply page, typography, and footer conventions."""
    for section in document.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.68)
        section.left_margin = Inches(0.78)
        section.right_margin = Inches(0.78)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.12

    title = styles["Title"]
    title.font.name = "Arial"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    title.font.size = Pt(25)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.space_after = Pt(16)

    for style_name, size, before, after in [
        ("Heading 1", 16, 10, 7),
        ("Heading 2", 12.5, 8, 5),
    ]:
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for section in document.sections:
        footer = section.footer
        footer.is_linked_to_previous = True
        paragraph = footer.paragraphs[0]
        add_page_number(paragraph)


def add_title_page(document: Document) -> None:
    """Create restrained academic front matter."""
    paragraph = document.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(95)
    paragraph.add_run(
        "Training Data Exploration and Linear Model Specification Assessment"
    )

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(30)
    run = subtitle.add_run(
        "Robust Detection of Anomalous Taxi Fares Using Least Trimmed Squares Regression"
    )
    set_run_font(run, size=14)
    run.bold = True

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.line_spacing = 1.5
    for text in [
        "Stage Report",
        "Aditya Kumar",
        "IED/10007/23",
        "3 November 2026",
    ]:
        run = meta.add_run(text + "\n")
        set_run_font(run, size=11.5)

    scope = document.add_paragraph()
    scope.alignment = WD_ALIGN_PARAGRAPH.CENTER
    scope.paragraph_format.space_before = Pt(48)
    run = scope.add_run(
        "Scope limited to the 40,000 training records. Test outcomes were not examined."
    )
    set_run_font(run, size=10.5)
    run.italic = True
    document.add_page_break()


def add_heading(document: Document, text: str, level: int = 1) -> None:
    """Add a black heading with no punctuation."""
    document.add_heading(text, level=level)


def add_body(document: Document, text: str, bold_lead: str | None = None) -> None:
    """Add a body paragraph with an optional bold opening phrase."""
    paragraph = document.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        run = paragraph.add_run(bold_lead)
        set_run_font(run)
        run.bold = True
        remainder = text[len(bold_lead) :]
        run = paragraph.add_run(remainder)
        set_run_font(run)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)


def add_bullets(document: Document, items: list[str]) -> None:
    """Add compact report bullets."""
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(4)
        run = paragraph.add_run(item)
        set_run_font(run)


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
    font_size: float = 9.2,
) -> None:
    """Add a polished numeric comparison table."""
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header_row = table.rows[0]
    set_repeat_table_header(header_row)
    for index, header_text in enumerate(headers):
        cell = header_row.cells[index]
        set_cell_shading(cell, NAVY)
        set_cell_borders(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(header_text)
        set_run_font(run, size=font_size)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        if widths:
            cell.width = Inches(widths[index])

    for row_index, values in enumerate(rows):
        row = table.add_row()
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            set_cell_borders(cell)
            set_cell_margins(cell)
            if row_index % 2:
                set_cell_shading(cell, PALE_BLUE)
            else:
                set_cell_shading(cell, "FFFFFF")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = paragraph.add_run(str(value))
            set_run_font(run, size=font_size)
            if widths:
                cell.width = Inches(widths[column_index])
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_figure(
    document: Document, image_path: Path, caption: str, width: float = 6.75
) -> None:
    """Insert a centered figure followed by its caption."""
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(8)
    run = caption_paragraph.add_run(caption)
    set_run_font(run, size=9)
    run.italic = True


def fmt(value: float, digits: int = 2) -> str:
    """Format a report number without false precision."""
    return f"{value:,.{digits}f}"


def create_report() -> Path:
    """Build the complete stage report from machine-readable outputs."""
    descriptive = pd.read_csv(TABLES / "training_descriptive_statistics.csv").set_index(
        "variable"
    )
    pearson = pd.read_csv(TABLES / "training_correlation_pearson.csv").set_index(
        "variable"
    )
    spearman = pd.read_csv(TABLES / "training_correlation_spearman.csv").set_index(
        "variable"
    )
    vif = pd.read_csv(TABLES / "training_vif.csv")
    coefficients = pd.read_csv(TABLES / "training_ols_coefficients.csv")
    diagnostics = pd.read_csv(TABLES / "training_ols_diagnostics.csv").set_index(
        "diagnostic"
    )["value"]
    influence = pd.read_csv(TABLES / "training_influence_summary.csv").iloc[0]
    residual_scale = pd.read_csv(
        TABLES / "training_residual_scale_by_fitted_decile.csv"
    )
    top_influence = pd.read_csv(TABLES / "training_top_influential_records.csv")
    summary = json.loads(
        (PROJECT_ROOT / "outputs" / "training_eda_summary.json").read_text(
            encoding="utf-8"
        )
    )

    document = Document()
    configure_document(document)
    add_title_page(document)

    add_heading(document, "Executive Summary")
    add_body(
        document,
        "The 40,000-record training sample supports a strong fare relationship, but it does not satisfy the conditions needed for unqualified classical OLS interpretation. Distance has the clearest association with fare. Extreme duration records create substantial leverage, residual variance rises with fitted fare, and residual means follow systematic curves across distance and duration. These findings justify the planned OLS and LTS comparison while requiring HC3 uncertainty for OLS and explicit influence reporting.",
    )
    add_bullets(
        document,
        [
            f"The diagnostic OLS model explains {100 * diagnostics['training_r_squared']:.1f}% of training-fare variation, with MAE ${diagnostics['training_mae']:.2f} and RMSE ${diagnostics['training_rmse']:.2f}. These are in-sample diagnostics, not test performance.",
            f"Fare and distance have Pearson correlation {pearson.loc['fare_amount', 'trip_distance']:.3f}. Fare and duration have Pearson correlation {pearson.loc['fare_amount', 'duration_min']:.3f} but Spearman correlation {spearman.loc['fare_amount', 'duration_min']:.3f}, showing strong tail distortion.",
            f"There are {int(influence['high_leverage_count']):,} observations above the 2p/n leverage threshold, {int(influence['cook_above_threshold_count']):,} above Cook's 4/n screening threshold, and {int(influence['absolute_studentized_residual_above_3_count']):,} with absolute internally studentized residual above 3.",
            "The primary additive linear specification should remain the common, interpretable basis for OLS and LTS. If LTS residual diagnostics retain the same curvature, nonlinear distance and duration terms should be added as a prespecified sensitivity model rather than selected from test results.",
        ],
    )

    add_heading(document, "Scope and Data Protection")
    add_body(
        document,
        f"The fixed modelling dataset contains 50,000 eligible January 2026 standard-rate Yellow Taxi trips. This report uses only the 40,000 observations labelled train. The training source-ID fingerprint is {summary['training_source_id_sha256']}. The 10,000 test outcomes were not summarized, plotted, transformed, or used to choose the model form.",
    )
    add_body(
        document,
        "The preparation rules require a January pickup, standard rate code 1, positive fare and distance, duration of at least one minute, and passenger count of at least one. No upper fare, distance, duration, or passenger-count limit was imposed. Visual windows stop at the 99.5th percentile only to keep dense plots readable; every tail observation remains in the calculations.",
    )

    add_heading(document, "Statistical Examination")
    add_body(
        document,
        "The analysis combines distribution summaries, Pearson and Spearman correlations, variance inflation factors, a training-only OLS diagnostic fit, HC3 standard errors, Breusch-Pagan and Ramsey RESET tests, residual-shape checks, leverage, Cook's distance, and binned residual means. Large-sample test statistics are interpreted together with graphical patterns and effect sizes.",
    )
    document.add_page_break()

    add_heading(document, "Training Distributions")
    add_figure(
        document,
        FIGURES / "training_distributions.png",
        "Figure 1  Central training distributions shown through the 99.5th percentile with full maxima reported",
    )
    rows = []
    for variable in ["fare_amount", "trip_distance", "duration_min", "passenger_count"]:
        row = descriptive.loc[variable]
        rows.append(
            [
                row["label"],
                fmt(row["median"]),
                fmt(row["p95"]),
                fmt(row["p99"]),
                fmt(row["maximum"]),
                fmt(row["skewness"]),
            ]
        )
    add_table(
        document,
        ["Variable", "Median", "P95", "P99", "Maximum", "Skewness"],
        rows,
        widths=[1.65, 0.75, 0.75, 0.75, 0.85, 0.85],
    )
    add_body(
        document,
        f"Fare, distance, and duration are strongly right-skewed. Duration is the most extreme: its median is {descriptive.loc['duration_min', 'median']:.2f} minutes and its 99th percentile is {descriptive.loc['duration_min', 'p99']:.2f}, but the maximum is {descriptive.loc['duration_min', 'maximum']:.2f} minutes. Such records can be valid, erroneous, or interrupted trips; the current policy keeps them so that LTS can assess their compatibility with the dominant fare relationship.",
    )
    document.add_page_break()

    add_heading(document, "Fare Relationships")
    add_figure(
        document,
        FIGURES / "training_fare_relationships.png",
        "Figure 2  Fare relationships in a 99.5th percentile visual window using log-density hexagons",
    )
    add_body(
        document,
        "Fare rises strongly with both distance and ordinary-duration trips. Distance shows a narrower relationship with fare, while the duration plot widens as trips become longer. The widening vertical spread signals heteroskedasticity and may also reflect congestion, route differences, surcharges not represented by the predictors, and unusual records.",
    )
    add_body(
        document,
        "The plots show several parallel fare bands rather than one perfectly homogeneous cloud. A single additive regression line can summarize the dominant relationship, but it cannot reproduce every tariff or operational detail. This is a model limitation rather than a reason to delete observations.",
    )
    document.add_page_break()

    add_heading(document, "Association and Collinearity")
    add_figure(
        document,
        FIGURES / "training_correlation_heatmap.png",
        "Figure 3  Pearson correlations among the response and proposed predictors",
        width=6.15,
    )
    corr_rows = [
        [
            "Fare and distance",
            fmt(pearson.loc["fare_amount", "trip_distance"], 3),
            fmt(spearman.loc["fare_amount", "trip_distance"], 3),
        ],
        [
            "Fare and duration",
            fmt(pearson.loc["fare_amount", "duration_min"], 3),
            fmt(spearman.loc["fare_amount", "duration_min"], 3),
        ],
        [
            "Distance and duration",
            fmt(pearson.loc["trip_distance", "duration_min"], 3),
            fmt(spearman.loc["trip_distance", "duration_min"], 3),
        ],
    ]
    add_table(
        document,
        ["Variable pair", "Pearson", "Spearman"],
        corr_rows,
        widths=[2.6, 1.1, 1.1],
    )
    add_body(
        document,
        "The rank correlations show that ordinary trips follow a strong monotonic distance-duration-fare structure. Pearson correlation is much lower for duration because a few exceptionally long recorded durations do not carry proportionally large fares. This discrepancy is direct evidence that tail observations can dominate covariance-based OLS quantities.",
    )
    vif_rows = [
        [row.term.replace("_", " ").title(), fmt(row.vif, 2)]
        for row in vif.itertuples()
        if row.term != "const"
    ]
    add_heading(document, "Variance Inflation Factors", level=2)
    add_table(document, ["Predictor", "VIF"], vif_rows, widths=[2.8, 1.0])
    add_body(
        document,
        "All predictor VIF values are between 1.01 and 1.21, so conventional linear collinearity is low. This result does not contradict the high Spearman association between distance and duration: VIF measures linear dependence and is itself affected by extreme tail geometry.",
    )

    add_heading(document, "Training Only OLS Diagnostic Model")
    add_body(
        document,
        f"The additive diagnostic model has training R squared {diagnostics['training_r_squared']:.4f}, adjusted R squared {diagnostics['training_adjusted_r_squared']:.4f}, MAE ${diagnostics['training_mae']:.2f}, RMSE ${diagnostics['training_rmse']:.2f}, and median absolute error ${diagnostics['training_median_absolute_error']:.2f}. These values describe fit to the training records and must not be presented as out-of-sample performance.",
    )
    coefficient_rows = []
    for row in coefficients.itertuples():
        label = row.term.replace("_", " ").title()
        coefficient_rows.append(
            [
                label,
                fmt(row.coefficient, 3),
                fmt(row.standard_error_conventional, 3),
                fmt(row.standard_error_hc3, 3),
                f"{row.hc3_ci_lower_95:.3f} to {row.hc3_ci_upper_95:.3f}",
            ]
        )
    add_table(
        document,
        ["Term", "Coefficient", "OLS SE", "HC3 SE", "HC3 95 percent CI"],
        coefficient_rows,
        widths=[1.55, 0.95, 0.8, 0.8, 1.55],
        font_size=8.8,
    )
    add_body(
        document,
        f"Holding the other predictors fixed, the diagnostic OLS slope is ${coefficients.loc[coefficients.term.eq('trip_distance'), 'coefficient'].iloc[0]:.2f} per mile and ${coefficients.loc[coefficients.term.eq('duration_min'), 'coefficient'].iloc[0]:.2f} per minute. These slopes are associations within the specified model. They are not causal effects, and the duration slope is especially vulnerable to the extreme-duration leverage documented below.",
    )
    add_body(
        document,
        "HC3 standard errors are materially larger than conventional OLS standard errors. For distance, HC3 is about 16 times the conventional value; for duration, it is about 39 times larger. This difference shows why conventional homoskedastic standard errors would give misleading precision.",
    )
    document.add_page_break()

    add_heading(document, "Residual Assumptions")
    add_figure(
        document,
        FIGURES / "training_residual_diagnostics.png",
        "Figure 4  Training-only OLS residual spread and normal quantile comparison",
    )
    diagnostic_rows = [
        [
            "Breusch-Pagan heteroskedasticity",
            fmt(diagnostics["breusch_pagan_lm"], 1),
            "Less than 1e-16",
        ],
        [
            "Ramsey RESET functional form",
            fmt(diagnostics["ramsey_reset_f"], 1),
            "Less than 1e-16",
        ],
        [
            "Jarque-Bera normality",
            fmt(diagnostics["jarque_bera"], 1),
            "Less than 1e-16",
        ],
        ["Durbin-Watson", fmt(diagnostics["durbin_watson"], 3), "Not a p value"],
    ]
    add_table(
        document,
        ["Diagnostic", "Statistic", "Reported probability"],
        diagnostic_rows,
        widths=[2.8, 1.25, 1.5],
        font_size=8.9,
    )
    add_body(
        document,
        f"Residuals have skewness {diagnostics['residual_skewness']:.2f} and kurtosis {diagnostics['residual_kurtosis']:.1f}. The Q Q plot confirms heavy tails and asymmetry. With 40,000 observations, formal tests are extremely sensitive, but the visual departures and large effect sizes are also substantial. Normal-theory OLS inference should therefore not be used without qualification.",
    )
    add_body(
        document,
        f"Durbin-Watson is {diagnostics['durbin_watson']:.3f}, close to 2 in the stored random sample order. This does not prove temporal independence because the ordering is not a continuous trip sequence and multiple trips may share operational conditions.",
    )
    document.add_page_break()

    add_heading(document, "Functional Form and Residual Scale")
    add_figure(
        document,
        FIGURES / "training_binned_residuals.png",
        "Figure 5  Mean OLS residuals within twenty predictor quantile bins with 95 percent mean confidence bands",
    )
    add_body(
        document,
        "Mean residuals should remain near zero across predictor values when the conditional mean is well represented. Instead, distance residuals move from negative to positive and back to negative, while duration residuals rise steadily from approximately -$2.48 to $4.64. Ramsey RESET reaches the same conclusion: the proposed linear mean is an approximation, not a complete fare function.",
    )
    selected_deciles = residual_scale.loc[
        residual_scale["fitted_decile"].isin([1, 5, 10])
    ]
    scale_rows = [
        [
            str(int(row.fitted_decile)),
            fmt(row.fitted_mean),
            fmt(row.mean_absolute_residual),
            fmt(row.residual_rmse),
        ]
        for row in selected_deciles.itertuples()
    ]
    add_table(
        document,
        [
            "Fitted fare decile",
            "Mean fitted fare",
            "Mean absolute residual",
            "Residual RMSE",
        ],
        scale_rows,
        widths=[1.35, 1.35, 1.55, 1.2],
        font_size=8.9,
    )
    add_body(
        document,
        f"Residual RMSE rises from ${residual_scale.iloc[0]['residual_rmse']:.2f} in the lowest fitted-fare decile to ${residual_scale.iloc[-1]['residual_rmse']:.2f} in the highest. This more than threefold increase is a practical measure of heteroskedasticity, not merely a significant p value.",
    )
    document.add_page_break()

    add_heading(document, "Leverage and Influence")
    influence_rows = [
        [
            "High leverage above 2p/n",
            f"{int(influence['high_leverage_count']):,}",
            f"{100 * influence['high_leverage_fraction']:.2f}%",
        ],
        [
            "Cook's distance above 4/n",
            f"{int(influence['cook_above_threshold_count']):,}",
            f"{100 * influence['cook_above_threshold_fraction']:.2f}%",
        ],
        [
            "Absolute studentized residual above 3",
            f"{int(influence['absolute_studentized_residual_above_3_count']):,}",
            f"{100 * influence['absolute_studentized_residual_above_3_fraction']:.2f}%",
        ],
    ]
    add_table(
        document,
        ["Screening condition", "Training records", "Fraction"],
        influence_rows,
        widths=[3.2, 1.2, 1.0],
    )
    add_body(
        document,
        f"The maximum leverage is {influence['maximum_leverage']:.3f}, compared with the 2p/n screening threshold {influence['leverage_threshold_2p_over_n']:.5f}. Maximum Cook's distance is {influence['maximum_cooks_distance']:.2f}. These are screening statistics, not automatic deletion rules.",
    )
    top_rows = []
    for row in top_influence.head(6).itertuples():
        top_rows.append(
            [
                str(int(row.source_row_id)),
                fmt(row.fare_amount),
                fmt(row.trip_distance),
                fmt(row.duration_min),
                fmt(row.residual),
                fmt(row.cooks_distance, 3),
            ]
        )
    add_heading(document, "Highest Cook Distance Records", level=2)
    add_table(
        document,
        ["Source row", "Fare", "Miles", "Minutes", "Residual", "Cook distance"],
        top_rows,
        widths=[1.1, 0.7, 0.7, 0.85, 0.85, 1.0],
        font_size=8.6,
    )
    add_body(
        document,
        "The four most influential records report durations near 1,412 to 1,437 minutes with fares from $7.90 to $24.00. OLS predicts fares above $126 for these records, producing residuals near -$114 to -$120 and Cook's distances from 47.77 to 56.62. Their influence explains why the OLS duration coefficient and its uncertainty are unstable. They remain in the dataset for robust modelling and later audit review.",
    )
    document.add_page_break()

    add_heading(document, "Model Specification Decision")
    add_body(
        document,
        "The primary comparison will retain the same additive linear predictors for OLS and LTS: distance, duration, passenger count, pickup-hour sine and cosine, and weekend status. This common specification isolates the effect of the estimator. It also preserves direct coefficient interpretation and matches the assignment's focus on resistance to anomalous fares.",
    )
    add_bullets(
        document,
        [
            "Fit all scaling parameters on the 40,000 training records and apply them unchanged to the test set.",
            "Use HC3 uncertainty for OLS and report conventional standard errors only as a diagnostic comparison.",
            "Fit raw and explicitly reweighted FAST-LTS at alpha 0.75, with alpha 0.50 and 0.90 sensitivity checks.",
            "Retain all extreme predictor and fare observations; classify LTS residual flags as audit candidates rather than proven errors or fraud.",
            "After the robust fit, repeat the binned residual check. If meaningful curvature remains under LTS, add a training-prespecified nonlinear sensitivity model using distance and duration terms without consulting test outcomes.",
        ],
    )
    add_heading(document, "Limitations")
    add_body(
        document,
        "This phase does not estimate out-of-sample performance and does not establish causal fare effects. TLC records are provider-submitted, passenger counts are driver-reported, and the predictors omit location, congestion-fee detail, tolls, and route information. Formal test p values are affected by the large sample, and the diagnostic OLS fit is itself sensitive to the influential observations it is intended to reveal.",
    )
    add_heading(document, "Next Statistical Phase")
    add_body(
        document,
        "The next phase will freeze training-only scaling, fit clean OLS and reweighted FAST-LTS, perform alpha sensitivity, and evaluate both models once on the untouched 10,000-record test set. Controlled $100 response contamination will follow only after clean-model results are recorded.",
    )

    add_heading(document, "Data and Reproducibility")
    add_body(
        document,
        "Data source: New York City Taxi and Limousine Commission, January 2026 Yellow Taxi Trip Records. Preparation, split assignment, diagnostics, figures, and tables are generated by repository commands with seed 42. The machine-readable tables in outputs/tables are the source of the values reported here.",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(create_report())
