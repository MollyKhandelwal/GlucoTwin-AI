
from pathlib import Path
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Flowable,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "GlucoTwin_AI_Architecture.pdf"


# ---------------------------------------------------------
# THEME
# ---------------------------------------------------------
NAVY = colors.HexColor("#102A43")
TEAL = colors.HexColor("#087E8B")
CYAN = colors.HexColor("#D9F3F4")
LIGHT = colors.HexColor("#F3F7FA")
DARK = colors.HexColor("#243B53")
GREY = colors.HexColor("#627D98")
WHITE = colors.white
GREEN = colors.HexColor("#DFF5E8")
AMBER = colors.HexColor("#FFF2CC")
BORDER = colors.HexColor("#D9E2EC")


# ---------------------------------------------------------
# DOCUMENT STYLES
# ---------------------------------------------------------
styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=29,
        leading=35,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=12,
    )
)

styles.add(
    ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=20,
        alignment=TA_CENTER,
        textColor=GREY,
        spaceAfter=12,
    )
)

styles.add(
    ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=22,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=9,
        keepWithNext=True,
    )
)

styles.add(
    ParagraphStyle(
        name="SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=TEAL,
        spaceBefore=7,
        spaceAfter=5,
        keepWithNext=True,
    )
)

styles.add(
    ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=DARK,
        spaceAfter=7,
    )
)

styles.add(
    ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=GREY,
        spaceAfter=4,
    )
)

styles.add(
    ParagraphStyle(
        name="CellCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=DARK,
    )
)

styles.add(
    ParagraphStyle(
        name="CellHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=WHITE,
    )
)

styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=14,
        textColor=NAVY,
    )
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def P(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def make_table(headers, rows, widths):
    data = [[P(str(h), "CellHeader") for h in headers]]

    for row in rows:
        data.append([
            item if isinstance(item, Paragraph)
            else P(str(item), "CellCustom")
            for item in row
        ])

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
            ]
        )
    )

    return table


def callout(text, background=CYAN, width=170 * mm):
    table = Table(
        [[P(text, "Callout")]],
        colWidths=[width],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


# ---------------------------------------------------------
# ARCHITECTURE DIAGRAM FLOWABLE
# ---------------------------------------------------------
class ArchitectureDiagram(Flowable):
    def __init__(self, diagram_type="main", width=170 * mm, height=115 * mm):
        super().__init__()
        self.diagram_type = diagram_type
        self.width = width
        self.height = height

    def draw_box(self, c, x, y, w, h, title, subtitle="", fill=CYAN):
        c.setFillColor(fill)
        c.setStrokeColor(BORDER)
        c.roundRect(x, y, w, h, 7, fill=1, stroke=1)

        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(x + w / 2, y + h - 15, title)

        if subtitle:
            c.setFillColor(DARK)
            c.setFont("Helvetica", 6.7)

            lines = subtitle.split("\n")
            line_y = y + h - 28

            for line in lines:
                c.drawCentredString(x + w / 2, line_y, line)
                line_y -= 9

    def arrow(self, c, x1, y1, x2, y2):
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.4)
        c.line(x1, y1, x2, y2)

        import math

        angle = math.atan2(y2 - y1, x2 - x1)
        size = 5

        left = (
            x2 - size * math.cos(angle - math.pi / 6),
            y2 - size * math.sin(angle - math.pi / 6),
        )
        right = (
            x2 - size * math.cos(angle + math.pi / 6),
            y2 - size * math.sin(angle + math.pi / 6),
        )

        path = c.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(*left)
        path.lineTo(*right)
        path.close()

        c.drawPath(path, fill=1, stroke=0)

    def draw(self):
        c = self.canv
        w = self.width
        h = self.height

        if self.diagram_type == "main":
            # Two separate data sources
            box_w = 47 * mm
            box_h = 19 * mm
            left_x = 4 * mm
            right_x = w - box_w - 4 * mm
            top_y = h - box_h - 4 * mm

            self.draw_box(
                c, left_x, top_y, box_w, box_h,
                "Synthetic CGM",
                "Generated glucose readings",
            )
            self.draw_box(
                c, right_x, top_y, box_w, box_h,
                "Synthetic EHR",
                "Generated patient attributes",
            )

            # Integration
            integration_y = h - 43 * mm
            integration_x = (w - 58 * mm) / 2

            self.draw_box(
                c, integration_x, integration_y, 58 * mm, 20 * mm,
                "Synthetic Data Integration",
                "Patient features + glucose history",
                fill=GREEN,
            )

            self.arrow(
                c,
                left_x + box_w / 2,
                top_y,
                integration_x + 10 * mm,
                integration_y + 20 * mm,
            )
            self.arrow(
                c,
                right_x + box_w / 2,
                top_y,
                integration_x + 48 * mm,
                integration_y + 20 * mm,
            )

            # Shared synthetic model pipeline
            stages = [
                ("Feature Engineering", "Recent glucose features"),
                ("Forecast Target", "Approx. +60 minutes"),
                ("Model Evaluation", "Chronological holdout"),
                ("Metrics & Outputs", "MAE, RMSE, predictions"),
            ]

            stage_w = 38 * mm
            gap = (w - 4 * stage_w) / 3
            stage_y = h - 80 * mm

            for i, (title, subtitle) in enumerate(stages):
                x = i * (stage_w + gap)
                self.draw_box(
                    c, x, stage_y, stage_w, 22 * mm,
                    title, subtitle,
                    fill=AMBER if i == 2 else LIGHT,
                )

                if i == 0:
                    self.arrow(
                        c,
                        integration_x + 29 * mm,
                        integration_y,
                        x + stage_w / 2,
                        stage_y + 22 * mm,
                    )
                else:
                    prev_x = (i - 1) * (stage_w + gap)
                    self.arrow(
                        c,
                        prev_x + stage_w,
                        stage_y + 11 * mm,
                        x,
                        stage_y + 11 * mm,
                    )

            # Dashboard
            dash_w = 58 * mm
            dash_x = (w - dash_w) / 2
            dash_y = 3 * mm

            self.draw_box(
                c, dash_x, dash_y, dash_w, 19 * mm,
                "Streamlit Dashboard",
                "Overview, history and evaluation",
                fill=CYAN,
            )

            self.arrow(
                c,
                2 * stage_w + 2 * gap + stage_w / 2,
                stage_y,
                dash_x + dash_w / 2,
                dash_y + 19 * mm,
            )

            c.setFillColor(GREY)
            c.setFont("Helvetica-Oblique", 6.5)
            c.drawCentredString(
                w / 2,
                h - 3 * mm,
                "Synthetic workflow",
            )

        else:
            # Real CGM experiment
            box_w = 42 * mm
            box_h = 22 * mm
            gap = (w - 3 * box_w) / 2
            y = h - 31 * mm

            boxes = [
                ("Prepared CGM", "One participant"),
                ("Time-Series Features", "Historical glucose"),
                ("60-Minute Target", "Future glucose"),
            ]

            for i, (title, subtitle) in enumerate(boxes):
                x = i * (box_w + gap)
                self.draw_box(
                    c, x, y, box_w, box_h,
                    title, subtitle,
                )

                if i > 0:
                    previous_x = (i - 1) * (box_w + gap)
                    self.arrow(
                        c,
                        previous_x + box_w,
                        y + box_h / 2,
                        x,
                        y + box_h / 2,
                    )

            # Two model branches
            branch_y = h - 67 * mm
            branch_w = 48 * mm
            left = 18 * mm
            right = w - branch_w - 18 * mm

            self.draw_box(
                c, left, branch_y, branch_w, 22 * mm,
                "Persistence Baseline",
                "Current reading as estimate",
                fill=AMBER,
            )
            self.draw_box(
                c, right, branch_y, branch_w, 22 * mm,
                "Random Forest",
                "Regression prediction",
                fill=GREEN,
            )

            self.arrow(
                c,
                w / 2,
                y,
                left + branch_w / 2,
                branch_y + 22 * mm,
            )
            self.arrow(
                c,
                w / 2,
                y,
                right + branch_w / 2,
                branch_y + 22 * mm,
            )

            result_w = 60 * mm
            result_x = (w - result_w) / 2
            result_y = 4 * mm

            self.draw_box(
                c, result_x, result_y, result_w, 20 * mm,
                "Test Evaluation",
                "MAE, RMSE and prediction plots",
                fill=CYAN,
            )

            self.arrow(
                c,
                left + branch_w / 2,
                branch_y,
                result_x + 10 * mm,
                result_y + 20 * mm,
            )
            self.arrow(
                c,
                right + branch_w / 2,
                branch_y,
                result_x + result_w - 10 * mm,
                result_y + 20 * mm,
            )


# ---------------------------------------------------------
# PAGE HEADER / FOOTER
# ---------------------------------------------------------
def draw_page(canvas, doc):
    canvas.saveState()

    page_w, page_h = A4

    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(18 * mm, page_h - 5.3 * mm, "GLUCOTWIN AI")

    canvas.setStrokeColor(BORDER)
    canvas.line(18 * mm, 14 * mm, page_w - 18 * mm, 14 * mm)

    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        18 * mm,
        9 * mm,
        "Educational research prototype | Not for medical decisions",
    )
    canvas.drawRightString(
        page_w - 18 * mm,
        9 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# ---------------------------------------------------------
# BUILD DOCUMENT
# ---------------------------------------------------------
def main():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="GlucoTwin AI - System Architecture",
        author="Molly Khandelwal",
        subject="Digital-twin-inspired glucose forecasting prototype",
    )

    story = []

    # COVER
    story.append(Spacer(1, 25 * mm))
    story.append(P("GLUCOTWIN AI", "CoverTitle"))
    story.append(
        P(
            "System Architecture & Technical Overview",
            "CoverSubtitle",
        )
    )
    story.append(Spacer(1, 7 * mm))

    story.append(
        callout(
            "A digital-twin-inspired educational prototype exploring "
            "patient-state representation and short-horizon glucose "
            "forecasting using synthetic EHR data and CGM time series.",
            CYAN,
        )
    )

    story.append(Spacer(1, 12 * mm))
    story.append(P("PROJECT SUMMARY", "SectionHeading"))

    story.append(
        P(
            "GlucoTwin AI combines data preparation, time-series feature "
            "engineering, regression-based forecasting, model evaluation, "
            "and an interactive Streamlit dashboard. It presents a "
            "synthetic patient workflow separately from a real CGM "
            "experiment involving one participant."
        )
    )

    summary = [
        ["Project", "GlucoTwin AI"],
        ["Author", "Molly Khandelwal"],
        ["Programming language", "Python"],
        ["Dashboard", "Streamlit"],
        ["Forecast horizon", "Approximately 60 minutes"],
        [
            "Repository",
            "github.com/MollyKhandelwal/GlucoTwin-AI",
        ],
        ["Document generated", date.today().isoformat()],
    ]

    story.append(
        make_table(
            ["Field", "Details"],
            summary,
            [45 * mm, 125 * mm],
        )
    )

    story.append(Spacer(1, 10 * mm))
    story.append(
        callout(
            "Important: This is a research and educational prototype. "
            "It is not a medical device and must not be used for diagnosis, "
            "treatment, or clinical decision-making.",
            AMBER,
        )
    )

    story.append(PageBreak())

    # ARCHITECTURE
    story.append(P("1. System Architecture", "SectionHeading"))
    story.append(
        P(
            "The synthetic workflow demonstrates the integration of "
            "generated patient attributes and generated glucose readings. "
            "The resulting features feed a forecasting and evaluation "
            "pipeline, with saved outputs displayed in the dashboard."
        )
    )

    story.append(ArchitectureDiagram("main", height=118 * mm))
    story.append(Spacer(1, 5 * mm))

    story.append(
        P(
            "The diagram is a logical view of the implemented prototype. "
            "The synthetic workflow and real CGM workflow remain separate; "
            "the diagram does not imply that synthetic EHR records belong "
            "to the real CGM participant.",
            "SmallCustom",
        )
    )

    story.append(PageBreak())

    # REAL CGM WORKFLOW
    story.append(P("2. Real CGM Experiment", "SectionHeading"))
    story.append(
        P(
            "The real-data workflow uses prepared Dexcom CGM readings "
            "from one participant. Historical glucose values are used "
            "to construct forecasting features and an approximately "
            "60-minute target. A persistence baseline and a Random Forest "
            "regressor are evaluated on the same recorded test samples."
        )
    )

    story.append(ArchitectureDiagram("real", height=105 * mm))
    story.append(Spacer(1, 4 * mm))

    story.append(P("Recorded test metrics", "SubHeading"))

    real_results = [
        ["Persistence baseline", "9.431", "14.156"],
        ["Random Forest", "10.266", "14.152"],
    ]

    story.append(
        make_table(
            ["Model", "MAE (mg/dL)", "RMSE (mg/dL)"],
            real_results,
            [75 * mm, 47.5 * mm, 47.5 * mm],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "Interpretation: Random Forest has higher MAE and marginally "
            "lower RMSE than the persistence baseline. It does not "
            "outperform the baseline on both metrics.",
            AMBER,
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        P(
            "These results are exploratory and apply only to the evaluated "
            "participant and test period. They do not establish "
            "generalization to other participants or clinical accuracy."
        )
    )

    story.append(PageBreak())

    # COMPONENTS
    story.append(P("3. Components and Responsibilities", "SectionHeading"))

    components = [
        [
            "Data preparation",
            "Loads and prepares synthetic data or prepared CGM readings.",
        ],
        [
            "Synthetic EHR",
            "Provides generated patient attributes for the synthetic experiment.",
        ],
        [
            "Feature engineering",
            "Creates historical glucose features and forecasting targets.",
        ],
        [
            "Persistence baseline",
            "Uses the current glucose value as the future estimate.",
        ],
        [
            "Random Forest",
            "Fits a regression model to historical glucose features.",
        ],
        [
            "Evaluation",
            "Calculates MAE and RMSE on held-out test samples.",
        ],
        [
            "Outputs",
            "Stores metrics, predictions, and visualizations.",
        ],
        [
            "Streamlit dashboard",
            "Displays patient overview, time series, forecasts, and evaluation.",
        ],
    ]

    story.append(
        make_table(
            ["Component", "Responsibility"],
            components,
            [43 * mm, 127 * mm],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(P("4. Technology Stack", "SectionHeading"))

    stack = [
        ["Python", "Core programming language"],
        ["Pandas and NumPy", "Data processing and feature preparation"],
        ["Scikit-learn", "Random Forest regression and evaluation"],
        ["Matplotlib", "Forecast and error visualization"],
        ["Streamlit", "Interactive dashboard"],
        ["CSV", "Data and experiment output storage"],
        ["Git and GitHub", "Version control and project hosting"],
    ]

    story.append(
        make_table(
            ["Technology", "Role"],
            stack,
            [45 * mm, 125 * mm],
        )
    )

    story.append(PageBreak())

    # METHODOLOGY
    story.append(P("5. Forecasting and Evaluation", "SectionHeading"))

    steps = [
        "<b>1. Load data:</b> Read the relevant synthetic or prepared CGM dataset.",
        "<b>2. Prepare timestamps:</b> Inspect ordering, missing values, and sampling intervals.",
        "<b>3. Build features:</b> Construct historical glucose features.",
        "<b>4. Define the target:</b> Estimate glucose approximately 60 minutes ahead.",
        "<b>5. Split chronologically:</b> Keep later observations for testing.",
        "<b>6. Evaluate:</b> Compare persistence and Random Forest predictions.",
        "<b>7. Save results:</b> Store metrics, predictions, and plots.",
        "<b>8. Visualize:</b> Present experiment outputs in Streamlit.",
    ]

    for step in steps:
        story.append(P(step))

    story.append(P("Evaluation metrics", "SubHeading"))

    metric_rows = [
        [
            "MAE",
            "Mean absolute error between predicted and observed glucose.",
        ],
        [
            "RMSE",
            "Root mean squared error; larger errors contribute more heavily.",
        ],
        [
            "Persistence baseline",
            "Predicts future glucose using the current reading.",
        ],
    ]

    story.append(
        make_table(
            ["Metric / Method", "Explanation"],
            metric_rows,
            [48 * mm, 122 * mm],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(P("6. Limitations and Responsible Use", "SectionHeading"))

    limitations = [
        "Synthetic EHR and CGM records are generated for demonstration.",
        "The real CGM experiment contains data from one participant.",
        "Synthetic EHR records are not a clinically matched record for the real CGM participant.",
        "Single-participant performance does not establish population-level generalization.",
        "External validation on additional participants is required.",
        "The dashboard must not be used to make medical decisions.",
    ]

    for item in limitations:
        story.append(P(f"&bull; {item}"))

    story.append(Spacer(1, 4 * mm))
    story.append(P("7. Future Improvements", "SectionHeading"))

    future = [
        "Evaluate on multiple independent participants.",
        "Improve time-series validation and model selection.",
        "Explore prediction intervals and uncertainty estimation.",
        "Improve missing-data handling and robustness.",
        "Conduct rigorous external validation before any clinical application.",
    ]

    for item in future:
        story.append(P(f"&bull; {item}"))

    story.append(Spacer(1, 7 * mm))
    story.append(
        callout(
            "GlucoTwin AI is an educational proof of concept. "
            "Its current results do not demonstrate clinical validity.",
            AMBER,
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(P("Project Repository", "SubHeading"))
    story.append(
        P(
            '<link href="https://github.com/MollyKhandelwal/GlucoTwin-AI" '
            'color="#087E8B">'
            'https://github.com/MollyKhandelwal/GlucoTwin-AI</link>'
        )
    )

    doc.build(
        story,
        onFirstPage=draw_page,
        onLaterPages=draw_page,
    )

    print("\nArchitecture PDF created successfully.")
    print(f"File: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
