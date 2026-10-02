from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "BT2024259_report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))

story = [
    Paragraph("Polynomial Regression Assignment", styles["TitleCenter"]),
    Paragraph("Roll number: BT2024259", styles["Heading2"]),
    Paragraph(
        "This report describes the polynomial regression models used for the two personalized datasets. "
        "The objective was to predict the continuous target y while respecting the degree limits in the assignment.",
        styles["BodyText"],
    ),
    Spacer(1, 10),
    Paragraph("1. Data and preprocessing", styles["Heading2"]),
    Paragraph(
        "The var1 dataset contains six input variables (x1-x6), while var2 contains three (x1-x3). "
        "Each training file contains 1,000 rows and a target column y; each test file contains 1,000 rows "
        "and the matching input columns. The supplied data are already numeric and cleaned, with values "
        "bounded approximately in [-1, 1]. No target information or rows from one variant were used for the other. "
        "PolynomialFeatures expands each row into all interaction and power terms whose total degree is at most d. "
        "The expanded columns are standardized before ordinary least-squares linear regression to improve numerical conditioning.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph("2. Degree selection", styles["Heading2"]),
    Paragraph(
        "For each allowed degree, I used five-fold cross-validation with shuffling and random seed 42. "
        "The same folds were used for R2 and MSE. R2 was the primary selection criterion because it measures "
        "the fraction of target variation explained; MSE was retained as a scale-dependent diagnostic. "
        "This procedure evaluates generalization rather than choosing the degree from training error, which would "
        "favor unnecessarily flexible polynomials.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Table(
        [
            ["Variant", "Degree range", "Chosen degree", "Mean CV R2", "Mean CV MSE"],
            ["var1", "1-10", "4", "0.9331", "0.6799"],
            ["var2", "1-20", "8", "0.9928", "0.2878"],
        ],
        colWidths=[0.9 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]),
    ),
    Spacer(1, 8),
    Paragraph(
        "The selected degree for var1 is 4. Its validation R2 improves through degree 4 and then drops, "
        "indicating that higher-degree terms begin to overfit the available samples. For var2, the validation "
        "score improves through degree 8 and declines at degree 9 and above. Degree 8 therefore provides the best "
        "bias-variance trade-off among the permitted choices.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph("3. Final fitting and inference", styles["Heading2"]),
    Paragraph(
        "After selecting the degree, I refit the complete pipeline on all 1,000 training rows for that variant and "
        "predicted the corresponding 1,000 test rows. The final files contain one column, y, and preserve test-row order, "
        "as required by sample_submission.csv. The script is deterministic: paths are resolved relative to solve.py, "
        "the cross-validation seed is fixed, and no random augmentation or external data is used.",
        styles["BodyText"],
    ),
    Spacer(1, 8),
    Paragraph("4. Deliverables and reproducibility", styles["Heading2"]),
    Paragraph(
        "Training and inference code: solve.py. Dependencies: requirements.txt. Generated prediction files: "
        "BT2024259_pred_var1.csv and BT2024259_pred_var2.csv. The repository is available at "
        "https://github.com/Kandi-Rohan/ML_assignment. Running python solve.py from the repository root regenerates both files.",
        styles["BodyText"],
    ),
]

document = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=0.7 * inch,
    leftMargin=0.7 * inch,
    topMargin=0.65 * inch,
    bottomMargin=0.65 * inch,
)
document.build(story)
print(f"Wrote {OUTPUT}")