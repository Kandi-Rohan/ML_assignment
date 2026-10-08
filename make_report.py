from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
OUTPUT = ROOT / "BT2024259_report.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#718096"))
        
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 0.7 * inch, 0.4 * inch, page_str)
        self.setStrokeColor(HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(0.7 * inch, 0.52 * inch, A4[0] - 0.7 * inch, 0.52 * inch)
        
        self.restoreState()

styles = getSampleStyleSheet()

# Typography and Hierarchy
main_title = ParagraphStyle(
    "MainTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    alignment=1, # Center
    textColor=HexColor("#1a202c"),
    spaceAfter=4,
)

sub_title = ParagraphStyle(
    "SubTitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=13,
    alignment=1,
    textColor=HexColor("#4a5568"),
    spaceAfter=10,
)

meta_style = ParagraphStyle(
    "MetaText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    alignment=1,
    textColor=HexColor("#2d3748"),
)

sec_title = ParagraphStyle(
    "SecTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14.5,
    textColor=HexColor("#1a365d"),
    spaceBefore=7,
    spaceAfter=3,
)

body_text = ParagraphStyle(
    "BodyText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.6,
    textColor=HexColor("#2d3748"),
    spaceAfter=4,
)

bullet_item = ParagraphStyle(
    "BulletItem",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.3,
    leading=11.2,
    leftIndent=10,
    textColor=HexColor("#2d3748"),
    spaceAfter=2,
)

fig_caption = ParagraphStyle(
    "FigCaption",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=7.8,
    leading=10,
    textColor=HexColor("#4a5568"),
    alignment=1, # Center
    spaceBefore=3,
    spaceAfter=6,
)

code_snippet = ParagraphStyle(
    "CodeSnippet",
    parent=styles["Normal"],
    fontName="Courier",
    fontSize=8.2,
    leading=11,
    textColor=HexColor("#1a202c"),
)

story = []

# ==============================================================================
# PAGE 1: Overview, Pipeline Architecture & Selected Summary
# ==============================================================================
story.append(Paragraph("Polynomial Regression: Model Results", main_title))
story.append(Paragraph("Machine Learning Assignment 1 | Two Geothermal Problems", sub_title))

# Student Details Card
info_data = [
    [
        Paragraph("<b>Student Name:</b> Kandi Rohan", meta_style),
        Paragraph("<b>Roll Number:</b> BT2024259", meta_style),
    ]
]
info_table = Table(info_data, colWidths=[3.3 * inch, 3.3 * inch])
info_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#edf2f7")),
    ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#cbd5e0")),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
]))
story.append(info_table)
story.append(Spacer(1, 8))

story.append(Paragraph("1. Introduction and Data", sec_title))
story.append(Paragraph(
    "This report uses polynomial regression on two datasets. "
    "<b>Problem 1 (var1)</b> predicts turbine power score from six input values using degrees up to 10. "
    "<b>Problem 2 (var2)</b> predicts thermal score from three location values using degrees up to 20. "
    "Each problem has 1,000 training rows and 1,000 test rows.",
    body_text
))

story.append(Paragraph("2. Method Used", sec_title))
story.append(Paragraph(
    "I tested several polynomial models and compared their errors:",
    body_text
))
story.append(Paragraph(
    "&bull; <b>Polynomial features:</b> The model uses powers of each input and their interactions up to the selected degree.",
    bullet_item
))
story.append(Paragraph(
    "&bull; <b>Scaling:</b> The expanded features were scaled so that large powers did not dominate the model.",
    bullet_item
))
story.append(Paragraph(
    "&bull; <b>Models:</b> I compared Linear Regression, Ridge, Lasso, and Elastic Net with penalty values 0.01, 0.1, 1.0, and 10.0.",
    bullet_item
))
story.append(Paragraph(
    "&bull; <b>Validation:</b> I used shuffled 5-fold cross-validation with seed 42. Models were compared using average R2 and MSE.",
    bullet_item
))

story.append(Spacer(1, 4))
story.append(Paragraph("3. Best Models", sec_title))
story.append(Paragraph(
    "The best model for each problem was selected from all tested choices:",
    body_text
))

win_data = [
    ["Problem", "Phase", "Inputs", "Selected Algorithm", "Opt. Degree", "Penalty alpha", "CV R2", "CV MSE"],
    ["var1", "Steam Turbine Power", "6", "Lasso Regression", "5", "0.01", "0.9703", "0.3013"],
    ["var2", "Reservoir Thermal Map", "3", "Ridge Regression", "12", "1.0", "0.9935", "0.2566"],
]
win_table = Table(win_data, colWidths=[0.6 * inch, 1.6 * inch, 0.6 * inch, 1.3 * inch, 0.8 * inch, 0.7 * inch, 0.6 * inch, 0.6 * inch])
win_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a365d")),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f7fafc")),
    ("BACKGROUND", (0, 2), (-1, 2), HexColor("#edf2f7")),
]))
story.append(win_table)

# ==============================================================================
# PAGE 2: Degree Sweeps, Error Curves & Complexity Selection
# ==============================================================================
story.append(Paragraph("4. Error by Polynomial Degree", sec_title))
story.append(Paragraph(
    "The plots show how the validation error changes as the polynomial degree increases. "
    "Very small degrees are too simple, while very large degrees can fit noise. "
    "Ridge and Lasso help control this problem.",
    body_text
))

img_curves = Image(str(PLOTS_DIR / "fig2_cv_mse_comparison.png"), width=6.8 * inch, height=2.45 * inch)
story.append(img_curves)
story.append(Paragraph("<b>Figure 1:</b> Average validation MSE for different degrees. Ridge gives the best result for Problem 2 at degree 12.", fig_caption))

story.append(Spacer(1, 4))
story.append(Paragraph("5. Choosing the Degree", sec_title))
story.append(Paragraph(
    "For Problem 1, Lasso at degree 5 has the lowest validation MSE of 0.3013. "
    "After degree 5, the error does not improve enough to justify a more complicated model.",
    body_text
))

img_bars = Image(str(PLOTS_DIR / "fig3_var1_best_mse_bars.png"), width=6.6 * inch, height=2.75 * inch)
story.append(img_bars)
story.append(Paragraph("<b>Figure 2:</b> Lowest validation MSE found at each degree for Problem 1. The best result is Lasso at degree 5.", fig_caption))

story.append(Spacer(1, 4))
img_bars2 = Image(str(PLOTS_DIR / "fig6_var2_best_mse_bars.png"), width=6.6 * inch, height=2.75 * inch)
story.append(img_bars2)
story.append(Paragraph("<b>Figure 3:</b> Lowest validation MSE found at each degree for Problem 2. The best result is Ridge at degree 12.", fig_caption))

story.append(Spacer(1, 4))
img_r2 = Image(str(PLOTS_DIR / "fig5_cv_r2_curves.png"), width=6.6 * inch, height=2.45 * inch)
story.append(img_r2)
story.append(Paragraph("<b>Figure 4:</b> Average validation R2 by degree for both problems and all tested model settings.", fig_caption))

story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Important observation:</b> Ordinary Linear Regression becomes unstable at high degrees. Its Problem 1 error rises from 0.6799 at degree 4 to 78.4 at degree 6. "
    "For Problem 2, its error reaches 1160.8 at degree 20. Ridge and Lasso give more stable results.",
    body_text
))

# ==============================================================================
# PAGE 3: Out-of-Fold Goodness-of-Fit & Residual Diagnostics
# ==============================================================================
story.append(Paragraph("6. Model Predictions", sec_title))
story.append(Paragraph(
    "The next plot compares predicted values with the real training values from cross-validation. "
    "Points close to the diagonal line show good predictions. The colours show the size of the prediction error.",
    body_text
))

img_fit = Image(str(PLOTS_DIR / "fig4_actual_vs_predicted.png"), width=6.8 * inch, height=2.5 * inch)
story.append(img_fit)
story.append(Paragraph("<b>Figure 5:</b> Predicted values compared with actual values. Problem 1 gives R2 = 0.9705 and Problem 2 gives R2 = 0.9939.", fig_caption))

story.append(Spacer(1, 4))
story.append(Paragraph("7. Prediction Errors", sec_title))
story.append(Paragraph(
    "The residual plot checks whether the errors are spread evenly. "
    "For Problem 1, the errors are centred close to zero and have a standard deviation of about 0.549.",
    body_text
))

img_res = Image(str(PLOTS_DIR / "fig_residuals.png"), width=6.6 * inch, height=2.2 * inch)
story.append(img_res)
story.append(Paragraph("<b>Figure 6:</b> Prediction errors for Problem 1 using Lasso degree 5.", fig_caption))

img_res2 = Image(str(PLOTS_DIR / "fig_residuals_var2.png"), width=6.6 * inch, height=2.2 * inch)
story.append(img_res2)
story.append(Paragraph("<b>Figure 7:</b> Prediction errors for Problem 2 using Ridge degree 12.", fig_caption))

# ==============================================================================
# PAGE 4: Comparative Discussion, Deliverables & Reproducibility
# ==============================================================================
story.append(Paragraph("8. Interpretation of Results", sec_title))
story.append(Paragraph(
    "The two problems have different input data, so different models work best:",
    body_text
))
story.append(Paragraph(
    "&bull; <b>Problem 1:</b> Lasso with degree 5 is best. It can ignore some less useful polynomial terms and gives a low error.",
    bullet_item
))
story.append(Paragraph(
    "&bull; <b>Problem 2:</b> Ridge with degree 12 is best. It keeps the useful terms but reduces the effect of very large coefficients.",
    bullet_item
))
story.append(Paragraph(
    "&bull; <b>Repeated points:</b> Some Problem 2 coordinates appear more than once. Their median variance is about 0.255, which is close to the final validation MSE of 0.2566.",
    bullet_item
))

story.append(Spacer(1, 4))
story.append(Paragraph("9. Final Files", sec_title))
story.append(Paragraph(
    "The final models were trained again using all 1,000 training rows. The prediction files contain one column named y and keep the same order as the test rows:",
    body_text
))

deliv_data = [
    ["Phase", "Dataset", "Winning Configuration", "Output File", "Row Count", "Target Col."],
    ["Phase 1", "Turbine Calibration", "Lasso (deg=5, alpha=0.01)", "BT2024259_pred_var1.csv", "1000", "y"],
    ["Phase 2", "Thermal Reservoir", "Ridge (deg=12, alpha=1.0)", "BT2024259_pred_var2.csv", "1000", "y"],
]
deliv_table = Table(deliv_data, colWidths=[0.8 * inch, 1.4 * inch, 1.6 * inch, 1.7 * inch, 0.7 * inch, 0.6 * inch])
deliv_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2d3748")),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.8),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f7fafc")),
    ("BACKGROUND", (0, 2), (-1, 2), HexColor("#edf2f7")),
]))
story.append(deliv_table)

story.append(Spacer(1, 6))
story.append(Paragraph(
    '<b>GitHub Code Repository:</b> '
    '<link href="https://github.com/Kandi-Rohan/ML_assignment" color="blue">'
    'https://github.com/Kandi-Rohan/ML_assignment</link>',
    body_text,
))

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=0.7 * inch,
    rightMargin=0.7 * inch,
    topMargin=0.6 * inch,
    bottomMargin=0.6 * inch,
)

doc.build(story, canvasmaker=NumberedCanvas)
print(f"Successfully compiled {OUTPUT}")
