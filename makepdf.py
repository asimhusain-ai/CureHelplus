import io
import textwrap
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.backends.backend_pdf import PdfPages
from helper import fetch_gemini_recommendations

def generate_pdf_report(predictions, selected_diseases):
    COLORS = {
        'bg': '#f7f9fc',
        'text': '#333333',
        'header': '#0d47a1',
        'accent': '#1976d2',
        'low_risk': '#4caf50',
        'medium_risk': '#ff9800',
        'high_risk': '#d32f2f'
    }

    pdf_buffer = io.BytesIO()

    # Normalize selected_diseases
    if selected_diseases:
        if isinstance(selected_diseases, str):
            if selected_diseases.lower() in ['full report', 'all']:
                selected_diseases = list(predictions.keys())
            else:
                selected_diseases = [selected_diseases]
        else:
            selected_diseases = list(selected_diseases)
    else:
        selected_diseases = list(predictions.keys())

    # Filter predictions based on selection
    selected_predictions = {d: predictions[d] for d in selected_diseases if d in predictions}

    def _plot_gauge(ax, risk, color):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Wedge((0, 0), 1, 0, 180, width=0.3, facecolor='#e0e0e0', alpha=0.7))
        ax.add_patch(Wedge((0, 0), 1, 180 - (risk / 100) * 180, 180, width=0.3, facecolor=color, alpha=1))
        ax.add_patch(Circle((0, 0), 0.7, facecolor='white'))
        ax.text(0, 0.15, f"{risk:.1f}%", ha='center', va='center', fontsize=20, fontweight='bold', color=color)
        ax.text(0, -0.05, "Risk Score", ha='center', va='center', fontsize=10, color='#555555')

    def _format_section(items, width=110):
        if not items:
            return "No data available."
        lines = []
        for idx, entry in enumerate(items, 1):
            bullet = f"{idx}. {entry}"
            wrapped = textwrap.fill(bullet, width=width, subsequent_indent="   ")
            lines.append(wrapped)
        return "\n".join(lines)

    with PdfPages(pdf_buffer) as pdf:
        for disease, data in selected_predictions.items():
            disease_lower = str(disease).strip().lower()
            display_disease = "Type-2 Diabetes" if disease_lower in {"diabetes", "type-2 diabetes", "type 2 diabetes", "type-2 diabetes mellitus", "type 2 diabetes mellitus"} else disease
            inputs = data.get("inputs", {})
            prob_candidates = (
                data.get("prob"),
                data.get("probability"),
                data.get("score"),
            )

            probability_value = 0.0
            for candidate in prob_candidates:
                if candidate is None:
                    continue
                try:
                    probability_value = float(candidate)
                    break
                except (TypeError, ValueError):
                    continue
            if 0 < probability_value <= 1:
                probability_value *= 100
            probability_value = max(0.0, min(probability_value, 100.0))
            risk = probability_value
            severity = data.get("severity", "N/A")

            recommendations = fetch_gemini_recommendations(display_disease, risk)
            prevention_text = _format_section(recommendations.get("prevention_measures", []))
            intervention_text = _format_section(recommendations.get("medicine_suggestions", []))

            risk_color = COLORS['high_risk'] if risk >= 75 else COLORS['medium_risk'] if risk > 40 else COLORS['low_risk']

            fig = plt.figure(figsize=(8.27, 11.69), facecolor=COLORS['bg'])
            # Use a grid layout so every section stays aligned and fits on one page.
            gs = fig.add_gridspec(
                nrows=12,
                ncols=2,
                left=0.06,
                right=0.94,
                top=0.93,
                bottom=0.08,
                hspace=0.35,
                wspace=0.5,
            )

            # Header
            ax_header = fig.add_subplot(gs[0, :])
            ax_header.axis('off')
            ax_header.text(0.5, 0.55, "CureHelp+", ha='center', va='center', fontsize=22, fontweight='bold', color=COLORS['header'])
            ax_header.text(
                0.5,
                0.15,
                "Disclaimer: This report is for informational purposes only. Always consult a doctor.",
                ha='center',
                va='center',
                fontsize=8,
                color='#555555',
                style='italic',
            )

            ax_disease = fig.add_subplot(gs[1, :])
            ax_disease.axis('off')
            disease_title = f"{display_disease} Risk"
            if severity != "N/A":
                disease_title += f" ({severity})"
            ax_disease.text(0.5, 0.5, disease_title, ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['accent'])

            # User inputs table
            ax_inputs = fig.add_subplot(gs[2:6, 0])
            ax_inputs.axis('off')
            if inputs:
                table_data = [[k, str(v)] for k, v in inputs.items()]
                table = ax_inputs.table(cellText=table_data, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left', colWidths=[0.5, 0.5])
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.25)
                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor('#dddddd')
                    if row == 0:
                        cell.set_facecolor(COLORS['header'])
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor(COLORS['bg'] if row % 2 == 0 else '#ffffff')

            # Risk gauge
            ax_gauge = fig.add_subplot(gs[2:6, 1])
            _plot_gauge(ax_gauge, risk, risk_color)

            # Risk reduction protocols
            ax_protocols = fig.add_subplot(gs[6:9, :])
            ax_protocols.axis('off')
            ax_protocols.text(
                0.0,
                0.98,
                "Risk Reduction Protocols",
                fontsize=8,
                fontweight='bold',
                color=COLORS['accent'],
                ha='left',
                va='top',
                transform=ax_protocols.transAxes,
            )
            ax_protocols.text(
                0.0,
                0.82,
                prevention_text,
                fontsize=6,
                color=COLORS['text'],
                ha='left',
                va='top',
                transform=ax_protocols.transAxes,
            )

            # Clinical interventions
            ax_interventions = fig.add_subplot(gs[9:12, :])
            ax_interventions.axis('off')
            ax_interventions.text(
                0.0,
                0.98,
                "Clinical Interventions",
                fontsize=8,
                fontweight='bold',
                color=COLORS['accent'],
                ha='left',
                va='top',
                transform=ax_interventions.transAxes,
            )
            ax_interventions.text(
                0.0,
                0.82,
                intervention_text,
                fontsize=6,
                color=COLORS['text'],
                ha='left',
                va='top',
                transform=ax_interventions.transAxes,
            )

            # Footer
            ax_footer = fig.add_axes([0.05, 0, 0.9, 0.02])
            ax_footer.axis('off')
            ax_footer.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=7, color='#888888')

            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer