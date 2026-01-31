import io

from makepdf import generate_pdf_report


def test_generate_pdf_report_creates_pdf():
    predictions = {
        "Diabetes": {
            "prob": 42.5,
            "inputs": {"Age": 45, "BMI": 26},
            "severity": "Moderate",
        }
    }

    pdf_buffer = generate_pdf_report(predictions, ["Diabetes"])
    content = pdf_buffer.getvalue()
    assert isinstance(pdf_buffer, io.BytesIO)
    assert content.startswith(b"%PDF")
    assert len(content) > 100
