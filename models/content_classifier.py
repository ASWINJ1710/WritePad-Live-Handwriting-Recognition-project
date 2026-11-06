from models.math_recognition import recognize_math
from models.text_recognition import recognize_text


def classify_content(image):

    text_result = recognize_text(image).strip()
    math_result = recognize_math(image).strip()

    math_symbols = [
        "+", "=", "\\frac", "\\sqrt",
        "\\times", "\\div", "^"
    ]

    math_score = sum(
        1 for symbol in math_symbols
        if symbol in math_result
    )

    # Basic decision rule
    if math_score >= 2:
        content_type = "math"
    else:
        content_type = "text"

    return {
        "type": content_type,
        "text_prediction": text_result,
        "math_prediction": math_result,
        "math_score": math_score
    }