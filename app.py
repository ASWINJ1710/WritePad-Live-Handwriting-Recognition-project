
import streamlit as st
import numpy as np
import hashlib
from PIL import Image

from streamlit_drawable_canvas import st_canvas

from utils.preprocessing import preprocess_canvas
from models.mixed_recognition import recognize_mixed_content


st.set_page_config(
    page_title="AI Handwriting Recognition",
    page_icon="✍️",
    layout="wide"
)

# SESSION STATE
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "last_processed" not in st.session_state:
    st.session_state.last_processed = None

if "results" not in st.session_state:
    st.session_state.results = []

if "editable_text" not in st.session_state:
    st.session_state.editable_text = ""


# HEADER
st.title("✍️ AI Handwriting Recognition")

st.write(
    "Write letters, numbers, symbols and equations. "
    "Your recognized digital output appears on the right."
)

st.divider()


# RECOGNITION MODE
recognition_mode = st.selectbox(
    "Choose Recognition Mode",
    ["Text", "Math", "Combined"],
    help=(
        "Text: Handwriting recognition. "
        "Math: Mathematical expression recognition. "
        "Combined: Letters, numbers and mathematical symbols."
    )
)
mode = recognition_mode.lower()


# TWO-PANEL LAYOUT
left, right = st.columns(2)


# LEFT PANEL: WRITING CANVAS
with left:

    st.subheader("📝 Writing Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=450,
        width=550,
        drawing_mode="freedraw",
        update_streamlit=True,
        key=f"canvas_{st.session_state.canvas_key}"
    )


# GET CANVAS IMAGE
image_array = canvas_result.image_data

has_drawing = False

if image_array is not None:

    has_drawing = np.any(
        image_array[:, :, :3] < 245
    )

    # SAVE ORIGINAL HANDWRITING IMAGE FOR DEBUGGING
    if has_drawing:

        debug_image = Image.fromarray(
            image_array.astype(np.uint8)
        )

        debug_image.save("handwriting_test.png")


# AUTOMATIC RECOGNITION
if has_drawing:

    image_bytes = np.ascontiguousarray(
        image_array
    ).tobytes()

    # Include mode so switching modes triggers recognition
    current_hash = hashlib.md5(
        image_bytes + mode.encode()
    ).hexdigest()

    if current_hash != st.session_state.last_processed:

        try:

            with st.spinner("🤖 Recognizing handwriting..."):

                processed_image = preprocess_canvas(
                    image_array
                )

                results = recognize_mixed_content(
                    processed_image,
                    mode=mode
                )

                st.session_state.results = results

                recognized_text = []

                for result in results:

                    recognized_text.append(
                        result["prediction"]
                    )

                st.session_state.editable_text = "\n".join(
                    recognized_text
                )

                st.session_state.last_processed = current_hash

        except Exception as error:

            st.error("Recognition failed.")
            st.exception(error)


# RIGHT PANEL: DIGITAL OUTPUT
with right:

    st.subheader("🤖 Recognized Digital Output")

    if st.session_state.results:

        for result in st.session_state.results:

            st.caption(
                f"Region {result['region']} | "
                f"{result['type'].upper()}"
            )

            if result["type"] == "math":

                st.latex(
                    result["prediction"]
                )

            else:

                st.markdown(
                    f"**{result['prediction']}**"
                )

            st.divider()

        st.subheader("✏️ Editable Output")

        st.text_area(
            "Correct the recognized text if needed",
            key="editable_text",
            height=180
        )

    else:

        st.info(
            "Start writing in the canvas. "
            "Your recognized digital output will appear here."
        )


# DEBUG INFORMATION
with st.expander("🔍 Debug Information"):

    st.write("Recognition mode:", mode)
    st.write("Canvas image available:", image_array is not None)
    st.write("Drawing detected:", has_drawing)

    if has_drawing:
        st.success(
            "Original handwriting saved as handwriting_test.png"
        )


# CLEAR BUTTON
st.divider()

if st.button(
    "🗑️ Clear Canvas and Output",
    use_container_width=True
):

    st.session_state.canvas_key += 1
    st.session_state.last_processed = None
    st.session_state.results = []
    st.session_state.editable_text = ""

    st.rerun()
