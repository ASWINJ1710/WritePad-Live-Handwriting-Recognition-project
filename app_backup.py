import streamlit as st
import numpy as np
import io
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

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "last_processed" not in st.session_state:
    st.session_state.last_processed = None

if "results" not in st.session_state:
    st.session_state.results = []

if "editable_text" not in st.session_state:
    st.session_state.editable_text = ""


st.title("✍️ AI Handwriting & Symbol Recognition")
st.write("Write text, numbers, symbols and equations. View the recognized output separately.")

st.divider()

left, right = st.columns(2)

# LEFT: WRITING CANVAS
with left:
    st.subheader("📝 Writing Canvas")

    stroke_width = st.slider(
        "Pen Thickness",
        1, 15, 4,
        key="pen_width"
    )

    stroke_color = st.color_picker(
        "Pen Color",
        "#000000"
    )

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=350,
        width=600,
        drawing_mode="freedraw",
        key=f"writing_canvas_{st.session_state.canvas_key}"
    )

# RIGHT: DETECTION CANVAS
with right:
    st.subheader("🔍 Detection Canvas")

    if "original_image" in st.session_state:
        st.image(
            st.session_state.original_image,
            caption="Your detected handwriting",
            use_container_width=True
        )
    else:
        st.info("Your handwriting image will appear here.")


image_array = canvas_result.image_data

has_drawing = False

if image_array is not None:
    has_drawing = np.any(image_array[:, :, :3] < 245)


st.divider()

# AUTOMATIC RECOGNITION
st.subheader("🤖 Recognition Results")

if has_drawing:

    image_bytes = np.ascontiguousarray(
        image_array
    ).tobytes()

    current_hash = hashlib.md5(
        image_bytes
    ).hexdigest()

    if current_hash != st.session_state.last_processed:

        try:
            with st.spinner("Recognizing handwriting..."):

                original_image = Image.fromarray(
                    image_array.astype(np.uint8)
                ).convert("RGB")

                processed_image = preprocess_canvas(
                    image_array
                )

                results = recognize_mixed_content(
                    processed_image
                )

                st.session_state.original_image = original_image
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

# SHOW RECOGNITION
if st.session_state.results:

    for result in st.session_state.results:

        st.markdown(
            f"**Region {result['region']} — {result['type'].upper()}**"
        )

        if result["type"] == "math":
            st.latex(result["prediction"])
        else:
            st.write(result["prediction"])

        st.divider()

    st.subheader("✏️ Editable Recognized Output")

    st.text_area(
        "Edit your recognized text and symbols",
        key="editable_text",
        height=150
    )

elif not has_drawing:
    st.caption("Start writing in the left canvas.")

# CLEAR BUTTON
if st.button("🗑️ Clear Both Canvases", use_container_width=True):

    st.session_state.canvas_key += 1
    st.session_state.last_processed = None
    st.session_state.results = []
    st.session_state.editable_text = ""

    if "original_image" in st.session_state:
        del st.session_state.original_image

    st.rerun()