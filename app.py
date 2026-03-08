import streamlit as st
import tempfile
import os
from PIL import Image

from parsing.lab_report_parser import extract_lab_values, convert_to_context
from vision.image_analyzer import analyze_image, convert_findings_to_context
from llm.router import generate_medical_answer
from security.jailbreak_guard import detect_jailbreak
from llm.llm import generate_text
from rag.retriever import retrieve

st.set_page_config(page_title="Medical RAG Chat", layout="wide")

st.title("Medical RAG Chatbot")
st.write("Upload a lab report PDF and/or an image, enter your question, and the app will use the RAG pipeline to answer using only the provided context and the knowledge base.")

query = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_pdf = st.file_uploader("Upload lab report (PDF)", type=["pdf"]) 
with col2:
    uploaded_image = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"]) 

if st.button("Ask"):

    if not query:
        st.warning("Please enter a question before asking.")
        st.stop()

    # Jailbreak protection
    if detect_jailbreak(query):
        st.error("⚠️ Unsafe or malicious query detected. Please ask a valid medical question.")
        st.stop()
    else:
        st.info("Processing inputs and generating answer...")

        combined_context = []

        # Handle PDF lab report
        if uploaded_pdf is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    pdf_path = tmp.name

                labs = extract_lab_values(pdf_path)
                st.subheader("Extracted Lab Values")
                st.write(labs)

                if labs:
                    lab_ctx = convert_to_context(labs)
                    st.write(lab_ctx)
                    combined_context.extend(lab_ctx)
            except Exception as e:
                st.error(f"Failed to parse lab report: {e}")

        # Handle image analysis
        if uploaded_image is not None:
            try:
                suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_image.read())
                    img_path = tmp.name

                findings = analyze_image(img_path)
                st.subheader("Image Findings")
                st.write(findings)

                if findings:
                    img_ctx = convert_findings_to_context(findings)
                    st.write(img_ctx)
                    combined_context.extend(img_ctx)

                # display the uploaded image
                try:
                    img = Image.open(img_path)
                    st.image(img, caption="Uploaded image")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")

        if len(combined_context) == 0:
            combined_context = None
        else:
            st.subheader("Combined Context")
            st.write(combined_context)

        try:
            answer = generate_medical_answer(query, lab_context=combined_context)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Failed to generate answer: {e}")

