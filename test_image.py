from vision.image_analyzer import analyze_image, convert_findings_to_context
from llm.router import generate_medical_answer

# Analyze uploaded image
findings = analyze_image("sample_xray.jpg")

print("🩻 Raw Findings:", findings)

image_context = convert_findings_to_context(findings)

print("🧠 Converted Context:", image_context)

answer = generate_medical_answer(
    "Do you see anything unusual?",
    lab_context=image_context   # reused channel
)

print("\nFinal Answer:\n", answer)