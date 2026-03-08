print("🚀 Starting Lab Report Test...")

from parsing.lab_report_parser import convert_to_context
from llm.router import generate_medical_answer

# Simulated extracted values (pretend these came from a PDF)
lab_values = {
    "Hemoglobin": "9.8",
    "WBC": "13000"
}

print("📊 Lab Values:", lab_values)

lab_context = convert_to_context(lab_values)

print("📄 Converted Lab Context:", lab_context)

answer = generate_medical_answer(
    "Are my results concerning?",
    lab_context=lab_context
)

print("\n✅ Final Answer:\n")
print(answer)