from vision.image_analyzer import analyze_image, classifier, CANDIDATE_LABELS
from PIL import Image

print("raw classifier output:")
img = Image.open('sample_xray.jpg').convert('RGB')
out = classifier(img, candidate_labels=CANDIDATE_LABELS)
print(type(out))
print(out)

print("\nanalyze_image result:")
print(analyze_image('sample_xray.jpg'))
