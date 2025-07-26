# import cv2
# import os
# import torch
# from PIL import Image
# from model import SimpleCNN
# from ocr import index_to_char, devanagari_map
# from torchvision import transforms

# # Load model
# model = SimpleCNN(num_classes=len(index_to_char))
# model.load_state_dict(torch.load("ocr_model.pth"))
# model.eval()

# # Image transform
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
# ])

# def predict_char(image):
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(image)
#         pred_idx = torch.argmax(output, 1).item()
#     dev_char = index_to_char[pred_idx]
#     return dev_char

# def segment_characters(plate_path):
#     img = cv2.imread(plate_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     char_boxes = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if h > 20 and w > 10:  # basic filtering, tweak as needed
#             char_boxes.append((x, y, w, h))

#     # Sort left to right
#     char_boxes = sorted(char_boxes, key=lambda box: box[0])

#     characters = []
#     for i, (x, y, w, h) in enumerate(char_boxes):
#         char_crop = thresh[y:y+h, x:x+w]
#         char_img = Image.fromarray(char_crop).convert("L")
#         characters.append(char_img)

#     return characters

# def predict_plate(plate_path):
#     characters = segment_characters(plate_path)
#     predicted_text = ""
#     for char_img in characters:
#         dev_char = predict_char(char_img)
#         eng_char = devanagari_map.get(dev_char, dev_char)  # fallback to Devanagari
#         predicted_text += eng_char
#     print(f"Predicted Plate: {predicted_text}")

# # Run it
# predict_plate("/home/sarikaghimire/fyp-traffic/data/number-plate.jpg")


import cv2
import os
import torch
import numpy as np
from PIL import Image
from model import SimpleCNN
from ocr import index_to_char, devanagari_map
from torchvision import transforms

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "ocr_model.pth"
IMAGE_PATH = "/home/sarikaghimire/fyp-traffic/data/number-plate.jpg"
SAVE_SEGMENTS = True  # Set to False to disable saving char crops
SEGMENT_DIR = "char_segments"
INPUT_SIZE = 64
MIN_CHAR_HEIGHT = 20
MIN_CHAR_WIDTH = 10

# =========================
# MODEL SETUP
# =========================
model = SimpleCNN(num_classes=len(index_to_char))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# =========================
# HELPER: Pad & Resize Image
# =========================
def pad_and_resize(img, size=INPUT_SIZE):
    old_size = img.size  # (width, height)
    desired_size = max(old_size)

    # Create white background square
    new_img = Image.new("L", (desired_size, desired_size), 255)
    new_img.paste(img, ((desired_size - old_size[0]) // 2,
                        (desired_size - old_size[1]) // 2))

    return new_img.resize((size, size))

# =========================
# PREDICT SINGLE CHARACTER
# =========================
def predict_char(image):
    image = pad_and_resize(image)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, 1).item()
    return index_to_char[pred_idx]

# =========================
# SEGMENT CHARACTERS FROM PLATE
# =========================
def segment_characters(plate_path):
    if SAVE_SEGMENTS and not os.path.exists(SEGMENT_DIR):
        os.makedirs(SEGMENT_DIR)

    img = cv2.imread(plate_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary inverse threshold + morphological cleaning
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > MIN_CHAR_HEIGHT and w > MIN_CHAR_WIDTH:
            char_boxes.append((x, y, w, h))

    # Sort left to right
    char_boxes = sorted(char_boxes, key=lambda box: box[0])

    characters = []
    for i, (x, y, w, h) in enumerate(char_boxes):
        char_crop = thresh[y:y + h, x:x + w]
        char_img = Image.fromarray(char_crop).convert("L")
        if SAVE_SEGMENTS:
            char_img.save(os.path.join(SEGMENT_DIR, f"char_{i}.png"))
        characters.append(char_img)

    return characters

# =========================
# FULL PLATE PREDICTION
# =========================
def predict_plate(plate_path):
    characters = segment_characters(plate_path)
    predicted_text = ""

    print(f"Total Segmented Characters: {len(characters)}")
    for i, char_img in enumerate(characters):
        dev_char = predict_char(char_img)
        eng_char = devanagari_map.get(dev_char, dev_char)  # fallback
        predicted_text += eng_char
        print(f"Char {i + 1}: {dev_char} → {eng_char}")

    print("\n🔤 Predicted Plate Number:", predicted_text)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    predict_plate(IMAGE_PATH)
