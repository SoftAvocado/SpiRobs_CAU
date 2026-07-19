"""The single source of truth for what the detector can recognise.

This is the ONLY place to add or remove detectable objects. Edit ``TABLE_ITEMS``
below (add a string, remove a string) and re-run — no command-line flags, no
retraining. The final vocabulary handed to the open-vocabulary model is:

    DETECTION_CLASSES = the 80 standard COCO classes + TABLE_ITEMS (deduplicated)

Guidelines when editing TABLE_ITEMS:
  * Use concrete, everyday names ("power bank", not "portable energy device").
  * One object per string; lower-case.
  * More classes = slightly slower and more false positives. If detection gets
    noisy, trim items you don't need or raise --conf.
"""

from __future__ import annotations

# --- The 80 standard COCO classes (what YOLO detects out of the box) ---------
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# --- ~200 extra items commonly found on a table / desk -----------------------
# EDIT HERE to change what the detector looks for.
TABLE_ITEMS: list[str] = [
    # Writing & office
    "pen", "pencil", "marker", "highlighter", "eraser", "pencil sharpener",
    "ruler", "stapler", "staple remover", "paper clip", "binder clip",
    "push pin", "rubber band", "tape dispenser", "adhesive tape", "glue stick",
    "notebook", "notepad", "sticky note", "sheet of paper", "folder",
    "file folder", "ring binder", "envelope", "postage stamp", "business card",
    "index card", "calendar", "planner", "clipboard", "hole punch",
    "calculator", "pencil case", "desk organizer", "letter opener",
    "correction fluid", "crayon", "colored pencil", "fountain pen",
    "whiteboard marker",
    # Electronics & accessories
    "computer monitor", "desktop computer", "tablet", "smartphone",
    "phone charger", "charging cable", "usb cable", "hdmi cable", "power strip",
    "extension cord", "power adapter", "power bank", "headphones", "earbuds",
    "speaker", "microphone", "webcam", "mouse pad", "usb flash drive",
    "external hard drive", "memory card", "router", "printer", "e-reader",
    "smartwatch", "digital camera", "game controller", "projector",
    "docking station", "stylus", "laptop stand", "phone stand", "desk lamp",
    "led light strip", "wireless earbuds", "ring light", "graphics tablet",
    "network switch", "cable", "cable organizer",
    # Kitchen & dining
    "plate", "saucer", "mug", "coffee mug", "teacup", "teapot",
    "drinking glass", "tumbler", "pitcher", "thermos", "water bottle",
    "travel mug", "drinking straw", "napkin", "napkin holder", "placemat",
    "coaster", "tablecloth", "serving tray", "chopsticks", "spatula", "whisk",
    "tongs", "can opener", "bottle opener", "corkscrew", "cutting board",
    "salt shaker", "pepper shaker", "pepper grinder", "sugar bowl", "honey jar",
    "jam jar", "butter dish", "ketchup bottle", "mustard bottle",
    "olive oil bottle", "spice jar", "cereal box", "milk carton", "tea bag",
    "coffee grinder", "electric kettle", "blender", "french press",
    "mason jar", "food container", "lunch box", "aluminum foil",
    "paper towel roll",
    # Personal & bathroom
    "wallet", "purse", "keychain", "house key", "eyeglasses", "sunglasses",
    "glasses case", "wristwatch", "bracelet", "necklace", "comb", "hairbrush",
    "hand mirror", "lipstick", "nail polish", "perfume bottle", "deodorant",
    "lotion bottle", "hand sanitizer", "tissue box", "wet wipes", "cotton swab",
    "band aid", "pill bottle", "pill organizer", "thermometer", "face mask",
    "lip balm", "toothpaste", "razor", "makeup brush", "cosmetic bag",
    "nail clipper", "tweezers", "hair tie",
    # Food & drink
    "soda can", "beer bottle", "wine bottle", "juice box", "chocolate bar",
    "candy", "cookie", "cracker", "bag of chips", "grapes", "strawberry",
    "lemon", "tomato", "egg", "bread loaf", "bagel", "muffin", "croissant",
    "chewing gum", "nuts",
    # Tools & misc
    "screwdriver", "hammer", "wrench", "pliers", "tape measure",
    "utility knife", "flashlight", "battery", "coin", "lighter", "matchbox",
    "candle", "picture frame", "flower pot", "succulent", "magnifying glass",
    "dice", "playing cards", "rubik's cube", "stress ball", "magazine",
    "newspaper",
]


def _dedupe(names: list[str]) -> list[str]:
    """Case-insensitive de-duplication that preserves first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        cleaned = name.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


#: The full vocabulary the detector uses (COCO first, then table items).
DETECTION_CLASSES: list[str] = _dedupe(COCO_CLASSES + TABLE_ITEMS)
