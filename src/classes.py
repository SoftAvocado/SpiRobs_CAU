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
    "pen", "pencil", "marker", "eraser", "pencil sharpener", "ruler",
    "stapler", "paper clip", "binder clip", "rubber band", "tape",
    "glue stick", "notebook", "sticky note", "paper", "folder", "binder",
    "envelope", "card", "calendar", "clipboard", "calculator", "pencil case",

    # Electronics & accessories
    "tablet", "cable", "power strip", "power adapter", "power bank",
    "headphones", "earbuds", "speaker", "microphone", "webcam", "mouse pad",
    "usb flash drive", "watch", "camera", "game controller", "stylus",
    "desk lamp",

    # Kitchen & dining
    "plate", "mug", "teapot", "pitcher", "thermos", "drinking straw",
    "napkin", "coaster", "tray", "chopsticks", "spatula", "whisk", "tongs",
    "can opener", "bottle opener", "corkscrew", "cutting board", "shaker",
    "pepper grinder", "sugar bowl", "jar", "butter dish", "condiment bottle",
    "cereal box", "milk carton", "tea bag", "coffee grinder", "electric kettle",
    "french press", "food container", "paper towel roll",

    # Personal & bathroom
    "wallet", "keychain", "key", "glasses", "glasses case", "bracelet",
    "necklace", "comb", "hairbrush", "hand mirror", "lipstick", "nail polish",
    "perfume bottle", "deodorant", "lotion bottle", "hand sanitizer",
    "spray bottle", "tissue box", "wet wipes", "pill bottle", "pill organizer",
    "face mask", "lip balm", "toothpaste", "razor", "makeup brush",
    "cosmetic bag", "hair tie", "toy",

    # Food & drink
    "can", "juice box", "chocolate bar", "candy", "cookie", "cracker",
    "bag of chips", "grapes", "strawberry", "lemon", "tomato", "egg", "bread",
    "bagel", "muffin", "croissant",

    # Tools & misc
    "screwdriver", "hammer", "wrench", "pliers", "tape measure",
    "utility knife", "flashlight", "battery", "coin", "lighter", "matches",
    "candle", "picture frame", "magnifying glass", "dice", "playing cards",
    "magazine", "newspaper",
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
