
from ultralytics import YOLO
import cv2
import json
from collections import Counter
from pathlib import Path

# Get the project folder paths
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "data" / "output"

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Vehicle classes supported by YOLO
VEHICLE_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def detect_one(image_path):
    # Detect vehicles in the image
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=1600,
        classes=list(VEHICLE_IDS.keys()),
        verbose=False
    )[0]

    # Count each type of vehicle
    counts = Counter()

    for box in result.boxes:
        vehicle_class = int(box.cls[0])
        counts[VEHICLE_IDS[vehicle_class]] += 1

    # Save the image with detection boxes
    output_path = OUTPUT_DIR / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_path), result.plot())

    # Return the counts
    return {
        "direction": image_path.stem,
        "car": counts["car"],
        "motorcycle": counts["motorcycle"],
        "bus": counts["bus"],
        "truck": counts["truck"],
        "total": sum(counts.values())
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    directions = ["north", "east", "south", "west"]
    results = []

    for direction in directions:
        image_path = IMAGE_DIR / f"{direction}.jpg"

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        print(f"Detecting vehicles in {direction} direction...")

        vehicle_counts = detect_one(image_path)
        results.append(vehicle_counts)

    # Display results
    print("\nFinal Vehicle Counts:")
    print(json.dumps(results, indent=2))

    # Save counts to a JSON file
    counts_path = BASE_DIR / "data" / "counts.json"

    with open(counts_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nCounts saved to: {counts_path}")
    print(f"Annotated images saved to: {OUTPUT_DIR}")
