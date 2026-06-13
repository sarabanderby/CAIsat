"""
FastAPI backend for object detection in satellite imagery.

This service receives images from the frontend, preprocesses them for YOLOv8-OBB,
sends to the OpenShift AI model endpoint, and returns detected objects.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import numpy as np
from PIL import Image
import cv2
import io
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

app = FastAPI(title="Satellite Object Detection API")

# Simple in-memory counter for detections
detection_counter = 0

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model endpoint from environment variable
MODEL_ENDPOINT = os.getenv(
    "MODEL_ENDPOINT",
    "http://yolov8m-satelite-predictor.caisat.svc.cluster.local:8080/v2/models/yolov8m-satelite/infer"
)

# DOTA class names (15 classes for YOLOv8-OBB)
CLASS_NAMES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool'
]

# Detection thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))


def preprocess_image(image: Image.Image, input_size: int = 640) -> tuple:
    """
    Preprocess image for YOLOv8 inference.

    Args:
        image: PIL Image
        input_size: model input size (default 640)

    Returns:
        Tuple of (request_data, scale, padding, original_shape)
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array
    img_array = np.array(image)

    # Get original dimensions
    orig_h, orig_w = img_array.shape[:2]

    # Calculate scale to fit image into input_size while maintaining aspect ratio
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)  # Gray padding

    # Calculate padding offsets
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2

    # Place resized image in center
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    # Convert to float32 and normalize to [0, 1]
    preprocessed = padded.astype(np.float32) / 255.0

    # Transpose from HWC to CHW format (required by YOLO)
    preprocessed = np.transpose(preprocessed, (2, 0, 1))

    # Add batch dimension
    preprocessed = np.expand_dims(preprocessed, axis=0)

    # Create KServe v2 inference request
    request_data = {
        "inputs": [
            {
                "name": "images",
                "shape": list(preprocessed.shape),
                "datatype": "FP32",
                "data": preprocessed.flatten().tolist()
            }
        ]
    }

    return request_data, scale, (pad_w, pad_h), (orig_w, orig_h)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                        iou_threshold: float = 0.45) -> tuple:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.

    Returns:
        Tuple of (boxes, scores, classes) after NMS
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Apply OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=iou_threshold
    )

    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], scores[indices], classes[indices]
    else:
        return np.array([]), np.array([]), np.array([])


def postprocess_yolov8_obb(output_data: dict, scale: float, padding: tuple,
                          original_shape: tuple, conf_threshold: float = 0.5,
                          iou_threshold: float = 0.45) -> List[Dict]:
    """
    Post-process YOLOv8-OBB output.

    Args:
        output_data: KServe v2 inference response
        scale: resize scale factor
        padding: (pad_w, pad_h) padding offsets
        original_shape: (orig_w, orig_h) original image dimensions
        conf_threshold: confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detections: [{"class": "ship", "confidence": 0.89, "box": [x1, y1, x2, y2]}, ...]
    """
    # Extract output from response
    outputs = output_data["outputs"][0]
    shape = outputs["shape"]
    data = outputs["data"]

    # Reshape output
    output = np.array(data).reshape(shape)

    # Remove batch dimension and transpose
    output = output.squeeze(0).T  # (8400, 20)

    # Split: bbox[0-3], classes[4-18], angle[19]
    boxes_cxcywh = output[:, :4]      # indices 0-3: bbox (cx, cy, w, h)
    class_scores = output[:, 4:19]    # indices 4-18: 15 class scores

    # Get max class score and ID
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert center format (cx, cy, w, h) to corners (x1, y1, x2, y2)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Apply NMS
    boxes, scores, class_ids = non_max_suppression(boxes_xyxy, scores, class_ids, iou_threshold)

    if len(boxes) == 0:
        return []

    # Scale back to original image
    boxes = np.array(boxes)
    pad_w, pad_h = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

    # SWAP WIDTH AND HEIGHT (rotate box 90 degrees)
    # Calculate centers and dimensions
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Swap width and height
    new_widths = heights
    new_heights = widths

    # Rebuild boxes with swapped dimensions
    boxes[:, 0] = centers_x - new_widths / 2
    boxes[:, 1] = centers_y - new_heights / 2
    boxes[:, 2] = centers_x + new_widths / 2
    boxes[:, 3] = centers_y + new_heights / 2

    # Clip to original image bounds
    orig_w, orig_h = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # Build result list
    detections = []
    for i in range(len(boxes)):
        box = boxes[i]
        score = float(scores[i])
        class_id = int(class_ids[i])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"

        detections.append({
            "class": class_name,
            "confidence": score,
            "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        })

    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return detections


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Satellite Object Detection API",
        "status": "operational",
        "model_endpoint": MODEL_ENDPOINT,
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/health")
async def health():
    """Kubernetes health check."""
    return {"status": "healthy"}


@app.get("/api/stats")
async def get_stats():
    """Get detection statistics."""
    return {
        "total_detections": detection_counter,
        "status": "operational",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": len(CLASS_NAMES)
    }


@app.post("/api/detect")
async def detect_objects(image: UploadFile = File(...)):
    """
    Detect objects in an uploaded satellite image.

    Args:
        image: Uploaded image file

    Returns:
        JSON with detections: {"detections": [...], "count": N}
    """
    global detection_counter

    try:
        print("=== DETECTION REQUEST STARTED ===")
        # Read and decode image
        print("Reading uploaded file...")
        contents = await image.read()
        print(f"File size: {len(contents)} bytes")

        print("Decoding image...")
        img = Image.open(io.BytesIO(contents))
        print(f"Received image: {img.size[0]}x{img.size[1]}, mode: {img.mode}")

        # Preprocess image
        print("Preprocessing image...")
        request_data, scale, padding, original_shape = preprocess_image(img)
        print(f"Input shape: {request_data['inputs'][0]['shape']}")
        print(f"Scale: {scale:.4f}, Padding: {padding}, Original: {original_shape}")

        print(f"Sending request to model endpoint: {MODEL_ENDPOINT}")

        # Send to model endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                MODEL_ENDPOINT,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"Model response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Model endpoint error: {error_text}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Model inference failed: {error_text}"
                    )

                result = await response.json()
                print(f"Response keys: {list(result.keys())}")

        print("Model inference successful")

        # Postprocess output
        print("Starting postprocess...")
        detections = postprocess_yolov8_obb(
            result,
            scale,
            padding,
            original_shape,
            conf_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        print(f"Found {len(detections)} objects")

        # Increment counter
        detection_counter += 1
        print(f"Total detections processed: {detection_counter}")

        # Return JSON response
        response_data = {
            "detections": detections,
            "count": len(detections),
            "image_size": {"width": original_shape[0], "height": original_shape[1]}
        }

        print("=== DETECTION REQUEST COMPLETED ===")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"Error processing image: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
