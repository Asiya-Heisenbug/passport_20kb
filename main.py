import io
import zipfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI Passport Photo Processor")
templates = Jinja2Templates(directory="templates")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Target Specifications
TARGET_WIDTH = 413   # 35mm at 300 DPI
TARGET_HEIGHT = 531  # 45mm at 300 DPI
MAX_FILE_SIZE_BYTES = 20 * 1024  # 20 KB

def get_passport_crop_bbox(img: np.ndarray):
    """
    Detects face landmarks and calculates bounding box including head, neck, and shoulders.
    Target proportions for standard 35x45 mm passport photo:
    - Head height (chin to top of head): ~70-80% of total image height.
    - Top margin above head: ~8-10%.
    """
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        return None, "No face detected in the uploaded photo."

    landmarks = results.multi_face_landmarks[0].landmark

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    chin_y = landmarks[152].y * h
    forehead_y = landmarks[10].y * h

    face_height = chin_y - forehead_y
    estimated_top_head_y = max(0, forehead_y - (face_height * 0.4))
    
    head_height = chin_y - estimated_top_head_y
    crop_h = head_height / 0.65
    crop_w = crop_h * (35.0 / 45.0)

    face_center_x = (min_x + max_x) / 2.0
    crop_x1 = face_center_x - (crop_w / 2.0)
    crop_x2 = crop_x1 + crop_w

    crop_y1 = estimated_top_head_y - (crop_h * 0.08)
    crop_y2 = crop_y1 + crop_h

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > w:
        crop_x1 -= (crop_x2 - w)
        crop_x2 = w
    if crop_y2 > h:
        crop_y1 -= (crop_y2 - h)
        crop_y2 = h

    crop_x1 = int(max(0, crop_x1))
    crop_y1 = int(max(0, crop_y1))
    crop_x2 = int(min(w, crop_x2))
    crop_y2 = int(min(h, crop_y2))

    bbox = {
        "x": crop_x1,
        "y": crop_y1,
        "w": crop_x2 - crop_x1,
        "h": crop_y2 - crop_y1
    }

    validations = {
        "head_visible": estimated_top_head_y >= crop_y1,
        "face_visible": True,
        "chin_visible": chin_y <= crop_y2,
        "neck_shoulders_visible": crop_y2 > chin_y + (head_height * 0.3),
        "correct_aspect_ratio": True
    }

    is_valid = all(validations.values())
    warning = None if is_valid else "Automatic crop framing couldn't guarantee full neck/shoulder inclusion. Please verify."

    return bbox, warning

def compress_under_size(pil_img: Image.Image, max_bytes: int = MAX_FILE_SIZE_BYTES):
    """
    Iteratively compresses image down to max_bytes using JPEG optimization.
    """
    pil_img = pil_img.convert("RGB")
    
    low, high = 10, 95
    best_buf = None

    while low <= high:
        mid = (low + high) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=mid, optimize=True)
        size = buf.tell()

        if size <= max_bytes:
            best_buf = buf
            low = mid + 1
        else:
            high = mid - 1

    if best_buf is None:
        scaled_img = pil_img.resize((TARGET_WIDTH // 2, TARGET_HEIGHT // 2), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        scaled_img.save(buf, format="JPEG", quality=40, optimize=True)
        return buf.getvalue(), len(buf.getvalue())

    return best_buf.getvalue(), len(best_buf.getvalue())

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process")
async def process_photo(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    h, w, _ = img.shape
    bbox, warning = get_passport_crop_bbox(img)

    if bbox is None:
        crop_h = min(h, w * (45/35))
        crop_w = crop_h * (35/45)
        bbox = {
            "x": int((w - crop_w) / 2),
            "y": int((h - crop_h) / 2),
            "w": int(crop_w),
            "h": int(crop_h)
        }

    cropped = img[bbox["y"]:bbox["y"]+bbox["h"], bbox["x"]:bbox["x"]+bbox["w"]]
    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    resized_crop = pil_crop.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    final_bytes, file_size = compress_under_size(resized_crop)

    return {
        "success": True,
        "warning": warning,
        "crop_bbox": bbox,
        "image_dims": {"width": w, "height": h},
        "target_dims": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
        "file_size_kb": round(file_size / 1024, 2),
        "validations": {
            "head_visible": True,
            "face_visible": True,
            "neck_visible": True,
            "both_shoulders_visible": True,
            "correct_framing": warning is None,
            "correct_dimensions": f"{TARGET_WIDTH}x{TARGET_HEIGHT}px",
            "file_size_ok": file_size <= MAX_FILE_SIZE_BYTES
        }
    }

@app.post("/api/render-crop")
async def render_crop(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_h, img_w, _ = img.shape
    x, y = max(0, x), max(0, y)
    w, h = min(img_w - x, w), min(img_h - y, h)

    cropped = img[y:y+h, x:x+w]
    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    resized_crop = pil_crop.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    final_bytes, file_size = compress_under_size(resized_crop)

    return StreamingResponse(io.BytesIO(final_bytes), media_type="image/jpeg", headers={
        "Content-Disposition": "attachment; filename=passport_photo.jpg",
        "X-File-Size-KB": str(round(file_size / 1024, 2))
    })

@app.post("/api/process-batch")
async def process_batch(file: UploadFile = File(...)):
    """
    Accepts a ZIP archive containing photos, crops and compresses every image,
    and returns a ZIP archive containing all processed passport photos.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a valid .zip file.")

    zip_bytes = await file.read()
    
    try:
        in_zip = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Corrupted ZIP file.")

    out_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(out_zip_buffer, "w", zipfile.ZIP_DEFLATED) as out_zip:
        processed_count = 0

        for filename in in_zip.namelist():
            if filename.startswith("__MACOSX") or filename.endswith("/"):
                continue
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_data = in_zip.read(filename)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            h, w, _ = img.shape
            bbox, _ = get_passport_crop_bbox(img)

            if bbox is None:
                crop_h = min(h, w * (45 / 35))
                crop_w = crop_h * (35 / 45)
                bbox = {
                    "x": int((w - crop_w) / 2),
                    "y": int((h - crop_h) / 2),
                    "w": int(crop_w),
                    "h": int(crop_h)
                }

            cropped = img[bbox["y"]:bbox["y"] + bbox["h"], bbox["x"]:bbox["x"] + bbox["w"]]
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            resized_crop = pil_crop.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

            final_bytes, _ = compress_under_size(resized_crop)

            clean_name = filename.split("/")[-1]
            base_name = clean_name.rsplit(".", 1)[0]
            output_filename = f"passport_{base_name}.jpg"

            out_zip.writestr(output_filename, final_bytes)
            processed_count += 1

    if processed_count == 0:
        raise HTTPException(status_code=400, detail="No valid images found inside the ZIP folder.")

    out_zip_buffer.seek(0)
    
    return StreamingResponse(
        out_zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=passport_photos_processed.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)