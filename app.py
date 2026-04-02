from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
import cv2
import numpy as np
import face_recognition # type: ignore
import uuid
import os

app = FastAPI()

OUTPUT_DIR = "annotated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory="annotated_images"), name="images")

@app.post("/match")
async def match_face(
    reference: UploadFile = File(...),
    group: UploadFile = File(...)
):
    try:
        # Read images
        ref_bytes = await reference.read()
        grp_bytes = await group.read()

        ref_np = np.frombuffer(ref_bytes, np.uint8)
        grp_np = np.frombuffer(grp_bytes, np.uint8)

        ref_img = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
        grp_img = cv2.imdecode(grp_np, cv2.IMREAD_COLOR)

        # Convert to RGB
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        grp_rgb = cv2.cvtColor(grp_img, cv2.COLOR_BGR2RGB)

        # Encode reference
        ref_encodings = face_recognition.face_encodings(ref_rgb)
        if not ref_encodings:
            raise HTTPException(400, "No face in reference image")

        ref_encoding = ref_encodings[0]

        # Detect group faces
        locations = face_recognition.face_locations(grp_rgb)
        encodings = face_recognition.face_encodings(grp_rgb, locations)

        if not encodings:
            raise HTTPException(400, "No faces in group image")

        # Find best match
        best_idx = None
        best_distance = 999

        for i, enc in enumerate(encodings):
            dist = face_recognition.face_distance([ref_encoding], enc)[0]
            if dist < best_distance:
                best_distance = dist
                best_idx = i

        if best_distance > 0.6:
            raise HTTPException(400, "No match found")

        top, right, bottom, left = locations[best_idx]

        cx = (left + right) // 2
        h = bottom - top

        arrow_start = (
            cx + int(h * 0.5),
            top - int(h * 1.2)
        )

        arrow_end = (cx, top)

        cv2.arrowedLine(
            grp_img,
            arrow_start,
            arrow_end,
            (0, 0, 255),
            3,
            tipLength=0.2
        )

        # Save image
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, grp_img)

        return JSONResponse({
            "success": True,
            "image_url": f"http://127.0.0.1:8000/images/{filename}",
            "distance": float(best_distance)
        })

    except Exception as e:
        raise HTTPException(500, str(e))