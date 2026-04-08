from fastapi import FastAPI, UploadFile, File, HTTPException, Request # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from deepface import DeepFace # type: ignore
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

OUTPUT_DIR = "annotated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")


@app.post("/match")
async def match_face(request: Request,
                     reference: UploadFile = File(...),
                     group: UploadFile = File(...)):

    try:
        # Read images
        ref_bytes = await reference.read()
        grp_bytes = await group.read()

        ref_np = np.frombuffer(ref_bytes, np.uint8)
        grp_np = np.frombuffer(grp_bytes, np.uint8)

        ref_img = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
        grp_img = cv2.imdecode(grp_np, cv2.IMREAD_COLOR)

        if ref_img is None or grp_img is None:
            raise HTTPException(400, "Invalid image")

        # Detect faces in group using Haarcascade
        gray = cv2.cvtColor(grp_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            raise HTTPException(400, "No faces in group image")

        best_match = None
        best_score = 999
        best_face = None

        # Compare each detected face with reference
        for (x, y, w, h) in faces:
            face_crop = grp_img[y:y+h, x:x+w]

            try:
                result = DeepFace.verify(
                    ref_img,
                    face_crop,
                    model_name="Facenet",
                    enforce_detection=False
                )

                distance = result["distance"]

                if distance < best_score:
                    best_score = distance
                    best_face = (x, y, w, h)

            except:
                continue

        if best_face is None or best_score > 0.6:
            raise HTTPException(400, "No match found")

        x, y, w, h = best_face

        # Draw arrow
        cx = x + w // 2
        arrow_start = (cx + int(w * 0.5), y - int(h * 1.2))
        arrow_end = (cx, y)

        cv2.arrowedLine(grp_img, arrow_start, arrow_end, (0, 0, 255), 3)

        # Save image
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, grp_img)

        # Correct public URL
        image_url = f"{request.base_url}images/{filename}"

        return JSONResponse({
            "success": True,
            "image_url": image_url,
            "distance": float(best_score)
        })

    except Exception as e:
        raise HTTPException(500, str(e))
