import cv2
import numpy as np
import os
import csv
import io
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, flash, jsonify
import base64
from PIL import Image
from deepface import DeepFace
import json

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

BASE_DIR        = "/tmp/face_attendance"
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

@app.template_filter('enumerate')
def jinja_enumerate(iterable):
    return list(enumerate(iterable))

def pil_to_rgb(pil_img):
    pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)

def decode_image_from_bytes(img_bytes):
    pil_img = Image.open(io.BytesIO(img_bytes))
    rgb = pil_to_rgb(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, rgb

def decode_image_from_base64(image_data):
    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        return decode_image_from_bytes(img_bytes)
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None

def get_registered_users():
    if not os.path.exists(KNOWN_FACES_DIR):
        return []
    return [d for d in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]

def find_match(img_path):
    """Compare img_path against all registered faces. Returns name or None."""
    for person_name in get_registered_users():
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        for img_file in os.listdir(person_dir):
            known_path = os.path.join(person_dir, img_file)
            try:
                result = DeepFace.verify(
                    img1_path=img_path,
                    img2_path=known_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )
                if result["verified"]:
                    return person_name
            except Exception as e:
                print(f"[WARN] verify error: {e}")
    return None

def mark_attendance(name):
    today    = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    records  = []
    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            records = list(csv.reader(f))
            for row in records:
                if len(row) >= 3 and row[0] == name and row[1] == today:
                    already_marked = True
                    break
    if not already_marked:
        records.append([name, today, now_time])
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerows(records)
        return True
    return False

def get_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, "r") as f:
        return list(csv.reader(f))

@app.route("/")
def index():
    users = get_registered_users()
    attendance = get_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    today_attendance = [r for r in attendance if len(r) >= 2 and r[1] == today]
    return render_template("index.html", users=users,
                           today_attendance=today_attendance, today=today)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip().replace(" ", "_")
        if not name:
            flash("Please enter a valid name.", "error")
            return redirect(url_for("register"))

        frame = None
        uploaded = request.files.get("photo_file")
        if uploaded and uploaded.filename:
            try:
                frame, _ = decode_image_from_bytes(uploaded.read())
            except Exception as e:
                print(f"[ERROR] {e}")

        if frame is None:
            image_data = request.form.get("image_data", "")
            if image_data:
                frame, _ = decode_image_from_base64(image_data)

        if frame is None:
            flash("Could not decode image. Please try again.", "error")
            return redirect(url_for("register"))

        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        count    = len(os.listdir(person_dir))
        img_path = os.path.join(person_dir, f"{name}_{count+1}.jpg")
        cv2.imwrite(img_path, frame)

        # Verify a face exists in the image
        try:
            DeepFace.extract_faces(img_path=img_path, enforce_detection=True)
        except Exception:
            os.remove(img_path)
            flash("No face detected in photo. Please use a clearer image.", "error")
            return redirect(url_for("register"))

        flash(f"✅ {name.replace('_', ' ')} registered successfully!", "success")
        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/take_attendance")
def take_attendance():
    return render_template("take_attendance.html")

@app.route("/process_attendance", methods=["POST"])
def process_attendance():
    frame = None
    uploaded = request.files.get("photo_file")
    if uploaded and uploaded.filename:
        try:
            frame, _ = decode_image_from_bytes(uploaded.read())
        except Exception as e:
            print(f"[ERROR] {e}")

    if frame is None:
        image_data = request.form.get("image_data", "")
        if image_data:
            frame, _ = decode_image_from_base64(image_data)

    if frame is None:
        return jsonify({"status": "error", "message": "Could not decode image."})

    if not get_registered_users():
        return jsonify({"status": "error", "message": "No registered faces found."})

    # Save temp image for DeepFace
    tmp_path = os.path.join(BASE_DIR, "tmp_scan.jpg")
    cv2.imwrite(tmp_path, frame)

    matched_name = find_match(tmp_path)

    try:
        os.remove(tmp_path)
    except:
        pass

    if matched_name:
        marked = mark_attendance(matched_name)
        return jsonify({"status": "success", "results": [{
            "name":    matched_name.replace("_", " "),
            "marked":  marked,
            "confidence": 0
        }]})
    else:
        return jsonify({"status": "success", "results": [{
            "name": "Unknown", "marked": False, "confidence": 0
        }]})

@app.route("/attendance_records")
def attendance_records():
    return render_template("attendance_records.html", records=get_attendance())

@app.route("/download_csv")
def download_csv():
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["Name", "Date", "Time"])
    w.writerows(get_attendance())
    output.seek(0)
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=attendance.csv"})

@app.route("/delete_user/<n>", methods=["POST"])
def delete_user(name):
    import shutil
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        flash(f"User {name.replace('_', ' ')} deleted.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
