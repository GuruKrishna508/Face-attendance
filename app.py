import cv2
import face_recognition
import numpy as np
import os
import csv
import pickle
import io
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, flash, jsonify
import base64
from PIL import Image, ExifTags

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

@app.template_filter('enumerate')
def jinja_enumerate(iterable):
    return list(enumerate(iterable))


# ═══════════════════════════════════════════════════════════════════
#  THE ONLY CORRECT WAY TO GET A DLIB-COMPATIBLE RGB ARRAY
#  Strategy: decode with OpenCV (which always gives uint8 BGR),
#  then flip to RGB. This bypasses all PIL format quirks entirely.
# ═══════════════════════════════════════════════════════════════════

def bytes_to_rgb_via_cv2(img_bytes):
    """
    Decode image bytes using OpenCV (NOT PIL).
    OpenCV always returns uint8 BGR. We flip to RGB for face_recognition.
    This is the most reliable method — avoids all PIL mode issues.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # always uint8 BGR 3-ch

    if bgr is None:
        return None, None

    # Fix rotation for phone photos using PIL EXIF (only metadata, not pixel data)
    try:
        from PIL import Image, ExifTags
        pil_tmp = Image.open(io.BytesIO(img_bytes))
        exif = pil_tmp._getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                angle = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE,
                         8: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(exif[orientation_key])
                if angle is not None:
                    bgr = cv2.rotate(bgr, angle)
    except Exception:
        pass  # EXIF fix is optional — never crash because of it

    # BGR -> RGB, force contiguous uint8
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    print(f"[DEBUG] bytes_to_rgb_via_cv2: bgr={bgr.shape} rgb={rgb.shape} "
          f"dtype={rgb.dtype} contiguous={rgb.flags['C_CONTIGUOUS']}")
    return bgr, rgb


def decode_image_from_bytes(img_bytes):
    return bytes_to_rgb_via_cv2(img_bytes)


def decode_image_from_base64(image_data):
    """Decode base64 data-URL from browser webcam."""
    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        return bytes_to_rgb_via_cv2(img_bytes)
    except Exception as e:
        print(f"[ERROR] decode_image_from_base64: {e}")
        return None, None


def safe_face_locations(rgb):
    """Hard pre-flight check then call face_recognition."""
    assert rgb is not None,              "rgb is None"
    assert rgb.dtype == np.uint8,        f"dtype={rgb.dtype}, need uint8"
    assert rgb.ndim == 3,                f"ndim={rgb.ndim}, need 3"
    assert rgb.shape[2] == 3,           f"channels={rgb.shape[2]}, need 3"
    rgb = np.ascontiguousarray(rgb)      # guarantee contiguous

    print(f"[DEBUG] safe_face_locations -> shape={rgb.shape} "
          f"dtype={rgb.dtype} contiguous={rgb.flags['C_CONTIGUOUS']}")
    return face_recognition.face_locations(rgb)


# ── ENCODING / ATTENDANCE UTILITIES ─────────────────────────────────────────

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load encodings, rebuilding: {e}")
    return {"names": [], "encodings": []}


def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def encode_faces():
    data = {"names": [], "encodings": []}
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                encs  = face_recognition.face_encodings(image)
                if encs:
                    data["encodings"].append(encs[0])
                    data["names"].append(person_name)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")
    save_encodings(data)
    return data


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


def get_registered_users():
    if not os.path.exists(KNOWN_FACES_DIR):
        return []
    return [d for d in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]


# ── ROUTES ───────────────────────────────────────────────────────────────────

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
        rgb   = None

        # Method 1: file upload
        uploaded = request.files.get("photo_file")
        if uploaded and uploaded.filename:
            frame, rgb = decode_image_from_bytes(uploaded.read())
            if frame is not None:
                print("[DEBUG] Image source: file upload")

        # Method 2: webcam base64
        if rgb is None:
            image_data = request.form.get("image_data", "")
            if image_data:
                print(f"[DEBUG] Image source: base64 webcam")
                frame, rgb = decode_image_from_base64(image_data)

        if frame is None or rgb is None:
            flash("Could not decode image. Please try a different photo.", "error")
            return redirect(url_for("register"))

        try:
            locs = safe_face_locations(rgb)
        except Exception as e:
            print(f"[ERROR] face_locations: {e}")
            flash(f"Face detection error: {e}", "error")
            return redirect(url_for("register"))

        if not locs:
            flash("No face detected. Try better lighting or a clearer photo.", "error")
            return redirect(url_for("register"))

        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        count     = len(os.listdir(person_dir))
        save_path = os.path.join(person_dir, f"{name}_{count + 1}.jpg")
        cv2.imwrite(save_path, frame)
        encode_faces()
        flash(f"✅ {name.replace('_', ' ')} registered successfully!", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/take_attendance")
def take_attendance():
    return render_template("take_attendance.html")


@app.route("/process_attendance", methods=["POST"])
def process_attendance():
    frame = None
    rgb   = None

    uploaded = request.files.get("photo_file")
    if uploaded and uploaded.filename:
        frame, rgb = decode_image_from_bytes(uploaded.read())

    if rgb is None:
        image_data = request.form.get("image_data", "")
        if image_data:
            frame, rgb = decode_image_from_base64(image_data)

    if frame is None or rgb is None:
        return jsonify({"status": "error", "message": "Could not decode image."})

    data = load_encodings()
    if not data["encodings"] or not data["names"]:
        return jsonify({"status": "error",
                        "message": "No registered faces found. Please register someone first."})

    if len(data["encodings"]) != len(data["names"]):
        data = encode_faces()

    try:
        face_locs = safe_face_locations(rgb)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Face detection error: {e}"})

    if not face_locs:
        return jsonify({"status": "no_face", "message": "No face detected in the image."})

    face_encs = face_recognition.face_encodings(rgb, face_locs)
    if not face_encs:
        return jsonify({"status": "no_face",
                        "message": "Face found but could not be encoded. Try a clearer photo."})

    results = []
    for enc, loc in zip(face_encs, face_locs):
        matches   = face_recognition.compare_faces(data["encodings"], enc, tolerance=0.5)
        distances = face_recognition.face_distance(data["encodings"], enc)

        if len(distances) == 0:
            results.append({"name": "Unknown", "marked": False, "confidence": 0})
            continue

        best_idx = int(np.argmin(distances))
        if matches[best_idx]:
            n      = data["names"][best_idx]
            marked = mark_attendance(n)
            results.append({
                "name":       n.replace("_", " "),
                "marked":     marked,
                "confidence": round((1 - distances[best_idx]) * 100, 1)
            })
        else:
            results.append({"name": "Unknown", "marked": False, "confidence": 0})

    return jsonify({"status": "success", "results": results})


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


@app.route("/delete_user/<name>", methods=["POST"])
def delete_user(name):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        encode_faces()
        flash(f"✅ User {name.replace('_', ' ')} deleted.", "success")
    else:
        flash(f"User '{name}' not found.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)