import cv2
import numpy as np
import os, csv, io, base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, flash, jsonify
from PIL import Image

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

BASE_DIR        = "/tmp/face_attendance"
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.template_filter('enumerate')
def jinja_enumerate(iterable):
    return list(enumerate(iterable))

def decode_bytes(img_bytes):
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def decode_b64(data):
    try:
        if "," in data: data = data.split(",", 1)[1]
        return decode_bytes(base64.b64decode(data))
    except: return None

def detect(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return f if len(f) > 0 else []

def crop_face(bgr, x, y, w, h, sz=150):
    return cv2.resize(bgr[y:y+h, x:x+w], (sz, sz))

def get_hist(bgr):
    h = []
    for c in range(3):
        ch = cv2.calcHist([bgr], [c], None, [64], [0, 256])
        h.extend(ch.flatten())
    a = np.array(h, dtype=np.float32)
    return a / (a.sum() + 1e-6)

def find_match(bgr, thr=0.82):
    faces = detect(bgr)
    if len(faces) == 0: return None, "no_face"
    x, y, w, h = faces[0]
    qh = get_hist(crop_face(bgr, x, y, w, h))
    best, score = None, 0.0
    for person in get_users():
        pdir = os.path.join(KNOWN_FACES_DIR, person)
        for f in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, f))
            if img is None: continue
            s = float(np.dot(qh, get_hist(img)))
            if s > score: score, best = s, person
    return (best, score) if score >= thr else (None, score)

def get_users():
    if not os.path.exists(KNOWN_FACES_DIR): return []
    return [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
    rows  = []
    done  = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE) as f: rows = list(csv.reader(f))
        for r in rows:
            if len(r) >= 3 and r[0] == name and r[1] == today: done = True; break
    if not done:
        rows.append([name, today, now])
        with open(ATTENDANCE_FILE, "w", newline="") as f: csv.writer(f).writerows(rows)
        return True
    return False

def get_att():
    if not os.path.exists(ATTENDANCE_FILE): return []
    with open(ATTENDANCE_FILE) as f: return list(csv.reader(f))

def get_frame(req):
    up = req.files.get("photo_file")
    if up and up.filename:
        try: return decode_bytes(up.read())
        except: pass
    d = req.form.get("image_data", "")
    return decode_b64(d) if d else None

@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    att   = get_att()
    return render_template("index.html", users=get_users(),
        today_attendance=[r for r in att if len(r) >= 2 and r[1] == today], today=today)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name  = request.form.get("name", "").strip().replace(" ", "_")
        if not name: flash("Enter a name.", "error"); return redirect(url_for("register"))
        frame = get_frame(request)
        if frame is None: flash("Could not read image.", "error"); return redirect(url_for("register"))
        faces = detect(frame)
        if len(faces) == 0: flash("No face detected. Use a clear front-facing photo.", "error"); return redirect(url_for("register"))
        x, y, w, h = faces[0]
        pdir  = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(pdir, exist_ok=True)
        count = len(os.listdir(pdir))
        cv2.imwrite(os.path.join(pdir, f"{name}_{count+1}.jpg"), crop_face(frame, x, y, w, h))
        flash(f"✅ {name.replace('_', ' ')} registered!", "success")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/take_attendance")
def take_attendance():
    return render_template("take_attendance.html")

@app.route("/process_attendance", methods=["POST"])
def process_attendance():
    frame = get_frame(request)
    if frame is None: return jsonify({"status": "error", "message": "Could not decode image."})
    if not get_users(): return jsonify({"status": "error", "message": "No registered faces."})
    name, score = find_match(frame)
    if score == "no_face": return jsonify({"status": "no_face", "message": "No face detected."})
    if name:
        marked = mark_attendance(name)
        return jsonify({"status": "success", "results": [{"name": name.replace("_", " "), "marked": marked, "confidence": round(float(score)*100, 1)}]})
    return jsonify({"status": "success", "results": [{"name": "Unknown", "marked": False, "confidence": 0}]})

@app.route("/attendance_records")
def attendance_records():
    return render_template("attendance_records.html", records=get_att())

@app.route("/download_csv")
def download_csv():
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Name", "Date", "Time"])
    w.writerows(get_att())
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=attendance.csv"})

@app.route("/delete_user/<n>", methods=["POST"])
def delete_user(name):
    import shutil
    pdir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(pdir): shutil.rmtree(pdir); flash(f"{name.replace('_', ' ')} deleted.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
