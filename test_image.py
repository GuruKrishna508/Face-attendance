"""
Test face_recognition WITHOUT webcam.
Just run: python test_image.py
"""
import cv2
import numpy as np
import face_recognition
import sys, os, urllib.request

print("=== Step 1: Library versions ===")
print(f"OpenCV : {cv2.__version__}")
print(f"NumPy  : {np.__version__}")
print(f"face_recognition: {face_recognition.__version__}")

print("\n=== Step 2: Download a real face photo for testing ===")
test_img_path = "test_face.jpg"
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/800px-President_Barack_Obama.jpg"

try:
    print("Downloading test image...")
    urllib.request.urlretrieve(url, test_img_path)
    print(f"Downloaded OK: {test_img_path}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Creating a blank image as fallback (no faces expected)...")
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, blank)

print("\n=== Step 3: Load with OpenCV ===")
bgr = cv2.imread(test_img_path)
if bgr is None:
    print("ERROR: cv2.imread returned None — image file is corrupt or missing!")
    sys.exit(1)
print(f"BGR: shape={bgr.shape} dtype={bgr.dtype}")

print("\n=== Step 4: Convert BGR -> RGB ===")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
print(f"RGB: shape={rgb.shape} dtype={rgb.dtype} contiguous={rgb.flags['C_CONTIGUOUS']}")

print("\n=== Step 5: face_recognition.face_locations ===")
try:
    locs = face_recognition.face_locations(rgb)
    print(f"✅ SUCCESS! Faces found: {len(locs)}")
except RuntimeError as e:
    print(f"❌ RuntimeError: {e}")
    print("\nDIAGNOSIS: dlib itself is broken. Reinstall it:")
    print("  pip uninstall dlib face-recognition -y")
    print("  pip install dlib-19.24.99-cp312-cp312-win_amd64.whl")
    print("  pip install face-recognition")
    sys.exit(1)

print("\n=== Step 6: Simulate browser upload (bytes -> decode -> face_locations) ===")
try:
    _, encoded = cv2.imencode('.jpg', bgr)
    img_bytes  = encoded.tobytes()
    nparr      = np.frombuffer(img_bytes, np.uint8)
    dec_bgr    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dec_rgb    = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    dec_rgb    = np.ascontiguousarray(dec_rgb, dtype=np.uint8)
    locs2      = face_recognition.face_locations(dec_rgb)
    print(f"✅ Browser-path works! Faces: {len(locs2)}")
except Exception as e:
    print(f"❌ Browser-path failed: {type(e).__name__}: {e}")

if os.path.exists(test_img_path):
    os.remove(test_img_path)

print("\n=== Done — paste the output above to Claude ===")