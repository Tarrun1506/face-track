# app.py — CCTV Person Finder / Suspect Tracker Backend
# Powered by InsightFace (SCRFD + ArcFace) — GPU-accelerated, ~8x faster than MTCNN

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os, cv2, base64, uuid, traceback, subprocess
import numpy as np
import imageio_ffmpeg
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── InsightFace Setup — detection + recognition only (skip landmarks/gender) ──
print('[INFO] Loading InsightFace models (SCRFD + ArcFace only) ...')
face_app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition'],  # Skip 3D/2D landmarks & gender/age
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))   # ctx_id=0 → GPU 0 (RTX 4050)

import onnxruntime as ort
_providers = ort.get_available_providers()
GPU_NAME   = 'NVIDIA GeForce RTX 4050 Laptop GPU (CUDA)' if 'CUDAExecutionProvider' in _providers else 'CPU'
print(f'[INFO] Running on: {GPU_NAME}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_ref_embedding(bgr_img: np.ndarray):
    """Detect face in reference photo and return ArcFace embedding."""
    faces = face_app.get(bgr_img)
    if not faces:
        return None
    # Use the largest (most prominent) face
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb  = best.embedding
    return emb / np.linalg.norm(emb)


def match_frame(bgr_frame: np.ndarray, ref_emb: np.ndarray, thresh: float = 0.40):
    """Return (bbox, confidence_pct) for best suspect match, else (None, None).
    Downscales large frames before detection for speed.
    """
    h, w  = bgr_frame.shape[:2]

    # Downscale to max 640px wide/tall for faster detection
    scale = 1.0
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        det_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))
    else:
        det_frame = bgr_frame

    faces = face_app.get(det_frame)
    if not faces:
        return None, None

    best_box, best_conf = None, 0.0
    for face in faces:
        emb  = face.embedding / np.linalg.norm(face.embedding)
        sim  = float(np.dot(ref_emb, emb))
        conf = sim * 100
        if sim > thresh and conf > best_conf:
            # Scale bbox back to original resolution
            best_box  = face.bbox / scale
            best_conf = conf

    return best_box, best_conf


def draw_box(frame: np.ndarray, bbox, conf: float) -> np.ndarray:
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
    GREEN  = (50, 255, 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    label  = f' SUSPECT IDENTIFIED  {conf:.1f}% '
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), GREEN, -1)
    cv2.putText(frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'gpu': GPU_NAME})


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    if 'photo' not in request.files or 'video' not in request.files:
        return jsonify({'error': 'Both photo and video are required'}), 400
    try:
        uid        = uuid.uuid4().hex[:8]
        photo_path = os.path.join(UPLOAD_DIR, f'{uid}_ref.jpg')
        vf         = request.files['video']
        ext        = os.path.splitext(vf.filename)[1] or '.mp4'
        video_path = os.path.join(UPLOAD_DIR, f'{uid}_video{ext}')
        output_path= os.path.join(OUTPUT_DIR,  f'{uid}_tracked.mp4')

        request.files['photo'].save(photo_path)
        vf.save(video_path)

        # Reference embedding
        ref_img = cv2.imread(photo_path)
        if ref_img is None:
            return jsonify({'error': 'Cannot read photo'}), 400
        ref_emb = get_ref_embedding(ref_img)
        if ref_emb is None:
            return jsonify({'error': 'No face detected in reference photo. Use a clear frontal photo.'}), 400

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video file'}), 400

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Always use mp4v (reliable cross-platform); we transcode to H.264 after
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        raw_path = output_path.replace('.mp4', '_raw.mp4')
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (W, H))

        SKIP       = 3        # process every 3rd frame (faster)
        CARRY      = SKIP + 1 # carry bbox this many frames
        detections = []
        thumbnails = []
        timeline   = []
        carry_left = 0
        last_box   = None
        last_conf  = 0.0
        first_det  = True
        frame_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts       = round(frame_idx / fps, 2)
            detected = False

            if frame_idx % SKIP == 0:
                box, conf = match_frame(frame, ref_emb)
                if box is not None:
                    detected   = True
                    last_box   = box
                    last_conf  = conf
                    carry_left = CARRY
                    detections.append({'frame': frame_idx, 'timestamp': ts,
                                       'confidence': round(conf, 1)})
                    # Collect thumbnail
                    if len(thumbnails) < 8:
                        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                        x2, y2 = min(W, int(box[2])), min(H, int(box[3]))
                        crop   = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            thumb = cv2.resize(crop, (120, 120))
                            _, buf = cv2.imencode('.jpg', thumb,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
                            thumbnails.append({'data': base64.b64encode(buf).decode(),
                                               'timestamp': ts,
                                               'confidence': round(conf, 1)})
                else:
                    if carry_left > 0:
                        carry_left -= 1
                        detected = last_box is not None
                    else:
                        last_box = None

                timeline.append({'frame': frame_idx, 'timestamp': ts,
                                 'detected': detected or (carry_left > 0 and last_box is not None)})

            # Draw
            if last_box is not None:
                if first_det and detections:
                    first_det   = False
                    red_overlay = np.full_like(frame, (0, 0, 180))
                    frame       = cv2.addWeighted(frame, 0.6, red_overlay, 0.4, 0)
                draw_box(frame, last_box, last_conf)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # ── Transcode raw mp4v → H.264 for browser playback ──────────────────
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe,
            '-y',                          # overwrite output
            '-i', raw_path,               # input: raw mp4v
            '-c:v', 'libx264',            # H.264 codec
            '-pix_fmt', 'yuv420p',        # browser-compatible pixel format
            '-crf', '23',                 # quality (lower = better, 18-28 is fine)
            '-movflags', '+faststart',    # enable streaming from start
            output_path
        ], capture_output=True)
        if os.path.exists(raw_path):
            os.remove(raw_path)

        n        = len(detections)
        avg_conf = round(sum(d['confidence'] for d in detections) / n, 1) if n else 0
        thumbnails.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'match_found': n > 0,
            'stats': {
                'total_frames':     total_frames,
                'frames_analyzed':  total_frames // SKIP,
                'total_detections': n,
                'avg_confidence':   avg_conf,
                'first_seen':       detections[0]['timestamp'] if n else None,
                'last_seen':        detections[-1]['timestamp'] if n else None,
                'video_duration':   round(total_frames / fps, 1),
                'gpu':              GPU_NAME
            },
            'timeline':   timeline,
            'thumbnails': thumbnails[:5],
            'video_url':  f'/api/video/{uid}_tracked.mp4'
        })

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal processing error. Check server logs.'}), 500


@app.route('/api/video/<filename>')
def serve_video(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Video not found'}), 404

    file_size = os.path.getsize(path)
    range_header = request.headers.get('Range', None)

    if range_header:
        # Parse Range header for video seeking
        byte_start = 0
        byte_end   = file_size - 1
        match = range_header.replace('bytes=', '').split('-')
        if match[0]:  byte_start = int(match[0])
        if match[1]:  byte_end   = int(match[1])
        length = byte_end - byte_start + 1

        with open(path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(length)

        resp = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
        resp.headers['Content-Range']  = f'bytes {byte_start}-{byte_end}/{file_size}'
        resp.headers['Accept-Ranges']  = 'bytes'
        resp.headers['Content-Length'] = str(length)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    else:
        resp = send_file(path, mimetype='video/mp4')
        resp.headers['Accept-Ranges']  = 'bytes'
        resp.headers['Content-Length'] = str(file_size)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
