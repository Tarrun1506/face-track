# app.py — CCTV Person Finder / Suspect Tracker
# GPU Pipeline: InsightFace (SCRFD+ArcFace) + NVDEC input + NVENC output

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os, cv2, base64, uuid, traceback, subprocess, time, threading
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

# ── InsightFace — detection + recognition only ────────────────────────────────
print('[INFO] Loading InsightFace (SCRFD + ArcFace only) ...')
face_app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ── Live Tracking State ──────────────────────────────────────────────────────
# Stores the embedding of the suspect for live webcam tracking
live_ref_emb = None 

import onnxruntime as ort
_providers = ort.get_available_providers()
GPU_NAME   = 'NVIDIA GeForce RTX 4050 Laptop GPU (CUDA)' if 'CUDAExecutionProvider' in _providers else 'CPU'
HAS_NVENC  = 'CUDAExecutionProvider' in _providers   # RTX 4050 has NVENC/NVDEC
print(f'[INFO] Running on: {GPU_NAME}  |  NVENC available: {HAS_NVENC}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def cleanup_old_files():
    """Delete files in uploads/ and outputs/ older than 1 hour."""
    now = time.time()
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and (now - os.path.getmtime(p)) > 3600:
                try: os.remove(p)
                except: pass

def get_ref_embedding(bgr_img: np.ndarray):
    faces = face_app.get(bgr_img)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb  = best.embedding
    return emb / np.linalg.norm(emb)

THRESH = 0.28          # minimum similarity to count as a match
SIM_MAX = 0.85         # expected upper bound for ArcFace similarity

def sim_to_conf(sim: float) -> float:
    """Remap cosine similarity [THRESH, SIM_MAX] → display confidence [70, 99]%."""
    ratio = (sim - THRESH) / (SIM_MAX - THRESH)
    return round(min(99.0, max(70.0, 70.0 + ratio * 29.0)), 1)

def match_frame(bgr_frame: np.ndarray, ref_emb: np.ndarray):
    """Return (bbox, display_confidence%) for best suspect match, else (None, None)."""
    faces = face_app.get(bgr_frame)
    if not faces:
        return None, None
    best_box, best_conf = None, 0.0
    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        sim = float(np.dot(ref_emb, emb))
        if sim > THRESH:
            conf = sim_to_conf(sim)
            if conf > best_conf:
                best_box, best_conf = face.bbox, conf
    return best_box, best_conf

def draw_box(frame: np.ndarray, bbox, conf: float) -> np.ndarray:
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
    GREEN  = (50, 255, 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    label  = f' SUSPECT IDENTIFIED  {conf:.1f}% '
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), GREEN, -1)
    cv2.putText(frame, label, (x1, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1, cv2.LINE_AA)
    return frame

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'gpu': GPU_NAME})

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    cleanup_old_files()
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
            return jsonify({'error': 'No face detected in reference photo.'}), 400

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # ── Get video metadata via OpenCV ─────────────────────────────────
        _cap = cv2.VideoCapture(video_path)
        if not _cap.isOpened():
            return jsonify({'error': 'Cannot open video file'}), 400
        W   = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
        _cap.release()

        # Output resolution: max 720p
        MAX_OUT   = 720
        scale_out = min(1.0, MAX_OUT / max(W, H))
        OW = int(W * scale_out) & ~1
        OH = int(H * scale_out) & ~1
        print(f'[INFO] Input: {W}x{H}@{fps:.0f}fps → Output: {OW}x{OH}')

        # ── GPU-decode input ───────────────────────────────────────────────
        decode_cmd = [ffmpeg_exe]
        if HAS_NVENC:
            decode_cmd += ['-hwaccel', 'cuda']
        decode_cmd += [
            '-i', video_path,
            '-vf', f'scale={OW}:{OH}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'
        ]
        reader = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=10**8)

        # ── GPU-encode output ──────────────────────────────────────────────
        encoder    = 'h264_nvenc' if HAS_NVENC else 'libx264'
        enc_preset = 'p1'        if HAS_NVENC else 'ultrafast'
        writer_proc = subprocess.Popen([
            ffmpeg_exe, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{OW}x{OH}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', encoder, '-preset', enc_preset,
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            output_path
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(f'[INFO] Encoder: {encoder}  Decoder: {"NVDEC" if HAS_NVENC else "CPU"}')

        # ── Detect-Then-Track constants ────────────────────────────────────
        SCAN_SKIP    = 6    # run InsightFace on full frame every N frames while scanning
        ROI_SKIP     = 2    # run InsightFace on ROI crop every N frames while tracking
        ROI_PAD      = 1.5  # multiply box dims by this to get the search region
        MISS_LIMIT   = 15   # frames without ROI hit before falling back to full scan

        # ── State machine ──────────────────────────────────────────────────
        MODE_SCAN  = 'scan'
        MODE_TRACK = 'track'
        mode       = MODE_SCAN
        last_box   = None   # [x1,y1,x2,y2] in full-frame coords
        last_conf  = 0.0
        miss_count = 0      # consecutive frames where ROI search found nothing

        frame_size = OW * OH * 3
        detections = []
        thumbnails = []
        timeline   = []
        first_det  = True
        frame_idx  = 0

        def padded_roi(box, pad, W, H):
            """Return (rx1,ry1,rx2,ry2) — a padded search region clamped to frame."""
            x1, y1, x2, y2 = box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            hw = (x2-x1)*pad/2
            hh = (y2-y1)*pad/2
            return (max(0, int(cx-hw)), max(0, int(cy-hh)),
                    min(W, int(cx+hw)), min(H, int(cy+hh)))

        def save_thumbnail(frame, box):
            x1,y1,x2,y2 = max(0,int(box[0])),max(0,int(box[1])),min(OW,int(box[2])),min(OH,int(box[3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            thumb = cv2.resize(crop, (120,120))
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode()

        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame    = np.frombuffer(raw, dtype=np.uint8).reshape((OH, OW, 3)).copy()
            ts       = round(frame_idx / fps, 2)
            detected = False

            # ── PHASE 1: SCAN — full-frame face recognition ────────────────
            if mode == MODE_SCAN:
                if frame_idx % SCAN_SKIP == 0:
                    t0 = time.time()
                    box, conf = match_frame(frame, ref_emb)
                    print(f'[SCAN  #{frame_idx}] {(time.time()-t0)*1000:.0f}ms | found={box is not None}')

                    if box is not None:
                        last_box   = box.tolist() if hasattr(box, 'tolist') else list(box)
                        last_conf  = conf
                        detected   = True
                        miss_count = 0
                        mode       = MODE_TRACK
                        print(f'[INFO] Target ACQUIRED @frame {frame_idx} — entering ROI-TRACK mode')

                        detections.append({'frame': frame_idx, 'timestamp': ts,
                                           'confidence': round(conf, 1)})
                        if len(thumbnails) < 8:
                            d = save_thumbnail(frame, last_box)
                            if d:
                                thumbnails.append({'data': d, 'timestamp': ts,
                                                   'confidence': round(conf, 1)})

            # ── PHASE 2: TRACK — ROI-crop face recognition ─────────────────
            else:
                if frame_idx % ROI_SKIP == 0:
                    rx1, ry1, rx2, ry2 = padded_roi(last_box, ROI_PAD, OW, OH)
                    roi = frame[ry1:ry2, rx1:rx2]

                    box_in_roi, conf = match_frame(roi, ref_emb)
                    print(f'[TRACK #{frame_idx}] ROI={rx1},{ry1}-{rx2},{ry2} | found={box_in_roi is not None}')

                    if box_in_roi is not None:
                        # Translate box from ROI coords → full-frame coords
                        bx1,by1,bx2,by2 = box_in_roi
                        full_box = [rx1+bx1, ry1+by1, rx1+bx2, ry1+by2]
                        last_box  = full_box
                        last_conf = conf
                        detected  = True
                        miss_count = 0

                        detections.append({'frame': frame_idx, 'timestamp': ts,
                                           'confidence': round(conf, 1)})
                        if len(thumbnails) < 8:
                            d = save_thumbnail(frame, last_box)
                            if d:
                                thumbnails.append({'data': d, 'timestamp': ts,
                                                   'confidence': round(conf, 1)})
                    else:
                        # ROI miss — keep box visible but count miss
                        miss_count += 1
                        if miss_count >= MISS_LIMIT:
                            print(f'[INFO] Target LOST after {miss_count} misses — reverting to SCAN')
                            mode       = MODE_SCAN
                            last_box   = None
                            miss_count = 0
                        else:
                            # Still show last known box while searching
                            detected = True
                else:
                    # Non-checked frame: carry previous box if we have one
                    if last_box is not None:
                        detected = True

            timeline.append({'frame': frame_idx, 'timestamp': ts, 'detected': detected})

            # Draw bounding box on frame
            if detected and last_box is not None:
                if first_det and detections:
                    first_det   = False
                    red_overlay = np.full_like(frame, (0, 0, 180))
                    frame       = cv2.addWeighted(frame, 0.6, red_overlay, 0.4, 0)
                draw_box(frame, last_box, last_conf)

            writer_proc.stdin.write(frame.tobytes())
            frame_idx += 1

        reader.stdout.close()
        reader.wait()
        writer_proc.stdin.close()
        writer_proc.wait()

        n        = len(detections)
        avg_conf = round(sum(d['confidence'] for d in detections) / n, 1) if n else 0
        thumbnails.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'match_found': n > 0,
            'stats': {
                'total_frames':     frame_idx,
                'frames_analyzed':  sum(1 for t in timeline if t['detected'] or frame_idx % SCAN_SKIP == 0),
                'total_detections': n,
                'avg_confidence':   avg_conf,
                'first_seen':       detections[0]['timestamp'] if n else None,
                'last_seen':        detections[-1]['timestamp'] if n else None,
                'video_duration':   round(frame_idx / fps, 1),
                'gpu':              GPU_NAME
            },
            'timeline':   timeline,
            'thumbnails': thumbnails[:5],
            'video_url':  f'/api/video/{uid}_tracked.mp4'
        })

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal processing error. Check server logs.'}), 500


@app.route('/api/prepare_live', methods=['POST'])
def prepare_live():
    global live_ref_emb
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo provided'}), 400
    try:
        f = request.files['photo']
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error': 'Cannot read photo'}), 400
        live_ref_emb = get_ref_embedding(img)
        if live_ref_emb is None: return jsonify({'error': 'No face detected in photo'}), 400
        return jsonify({'status': 'ready'})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal error'}), 500

@app.route('/api/track_frame', methods=['POST'])
def track_frame():
    global live_ref_emb
    if live_ref_emb is None:
        return jsonify({'error': 'Reference not set'}), 400
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        # Decode base64 image from frontend
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({'error': 'Decode failed'}), 400

        # Fast match (no drawing, just return box and confidence)
        box, conf = match_frame(frame, live_ref_emb)
        if box is not None:
            return jsonify({
                'found': True,
                'box': box.tolist(),
                'confidence': conf
            })
        return jsonify({'found': False})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal error'}), 500

@app.route('/api/video/<filename>')
def serve_video(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Video not found'}), 404
    file_size    = os.path.getsize(path)
    range_header = request.headers.get('Range')
    if range_header:
        byte_start, byte_end = 0, file_size - 1
        match = range_header.replace('bytes=', '').split('-')
        if match[0]: byte_start = int(match[0])
        if match[1]: byte_end   = int(match[1])
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
    resp = send_file(path, mimetype='video/mp4')
    resp.headers['Accept-Ranges']  = 'bytes'
    resp.headers['Content-Length'] = str(file_size)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
