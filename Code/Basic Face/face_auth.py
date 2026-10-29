"""
face_auth.py - IR + RGB face enrollment and recognition for Windows Hello cameras.

Commands:
    python face_auth.py cameras          # list IR sources + test RGB indices
    python face_auth.py enroll <name>    # scan a face into the database
    python face_auth.py list             # show enrolled people
    python face_auth.py delete <name>    # remove a person
    python face_auth.py recognize        # live recognition with match probability

Options:
    --rgb N        OpenCV index of the normal webcam (default 0, -1 to disable)
    --samples N    frames to capture when enrolling (default 20)

Pipeline:
    IR frames  -> WinRT MediaFrameReader (the only way to reach the Hello sensor)
    RGB frames -> cv2.VideoCapture
    Detection  -> OpenCV YuNet       Recognition -> OpenCV SFace (128-D embeddings)
    Score      -> cosine similarity -> logistic curve -> "probability"
"""

import argparse
import asyncio
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Paths and tunables
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DB_DIR = HERE / "faces_db"

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")

DET_SCORE_THRESHOLD = 0.6      # YuNet confidence to accept a detection
SFACE_COSINE_THRESHOLD = 0.363 # OpenCV's suggested "same person" cutoff for SFace
PROB_STEEPNESS = 15.0          # how sharply probability rises around the threshold
SMOOTHING = 0.3                # EMA weight for new frames (0..1, higher = jumpier)
IR_WEIGHT = 0.6                # IR vs RGB weighting when both see a face
MIN_IR_FACE_CONTRAST = 18.0    # std-dev of IR face pixels; flat/black = likely screen/photo


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    paths = {}
    for key, url in (("yunet", YUNET_URL), ("sface", SFACE_URL)):
        dest = MODEL_DIR / url.rsplit("/", 1)[-1]
        if not dest.exists() or dest.stat().st_size < 10_000:
            print(f"Downloading {dest.name} ...")
            urllib.request.urlretrieve(url, dest)
        paths[key] = str(dest)
    return paths


class FaceEngine:
    def __init__(self):
        p = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(p["yunet"], "", (320, 320),
                                                  DET_SCORE_THRESHOLD, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(p["sface"], "")

    def detect(self, bgr):
        """Return the largest face row (YuNet format) or None."""
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def embed(self, bgr, face):
        aligned = self.recognizer.alignCrop(bgr, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)


def to_bgr(gray_or_bgr):
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr


def similarity_to_probability(sim):
    """Map cosine similarity to 0..1. Not a true probability - tune on your own data."""
    return 1.0 / (1.0 + math.exp(-PROB_STEEPNESS * (sim - SFACE_COSINE_THRESHOLD)))


# --------------------------------------------------------------------------
# IR camera via WinRT
# --------------------------------------------------------------------------
class IRCamera:
    """Reads the Windows Hello infrared stream through MediaFrameReader."""

    def __init__(self):
        self.capture = None
        self.reader = None
        self.prev = None

    @staticmethod
    async def list_sources():
        from winrt.windows.media.capture.frames import MediaFrameSourceGroup
        groups = await MediaFrameSourceGroup.find_all_async()
        out = []
        for g in groups:
            for info in g.source_infos:
                out.append((g, info))
        return out

    async def open(self):
        from winrt.windows.media.capture import (
            MediaCapture, MediaCaptureInitializationSettings, MediaCaptureMemoryPreference,
            MediaCaptureSharingMode, StreamingCaptureMode)
        from winrt.windows.media.capture.frames import (
            MediaFrameReaderAcquisitionMode, MediaFrameReaderStartStatus, MediaFrameSourceKind)

        target = None
        for g, info in await self.list_sources():
            if info.source_kind == MediaFrameSourceKind.INFRARED:
                target = (g, info)
                break
        if target is None:
            raise RuntimeError("No infrared frame source found. Run `cameras` to see what's available.")
        group, info = target

        last_err = None
        # Exclusive gives us control; shared works if Windows Hello already holds the camera.
        for mode in (MediaCaptureSharingMode.EXCLUSIVE_CONTROL, MediaCaptureSharingMode.SHARED_READ_ONLY):
            try:
                settings = MediaCaptureInitializationSettings()
                settings.source_group = group
                settings.sharing_mode = mode
                settings.memory_preference = MediaCaptureMemoryPreference.CPU
                settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
                cap = MediaCapture()
                await cap.initialize_async(settings)
                source = cap.frame_sources.lookup(info.id)
                reader = await cap.create_frame_reader_async(source)
                reader.acquisition_mode = MediaFrameReaderAcquisitionMode.REALTIME
                status = await reader.start_async()
                if status != MediaFrameReaderStartStatus.SUCCESS:
                    raise RuntimeError(f"Reader start failed: {status}")
                self.capture, self.reader = cap, reader
                print(f"IR camera opened: {group.display_name} ({mode.name})")
                return
            except Exception as e:  # try next mode
                last_err = e
        raise RuntimeError(f"Could not open IR camera: {last_err}")

    def _bitmap_to_gray(self, sb):
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import Buffer

        fmt = sb.bitmap_pixel_format
        channels = 1
        if fmt != BitmapPixelFormat.GRAY8:
            try:
                sb = SoftwareBitmap.convert(sb, BitmapPixelFormat.GRAY8)
            except Exception:
                sb = SoftwareBitmap.convert(sb, BitmapPixelFormat.BGRA8)
                channels = 4

        w, h = sb.pixel_width, sb.pixel_height
        buf = Buffer(w * h * 4 + 4096)          # generous: rows can be padded
        sb.copy_to_buffer(buf)
        try:
            raw = np.frombuffer(memoryview(buf), dtype=np.uint8)
        except TypeError:
            raw = np.frombuffer(bytes(buf), dtype=np.uint8)
        raw = raw[: buf.length]
        stride = buf.length // h
        img = raw[: stride * h].reshape(h, stride)[:, : w * channels].copy()
        if channels == 4:
            img = cv2.cvtColor(img.reshape(h, w, 4), cv2.COLOR_BGRA2GRAY)
        return img

    def read(self):
        """Latest IR frame as grayscale uint8, or None if no new frame yet."""
        frame = self.reader.try_acquire_latest_frame()
        if frame is None:
            return None
        try:
            vmf = frame.video_media_frame
            if vmf is None or vmf.software_bitmap is None:
                return None
            img = self._bitmap_to_gray(vmf.software_bitmap)
        finally:
            frame.close()
        # Many Hello cameras alternate emitter-on / emitter-off frames.
        # Keep whichever of the last two is brighter (the illuminated one).
        out = img
        if self.prev is not None and self.prev.shape == img.shape and self.prev.mean() > img.mean():
            out = self.prev
        self.prev = img
        return out

    async def close(self):
        if self.reader is not None:
            try:
                await self.reader.stop_async()
            except Exception:
                pass
        if self.capture is not None:
            self.capture.close()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def db_load():
    DB_DIR.mkdir(exist_ok=True)
    db = {}
    for f in DB_DIR.glob("*.npz"):
        d = np.load(f)
        db[f.stem] = {"ir": d["ir"], "rgb": d["rgb"]}
    return db


def db_save(name, ir_feats, rgb_feats):
    DB_DIR.mkdir(exist_ok=True)
    np.savez(DB_DIR / f"{name}.npz",
             ir=np.array(ir_feats, dtype=np.float32).reshape(-1, 128),
             rgb=np.array(rgb_feats, dtype=np.float32).reshape(-1, 128))


def best_similarity(feat, stored):
    """Mean of the top-3 cosine similarities against a person's stored embeddings."""
    if feat is None or stored is None or len(stored) == 0:
        return None
    sims = stored @ feat
    k = min(3, len(sims))
    return float(np.sort(sims)[-k:].mean())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def draw_face(img, face, color, label=None):
    x, y, w, h = map(int, face[:4])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(img, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def ir_face_contrast(gray, face):
    x, y, w, h = map(int, face[:4])
    crop = gray[max(0, y): y + h, max(0, x): x + w]
    return float(crop.std()) if crop.size else 0.0


def open_rgb(index):
    if index < 0:
        return None
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Warning: RGB camera {index} not available, running IR-only.")
        return None
    return cap


def show(windows):
    for name, img in windows.items():
        if img is not None:
            cv2.imshow(name, img)
    return cv2.waitKey(1) & 0xFF


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
async def cmd_cameras(args):
    print("WinRT frame sources:")
    try:
        for g, info in await IRCamera.list_sources():
            print(f"  [{g.display_name}]  kind={info.source_kind.name}  id={info.id}")
    except ImportError:
        print("  winrt packages not installed (see requirements.txt)")
    print("\nOpenCV indices:")
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            print(f"  {i}: {frame.shape[1]}x{frame.shape[0]}  mean brightness {frame.mean():.0f}")
        cap.release()


async def cmd_enroll(args):
    engine = FaceEngine()
    ir = IRCamera()
    await ir.open()
    rgb = open_rgb(args.rgb)
    ir_feats, rgb_feats = [], []
    last_grab = 0.0
    print(f"Enrolling '{args.name}'. Look at the camera, move your head slightly. Q to abort.")
    try:
        while len(ir_feats) < args.samples:
            ir_img = ir.read()
            rgb_img = None
            if rgb is not None:
                ok, rgb_img = rgb.read()
                rgb_img = rgb_img if ok else None
            if ir_img is None:
                await asyncio.sleep(0.005)
                continue

            ir_bgr = to_bgr(ir_img)
            ir_face = engine.detect(ir_bgr)
            rgb_face = engine.detect(rgb_img) if rgb_img is not None else None

            now = time.time()
            if ir_face is not None and now - last_grab > 0.3:
                ir_feats.append(engine.embed(ir_bgr, ir_face))
                if rgb_face is not None:
                    rgb_feats.append(engine.embed(rgb_img, rgb_face))
                last_grab = now

            msg = f"{len(ir_feats)}/{args.samples}"
            if ir_face is not None:
                draw_face(ir_bgr, ir_face, (0, 255, 0), msg)
            cv2.putText(ir_bgr, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if rgb_img is not None and rgb_face is not None:
                draw_face(rgb_img, rgb_face, (0, 255, 0))
            if show({"IR": ir_bgr, "RGB": rgb_img}) == ord("q"):
                print("Aborted.")
                return
    finally:
        await ir.close()
        if rgb is not None:
            rgb.release()
        cv2.destroyAllWindows()

    db_save(args.name, ir_feats, rgb_feats)
    print(f"Saved {len(ir_feats)} IR and {len(rgb_feats)} RGB samples for '{args.name}'.")


async def cmd_recognize(args):
    db = db_load()
    if not db:
        print("Database is empty. Enroll someone first.")
        return
    engine = FaceEngine()
    ir = IRCamera()
    await ir.open()
    rgb = open_rgb(args.rgb)
    smoothed = {name: 0.0 for name in db}
    print("Recognizing. Q to quit.")
    try:
        while True:
            ir_img = ir.read()
            rgb_img = None
            if rgb is not None:
                ok, rgb_img = rgb.read()
                rgb_img = rgb_img if ok else None
            if ir_img is None:
                await asyncio.sleep(0.005)
                continue

            ir_bgr = to_bgr(ir_img)
            ir_face = engine.detect(ir_bgr)
            rgb_face = engine.detect(rgb_img) if rgb_img is not None else None
            ir_feat = engine.embed(ir_bgr, ir_face) if ir_face is not None else None
            rgb_feat = engine.embed(rgb_img, rgb_face) if rgb_face is not None else None

            # Liveness: RGB sees a face but IR doesn't, or IR face is flat -> suspicious.
            live = 1.0
            if rgb_face is not None and ir_face is None:
                live = 0.1
            elif ir_face is not None and ir_face_contrast(ir_img, ir_face) < MIN_IR_FACE_CONTRAST:
                live = 0.3

            for name, rec in db.items():
                s_ir = best_similarity(ir_feat, rec["ir"])
                s_rgb = best_similarity(rgb_feat, rec["rgb"])
                if s_ir is not None and s_rgb is not None:
                    p = IR_WEIGHT * similarity_to_probability(s_ir) + \
                        (1 - IR_WEIGHT) * similarity_to_probability(s_rgb)
                elif s_ir is not None:
                    p = similarity_to_probability(s_ir)
                elif s_rgb is not None:
                    p = similarity_to_probability(s_rgb)
                else:
                    p = 0.0
                p *= live
                smoothed[name] = (1 - SMOOTHING) * smoothed[name] + SMOOTHING * p

            best = max(smoothed, key=smoothed.get)
            prob = smoothed[best]
            label = f"{best}: {prob * 100:.0f}%" if (ir_face is not None or rgb_face is not None) else "no face"
            color = (0, 200, 0) if prob >= 0.8 else (0, 200, 255) if prob >= 0.5 else (0, 0, 255)

            if ir_face is not None:
                draw_face(ir_bgr, ir_face, color, label)
            if rgb_img is not None and rgb_face is not None:
                draw_face(rgb_img, rgb_face, color, label)
            y = 25
            for name, val in sorted(smoothed.items(), key=lambda kv: -kv[1]):
                cv2.putText(ir_bgr, f"{name}: {val * 100:5.1f}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y += 20
            if live < 1.0:
                cv2.putText(ir_bgr, "LIVENESS CHECK FAILED", (10, ir_bgr.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if show({"IR": ir_bgr, "RGB": rgb_img}) == ord("q"):
                break
    finally:
        await ir.close()
        if rgb is not None:
            rgb.release()
        cv2.destroyAllWindows()


async def cmd_list(args):
    db = db_load()
    if not db:
        print("No faces enrolled.")
    for name, rec in db.items():
        print(f"  {name}: {len(rec['ir'])} IR, {len(rec['rgb'])} RGB samples")


async def cmd_delete(args):
    f = DB_DIR / f"{args.name}.npz"
    if f.exists():
        f.unlink()
        print(f"Deleted '{args.name}'.")
    else:
        print(f"No one called '{args.name}'.")


def main():
    ap = argparse.ArgumentParser(description="IR + RGB face auth")
    ap.add_argument("--rgb", type=int, default=0, help="OpenCV webcam index, -1 to disable")
    ap.add_argument("--samples", type=int, default=20, help="frames to capture when enrolling")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cameras")
    e = sub.add_parser("enroll"); e.add_argument("name")
    sub.add_parser("list")
    d = sub.add_parser("delete"); d.add_argument("name")
    sub.add_parser("recognize")
    args = ap.parse_args()

    handlers = {"cameras": cmd_cameras, "enroll": cmd_enroll, "list": cmd_list,
                "delete": cmd_delete, "recognize": cmd_recognize}
    try:
        asyncio.run(handlers[args.cmd](args))
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
