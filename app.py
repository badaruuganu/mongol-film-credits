from __future__ import annotations
import os, re, json, shutil, subprocess, tempfile, threading, time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from flask import Flask, render_template, request, jsonify, send_file
import yt_dlp
import cv2
import pytesseract
import xlwt

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
EXPORT_DIR = DATA_DIR / "exports"
for p in (JOBS_DIR, EXPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Optional: set these environment variables on the hosting server.
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", shutil.which("ffmpeg") or "ffmpeg")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
OCR_LANG = os.environ.get("OCR_LANG", "mon+eng")
JOBS = {}
LOCK = threading.Lock()

CATEGORIES = [
    "Жүжигчин", "Дүр", "Найруулагч", "Зохиолч", "Зохиолын эх сурвалж",
    "Продюсер", "Гүйцэтгэх продюсер", "Ерөнхий зураглаач", "Зураглаач",
    "Камерын баг", "Эвлүүлэгч", "Хөгжмийн зохиолч", "Дууны найруулагч",
    "Дууны баг", "Гэрэлтүүлэг", "Уран сайхны удирдагч", "Зураг авалтын дизайн",
    "Хувцас дизайнер", "Нүүр будалт", "Үс засалт", "Кастинг", "Продакшн",
    "Байршил/зураг авалтын газар", "Тусгай эффект", "VFX", "Анимейшн",
    "Туслах найруулагч", "Бусад"
]

def safe(s):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(s or "film"))[:100].strip() or "film"

def video_id(url):
    u = urlparse(url)
    if u.hostname in ("youtu.be", "www.youtu.be"):
        return u.path.strip("/")
    if u.hostname and ("youtube.com" in u.hostname):
        if u.path == "/watch":
            return parse_qs(u.query).get("v", [""])[0]
        if u.path.startswith("/shorts/") or u.path.startswith("/embed/"):
            return u.path.split("/")[2]
    return ""

def update_job(job_id, **kwargs):
    with LOCK:
        JOBS.setdefault(job_id, {}).update(kwargs)
        JOBS[job_id]["updated"] = time.time()

def extract_frames(video_path, frame_dir, tail_seconds=1200, interval=1.5, max_frames=900):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Видео нээж чадсангүй.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps else 0
    start = max(0, duration - tail_seconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames, t = [], start
    while t < duration and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w < 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        out = frame_dir / f"frame_{int(t*10):07d}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frames.append((out, round(t, 1)))
        t += interval
    cap.release()
    return frames, duration

def read_credit_text(path):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    # Credits may scroll through the center, not only at the bottom; OCR both full frame and lower area.
    variants = [img, img[int(h*0.35):h, :]]
    results = []
    for crop in variants:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        try:
            txt = pytesseract.image_to_string(gray, lang=OCR_LANG, config="--oem 1 --psm 6")
        except Exception:
            try:
                txt = pytesseract.image_to_string(gray, lang="eng", config="--oem 1 --psm 6")
            except Exception:
                txt = ""
        if txt.strip():
            results.append(txt.strip())
    return max(results, key=len) if results else ""

def parse_lines(text):
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" \t|•·-")
        if len(line) >= 2 and len(line) <= 160:
            # discard OCR noise consisting only of punctuation/numbers
            if re.search(r"[A-Za-zА-Яа-яЁёӨөҮү]", line):
                lines.append(line)
    return lines

def classify_line(line):
    """Conservative rule-based starter; ambiguous lines remain unclassified."""
    low = line.lower()
    rules = [
        (["director", "найруулагч"], "Найруулагч"),
        (["written by", "writer", "screenplay", "зохиолч", "зохиол"], "Зохиолч"),
        (["producer", "продюсер"], "Продюсер"),
        (["executive producer", "гүйцэтгэх продюсер"], "Гүйцэтгэх продюсер"),
        (["cinematography", "cinematographer", "director of photography", "зураглаач"], "Зураглаач"),
        (["edited by", "editor", "эвлүүлэгч", "эвлүүлэг"], "Эвлүүлэгч"),
        (["music by", "composer", "хөгжмийн зохиолч", "хөгжим"], "Хөгжмийн зохиолч"),
        (["sound", "дууны найруулагч", "дууны баг"], "Дууны баг"),
        (["casting", "cast", "starring", "жүжигчин", "гол дүр"], "Жүжигчин"),
        (["costume", "wardrobe", "хувцас"], "Хувцас дизайнер"),
        (["makeup", "make-up", "hair", "нүүр будалт", "үс засалт"], "Нүүр будалт/үс засалт"),
        (["production", "produced by", "продакшн"], "Продакшн"),
        (["location", "filmed in", "зураг авалтын газар"], "Байршил/зураг авалтын газар"),
        (["visual effects", "vfx", "special effects", "тусгай эффект"], "Тусгай эффект/VFX"),
    ]
    for keys, cat in rules:
        if any(k in low for k in keys):
            return cat
    return "Шалгах шаардлагатай"

def process_job(job_id, url):
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        update_job(job_id, status="working", progress=5, message="YouTube видеоны мэдээлэл авч байна…")
        # Do not use channel metadata in output; only video-level title/id/duration/source URL.
        opts = {
            "outtmpl": str(job_dir / "%(title).100B_%(id)s.%(ext)s"),
            "format": "best[height<=720]/best",
            "noplaylist": True, "quiet": True, "no_warnings": True,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared = Path(ydl.prepare_filename(info))
        if not prepared.exists():
            candidates = list(job_dir.glob("*"))
            videos = [p for p in candidates if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
            if not videos:
                raise RuntimeError("Видео файл татагдсангүй. Видео хувийн, нас/бүсийн хязгаарлалттай эсвэл таталт боломжгүй байж болно.")
            prepared = max(videos, key=lambda p:p.stat().st_size)
        title = info.get("title") or "Киноны нэр тодорхойгүй"
        duration = int(info.get("duration") or 0)
        update_job(job_id, title=title, duration=duration, progress=20, message="Төгсгөлийн титрийн кадруудыг гаргаж байна…")
        frames_dir = job_dir / "frames"
        frames, actual_duration = extract_frames(prepared, frames_dir)
        update_job(job_id, progress=45, message=f"{len(frames)} кадрыг OCR-оор уншиж байна…")
        raw_rows = []
        text_seen = set()
        total = max(1, len(frames))
        for idx, (frame, seconds) in enumerate(frames):
            text = read_credit_text(frame)
            for line in parse_lines(text):
                key = re.sub(r"\W+", "", line.casefold())
                if len(key) < 3:
                    continue
                if key in text_seen:
                    # keep first timestamp only
                    continue
                text_seen.add(key)
                raw_rows.append({
                    "film": title, "person": "", "role": classify_line(line),
                    "credit_text": line, "character": "", "timestamp": seconds,
                    "source": url, "confidence": "OCR — шалгах шаардлагатай",
                    "notes": "Титрийн кадраас таньсан текст; нэр/үүргийг баталгаажуулна."
                })
            if idx % 15 == 0:
                update_job(job_id, progress=45 + int(45 * idx / total),
                           message=f"Титр уншиж байна… {idx+1}/{total} кадр")
        if not raw_rows:
            raw_rows.append({
                "film": title, "person": "", "role": "Шалгах шаардлагатай",
                "credit_text": "Текст танигдсангүй. OCR хэлний багц, видео чанар болон титрийн хэсгийг шалгана уу.",
                "character": "", "timestamp": "", "source": url,
                "confidence": "Мэдээлэл олдоогүй", "notes": "Автомат танилт амжилтгүй."
            })
        # Save JSON for the UI and export.
        result_json = job_dir / "results.json"
        result_json.write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        xls_path = EXPORT_DIR / f"{safe(title)}_{job_id[:6]}.xls"
        export_xls(raw_rows, xls_path, title)
        update_job(job_id, status="done", progress=100, message="Боловсруулалт дууслаа.",
                   rows=raw_rows, export=str(xls_path), row_count=len(raw_rows))
    except Exception as e:
        update_job(job_id, status="error", progress=100, message=str(e))

def export_xls(rows, path, title):
    wb = xlwt.Workbook(encoding="utf-8")
    header = xlwt.easyxf("font: bold on, color white; pattern: pattern solid, fore_colour dark_blue;")
    ws = wb.add_sheet("Уран бүтээлчид", cell_overwrite_ok=True)
    headers = ["Киноны нэр", "Хүний нэр (шалгах)", "Үүрэг/ангилал", "Дүрийн нэр", "Титрээс таньсан мөр",
               "Кадрын цаг (сек)", "Эх видео линк", "Баталгаажуулалт", "Тэмдэглэл"]
    widths = [7000, 6000, 6000, 5000, 10000, 3500, 10000, 7000, 10000]
    for c, h in enumerate(headers):
        ws.write(0, c, h, header)
        ws.col(c).width = widths[c]
    for r, item in enumerate(rows, 1):
        values = [item.get("film",""), item.get("person",""), item.get("role",""),
                  item.get("character",""), item.get("credit_text",""), item.get("timestamp",""),
                  item.get("source",""), item.get("confidence",""), item.get("notes","")]
        for c, value in enumerate(values):
            ws.write(r, c, value if value is not None else "")
    ws.set_panes_frozen(True)
    ws.set_horz_split_pos(1)
    wb.save(str(path))

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/process")
def start_process():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    vid = video_id(url)
    if not vid:
        return jsonify({"error":"YouTube видео линк зөв оруулна уу."}), 400
    job_id = f"{vid}_{int(time.time())}"
    update_job(job_id, status="queued", progress=0, message="Дараалалд нэмлээ.")
    thread = threading.Thread(target=process_job, args=(job_id, url), daemon=True)
    thread.start()
    return jsonify({"job_id":job_id})

@app.get("/api/status/<job_id>")
def job_status(job_id):
    with LOCK:
        job = dict(JOBS.get(job_id, {}))
    if not job:
        return jsonify({"error":"Ажил олдсонгүй. Сервер дахин ассан байж магадгүй."}), 404
    result = {k:v for k,v in job.items() if k not in ("rows","export")}
    if job.get("rows") is not None:
        result["rows"] = job["rows"]
        result["download_url"] = f"/api/download/{job_id}"
    return jsonify(result)

@app.get("/api/download/<job_id>")
def download(job_id):
    with LOCK:
        job = dict(JOBS.get(job_id, {}))
    path = job.get("export")
    if not path or not Path(path).exists():
        return "Excel файл хараахан бэлэн болоогүй байна.", 404
    return send_file(path, as_attachment=True, download_name=Path(path).name)

@app.get("/health")
def health():
    return jsonify({"ok":True, "app":"MongolFilmCredits Web MVP"})

if __name__ == "__main__":
    # Development only. For public deployment use a production WSGI server and HTTPS.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
