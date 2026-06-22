#!/usr/bin/env python3
"""Transcribe / classify images with a local llama-server (Qwen3.6-VL).

Walks ./images, optionally pre-processes each image (auto-orient, VLM-grounded
crop to the text/page region, deskew + contrast + upscale), sends it to the
local llama-server OpenAI endpoint, and writes an Obsidian markdown note to
./transcriptions (frontmatter tags/type/source + transcription/description).

    python3 transcribe.py            # process new images only
    python3 transcribe.py --force    # redo everything
    PREP=0  python3 transcribe.py    # skip all image pre-processing
    CROP=0  python3 transcribe.py    # enhance only, skip the grounded crop
    KEEP_SERVER=1 python3 transcribe.py
"""
import base64
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUT = ROOT / "transcriptions"
PREP_DIR = ROOT / "_preprocessed"   # cleaned images actually sent to the model

# --- server config (override via env) ---
PORT = os.environ.get("PORT", "8080")
ENDPOINT = os.environ.get("ENDPOINT", f"http://localhost:{PORT}")
MODELDIR = Path(os.environ.get("MODELDIR", Path.home() / "models" / "qwen3.6"))
MODEL = Path(os.environ.get("MODEL", MODELDIR / "Qwen3.6-35B-A3B-UD-Q6_K.gguf"))
MMPROJ = Path(os.environ.get("MMPROJ", MODELDIR / "mmproj-F16.gguf"))
NGL = os.environ.get("NGL", "99")
CTX = os.environ.get("CTX", "16384")
KEEP_SERVER = os.environ.get("KEEP_SERVER") == "1"
# llama-server resolution order: $LLAMA_SERVER -> on PATH -> local fallback build
LLAMA_SERVER = os.environ.get("LLAMA_SERVER")
LLAMA_FALLBACK = Path.home() / "models/llama.cpp/llama-b9744/llama-server"

# --- preprocessing config ---
PREP = os.environ.get("PREP", "1") != "0"   # master switch for image prep
CROP = os.environ.get("CROP", "1") != "0"   # VLM-grounded crop (within PREP)
MIN_LONG_SIDE = 1500   # upscale crops smaller than this on the long edge
CROP_PAD = 0.0         # pad the detected box (0 = crop tight to the paper)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
MIME = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp",
        ".gif": "gif", ".bmp": "bmp", ".tiff": "tiff"}

PROMPT = """You are cataloguing an image for a personal knowledge vault.
Return ONLY a JSON object (no prose, no markdown fences) with these keys:
  "title":         a short human title (<=8 words)
  "type":          one of: document, screenshot, handwriting, photo, diagram, other
  "has_text":      true if the image contains readable text
  "transcription": ALL readable text, verbatim, preserving line breaks. "" if none.
                   Mark any word you cannot read confidently as [?].
  "description":   1-3 sentences describing the image content.
  "tags":          3-7 lowercase topical tags, single words or kebab-case, no '#'.
  "date":          the page's PRIMARY date as YYYY-MM-DD (when the page was
                   written). If several dates appear, pick the most prominent /
                   topmost. Infer the full year ('24 -> 2024). If only
                   month+year, use day 01. Use "" if no date is visible.
  "dates":         every date found on the page, each as YYYY-MM-DD ([] if none).
  "slug":          a 2-4 word lowercase hyphenated filename summary, e.g.
                   "food-notes" or "meeting-todo". Letters, digits, hyphens only.
Never invent text or dates that are not present in the image.
IMPORTANT: for any hard-to-read word, use the surrounding context (the topic and
the neighbouring words) to infer the most likely intended word before writing it
— these are personal handwritten notes (e.g. a plant listed with sweetcorn and
sunflower is far more likely "zinnias" than "minnias"). Only mark a word [?] if
you still cannot make a confident guess."""

LOCATE_PROMPT = """Return the TIGHTEST rectangle containing only the paper
document / notebook page(s) in this image. EXCLUDE everything around the
paper: any patterned or coloured background, the table or desk, shadows,
and any fingers or hands. Crop right up to the edges of the paper.
Return JSON with "found" and "box": [x0, y0, x1, y1] as FRACTIONS of
width and height from 0.0 to 1.0, (0,0)=top-left. Set found=false only if
there is no paper document visible."""

# Grammar-constrained output schemas (prevents markdown fences + runaways)
TRANSCRIBE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "type", "has_text", "transcription", "description", "tags"],
    "properties": {
        "title": {"type": "string"},
        "type": {"type": "string", "enum": ["document", "screenshot",
                 "handwriting", "photo", "diagram", "other"]},
        "has_text": {"type": "boolean"},
        "transcription": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"},
                 "minItems": 1, "maxItems": 7},
        "date": {"type": "string"},
        "dates": {"type": "array", "items": {"type": "string"}},
        "slug": {"type": "string"},
    },
}
TRANSCRIBE_SCHEMA["required"] += ["date", "dates", "slug"]
LOCATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["found", "box"],
    "properties": {
        "found": {"type": "boolean"},
        "box": {"type": "array", "items": {"type": "number"},
                "minItems": 4, "maxItems": 4},
    },
}

CORNERS_PROMPT = """Give the four corner points of the paper document /
notebook page(s) in this image, following the paper's actual (possibly
tilted) corners so it can be cropped and flattened. Order: top-left,
top-right, bottom-right, bottom-left. Coordinates are FRACTIONS of width
and height from 0.0 to 1.0, (0,0)=top-left. Set found=false if no paper
document is visible."""
CORNERS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["found", "corners"],
    "properties": {
        "found": {"type": "boolean"},
        "corners": {"type": "array", "minItems": 4, "maxItems": 4,
                    "items": {"type": "array", "items": {"type": "number"},
                              "minItems": 2, "maxItems": 2}},
    },
}


# ---------- model calls ----------
def data_uri(path: Path) -> str:
    mime = MIME.get(path.suffix.lower(), "png")
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def chat(path: Path, prompt: str, schema: dict = None,
         max_tokens: int = 4096, think: bool = False) -> str:
    if schema:
        rf = {"type": "json_schema",
              "json_schema": {"name": "out", "strict": True, "schema": schema}}
    else:
        rf = {"type": "json_object"}
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri(path)}},
        ]}],
        "temperature": 0,
        "response_format": rf,
        "max_tokens": max_tokens,
        # Qwen3.6 is a reasoning model. Thinking is OFF by default (otherwise it
        # burns the token budget on reasoning -> empty content / runaways), but
        # we enable it for the transcription pass when asked, since reasoning
        # helps it infer messy handwriting from context.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def ask(path: Path) -> dict:
    return parse_json(chat(path, PROMPT, schema=TRANSCRIBE_SCHEMA))


# ---------- image preprocessing (ImageMagick) ----------
def magick(*args) -> None:
    subprocess.run(["magick", *args], check=True, capture_output=True)


def identify_size(path: Path):
    out = subprocess.run(["magick", "identify", "-format", "%w %h", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[0]), int(out[1])


def to_fraction(box, w, h):
    """Normalize a model-returned box to 0..1 fractions, handling pixel or
    0..1000 conventions, then order + clamp."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    m = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if m > 1000:                      # absolute pixels
        x0, x1, y0, y1 = x0 / w, x1 / w, y0 / h, y1 / h
    elif m > 1.5:                     # 0..1000 convention
        x0, y0, x1, y1 = x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    area = (x1 - x0) * (y1 - y0)
    if area < 0.05 or area > 0.97:    # too small / basically the whole frame
        return None
    return x0, y0, x1, y1


def locate_region(path: Path):
    """Axis-aligned bounding box of the page (fallback when corners fail)."""
    try:
        d = parse_json(chat(path, LOCATE_PROMPT, schema=LOCATE_SCHEMA, max_tokens=200))
    except Exception:
        return None
    if not d.get("found") or "box" not in d:
        return None
    w, h = identify_size(path)
    return to_fraction(d["box"], w, h)


def locate_corners(path: Path):
    """Four corner points of the page -> ordered pixel quad [TL,TR,BR,BL], or
    None. Enables a perspective-warp crop that removes tilted background."""
    try:
        d = parse_json(chat(path, CORNERS_PROMPT, schema=CORNERS_SCHEMA, max_tokens=300))
    except Exception:
        return None
    if not d.get("found") or len(d.get("corners") or []) != 4:
        return None
    w, h = identify_size(path)
    flat = [abs(float(v)) for c in d["corners"] for v in c]
    m = max(flat) if flat else 0

    def to_px(x, y):
        x, y = float(x), float(y)
        if m > 1000:
            return x, y                              # pixels
        if m > 1.5:
            return x / 1000 * w, y / 1000 * h        # 0..1000
        return x * w, y * h                          # fractions

    pts = []
    for c in d["corners"]:
        x, y = to_px(c[0], c[1])
        pts.append((min(max(x, -0.05 * w), 1.05 * w),
                    min(max(y, -0.05 * h), 1.05 * h)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if (max(xs) - min(xs)) * (max(ys) - min(ys)) < 0.10 * w * h:
        return None                                  # quad too small
    s = sorted(pts, key=lambda p: p[0] + p[1])
    d2 = sorted(pts, key=lambda p: p[0] - p[1])
    quad = [s[0], d2[-1], s[-1], d2[0]]              # TL, TR, BR, BL
    return quad if len(set(quad)) == 4 else None


def warp_args(quad):
    """ImageMagick perspective-distort args + output size for a quad."""
    tl, tr, br, bl = quad
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    ow = max(int(max(d(br, bl), d(tr, tl))), 1)
    oh = max(int(max(d(tr, br), d(tl, bl))), 1)
    dmap = (f"{tl[0]:.1f},{tl[1]:.1f} 0,0 {tr[0]:.1f},{tr[1]:.1f} {ow},0 "
            f"{br[0]:.1f},{br[1]:.1f} {ow},{oh} {bl[0]:.1f},{bl[1]:.1f} 0,{oh}")
    return (["-virtual-pixel", "white",
             "-define", f"distort:viewport={ow}x{oh}+0+0",
             "-distort", "Perspective", dmap, "+repage"], ow, oh)


def preprocess(img: Path):
    """Return (path_to_send, note) — a cleaned image plus a short status word.
    Tries: perspective-warp to the page corners -> axis-box crop -> enhance only.
    Never modifies the original. Falls back to the original on any failure."""
    if not PREP or not shutil.which("magick"):
        return img, "raw"
    try:
        PREP_DIR.mkdir(exist_ok=True)
        base = PREP_DIR / f"{img.stem}.base.jpg"
        magick(str(img), "-auto-orient", "-quality", "95", str(base))  # upright
        w, h = identify_size(base)

        pre, deskew, note, ow, oh = [], ["-deskew", "40%", "+repage"], "enhanced", w, h
        if CROP:
            quad = locate_corners(base)
            if quad:
                pre, ow, oh = warp_args(quad)
                deskew, note = [], "warped"          # warp already straightens
            else:
                box = locate_region(base)
                if box:
                    x0, y0, x1, y1 = box
                    x0, y0 = max(0, x0 - CROP_PAD), max(0, y0 - CROP_PAD)
                    x1, y1 = min(1, x1 + CROP_PAD), min(1, y1 + CROP_PAD)
                    ow, oh = int((x1 - x0) * w), int((y1 - y0) * h)
                    cx, cy = int(x0 * w), int(y0 * h)
                    if ow > 0 and oh > 0:
                        pre = ["-crop", f"{ow}x{oh}+{cx}+{cy}", "+repage"]
                        note = "cropped"

        pipe = [str(base), *pre, *deskew, "-normalize", "-unsharp", "0x1"]
        if max(ow, oh) < MIN_LONG_SIDE:
            pipe += ["-resize", f"{MIN_LONG_SIDE}x{MIN_LONG_SIDE}"]  # enlarge small
        # distinct name (won't collide with the original) + compact jpeg, since
        # this is now the asset embedded in the note
        out = PREP_DIR / f"{img.stem}.cropped.jpg"
        magick(*pipe, "-quality", "90", str(out))
        base.unlink(missing_ok=True)
        return out, note
    except Exception as e:
        print(f"      (prep failed, using original: {e})")
        return img, "raw"


# ---------- note naming ----------
def valid_iso(date: str) -> bool:
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    # reject implausible years (model guessing a bare "7/1" as year 0007, etc.)
    return 1990 <= d.year <= datetime.date.today().year + 1


def sanitize(s: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s[:maxlen].strip("-") or "note"


def note_basename(d: dict, fallback: str) -> str:
    """'<date> <slug>' for dated pages, 'undated <slug>' otherwise."""
    slug = sanitize(d.get("slug") or fallback)
    date = str(d.get("date") or "").strip()
    return f"{date} {slug}" if valid_iso(date) else f"undated {slug}"


def existing_sources() -> dict:
    """Map 'images/NAME' -> existing note Path, via each note's source: line."""
    out = {}
    for p in OUT.glob("*.md"):
        try:
            m = re.search(r"^source:\s*(images/.+?)\s*$", p.read_text(), re.M)
        except Exception:
            continue
        if m:
            out[m.group(1)] = p
    return out


def unique_name(base: str, taken: set) -> str:
    if f"{base}.md" not in taken:
        return f"{base}.md"
    i = 2
    while f"{base} ({i}).md" in taken:
        i += 1
    return f"{base} ({i}).md"


# ---------- note output ----------
def yaml_list(items) -> str:
    return "\n".join(f"  - {str(i).strip()}" for i in items if str(i).strip())


def to_markdown(original: Path, sent: Path, d: dict) -> str:
    title = str(d.get("title") or original.stem).replace('"', "'")
    date = str(d.get("date") or "").strip()
    fm = ["---", f'title: "{title}"', f'type: {d.get("type", "other")}']
    if valid_iso(date):
        fm.append(f"date: {date}")
    fm += [f"source: images/{original.name}", "tags:",
           yaml_list(d.get("tags") or []) or "  - untagged",
           "---", "", f"![[{sent.name}]]", ""]   # embed the processed image
    body = []
    if d.get("has_text") and str(d.get("transcription", "")).strip():
        body += ["## Transcription", "", str(d["transcription"]).strip(), ""]
    if str(d.get("description", "")).strip():
        body += ["## Description", "", str(d["description"]).strip(), ""]
    return "\n".join(fm + body)


# ---------- server lifecycle ----------
def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server():
    if server_up():
        print(f"Using server already running at {ENDPOINT}")
        return None
    exe = LLAMA_SERVER or shutil.which("llama-server")
    if not exe and LLAMA_FALLBACK.exists():
        exe = str(LLAMA_FALLBACK)
    if not exe:
        print("llama-server not found. Install llama.cpp recent enough for the\n"
              "Qwen3.6 'qwen35moe' architecture, put llama-server on your PATH\n"
              "(or set the LLAMA_SERVER env var to its full path), then re-run.")
        return False
    for f in (MODEL, MMPROJ):
        if not f.exists():
            print(f"Missing model file: {f}\n(set MODEL/MMPROJ env, or see README)")
            return False
    env = dict(os.environ)
    # prebuilt tarball ships its .so libs next to the binary; resolve symlinks
    env["LD_LIBRARY_PATH"] = f"{Path(exe).resolve().parent}:{env.get('LD_LIBRARY_PATH','')}"
    log = open(MODELDIR / "llama-server.log", "w")
    print(f"Starting llama-server ({MODEL.name}) ... loading ~27GB into VRAM")
    proc = subprocess.Popen(
        [exe, "-m", str(MODEL), "--mmproj", str(MMPROJ),
         "-ngl", NGL, "-c", CTX, "--host", "127.0.0.1", "--port", PORT],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    for _ in range(300):
        if proc.poll() is not None:
            print(f"llama-server exited early — see {MODELDIR/'llama-server.log'}")
            return False
        if server_up():
            print("Server ready.\n")
            return proc
        time.sleep(1)
    print("Server did not become ready in time.")
    proc.terminate()
    return False


def main() -> int:
    proc = start_server()
    if proc is False:
        return 1
    try:
        return run()
    finally:
        if proc and not KEEP_SERVER:
            print("Stopping llama-server.")
            proc.terminate()


def run() -> int:
    force = "--force" in sys.argv
    OUT.mkdir(exist_ok=True)
    imgs = sorted(p for p in IMAGES.glob("*") if p.suffix.lower() in EXTS)
    if not imgs:
        print(f"No images in {IMAGES}/ — drop some in and re-run.")
        return 0
    print(f"{len(imgs)} image(s) | endpoint {ENDPOINT} | prep={'on' if PREP else 'off'}\n")
    src_map = existing_sources()                       # image -> existing note
    taken = {p.name for p in OUT.glob("*.md")}          # for collision checks
    for img in imgs:
        src = f"images/{img.name}"
        if src in src_map and not force:
            print(f"  skip  {img.name}")
            continue
        print(f"  ...   {img.name}", flush=True)
        try:
            send, note = preprocess(img)
            data = ask(send)
            old = src_map.get(src)
            if old:                                     # reprocessing: free old name
                taken.discard(old.name)
            target = unique_name(note_basename(data, img.stem), taken)
            if old and old.name != target:              # date/title changed -> rename
                old.unlink(missing_ok=True)
            dest = OUT / target
            dest.write_text(to_markdown(img, send, data))
            taken.add(target)
            src_map[src] = dest
            print(f"  ok    {img.name} [{note}] -> {dest.name}")
        except Exception as e:
            print(f"  FAIL  {img.name}: {e}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
