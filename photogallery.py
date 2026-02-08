import io
import os
import math
import uuid
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# -----------------------------
# App config
# -----------------------------
APP_TITLE = "Atharv Vision — Secure Image Pages"
IMAGES_PER_PAGE = 3                 # ✅ 3 images per PDF page
GRID_COLS = 1                       # ✅ UI: vertical stack preview (best for “perfect” look)
DEFAULT_WATERMARK_TEXT = "ATHARV VISION"

# Resize to keep performance stable for large batches
MAX_RENDER_DIM = 1600


# -----------------------------
# Data
# -----------------------------
@dataclass
class PhotoMeta:
    filename: str
    photo_id: str


# -----------------------------
# Helpers
# -----------------------------
def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def normalize_filename(name: str) -> str:
    return name


def safe_open_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGBA")


def resize_for_speed(img: Image.Image, max_dim: int = MAX_RENDER_DIM) -> Image.Image:
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def add_hologram_watermark(
    img: Image.Image,
    watermark_text: str,
    embed_text: str,
    strength: float = 0.24,
    angle: int = 28,
    pattern_gap: int = 240,
) -> Image.Image:
    base = img.copy().convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    font_main = load_font(34)
    font_side = load_font(22)
    font_id = load_font(18)

    # --- Repeated diagonal watermark pattern ---
    tile_w, tile_h = w * 2, h * 2
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)

    alpha = int(255 * max(0.08, min(0.6, strength)))
    color = (255, 255, 255, alpha)

    for y in range(0, tile_h, pattern_gap):
        for x in range(0, tile_w, pattern_gap):
            td.text((x, y), watermark_text, font=font_main, fill=color)

    tile = tile.rotate(angle, resample=Image.BICUBIC, expand=False)
    overlay.alpha_composite(tile, dest=(-w // 2, -h // 2))

    # --- Side watermark strip (right edge) ---
    side = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(side)
    strip_w = max(44, w // 26)
    sd.rectangle([w - strip_w, 0, w, h], fill=(0, 0, 0, 80))

    side_text_img = Image.new("RGBA", (h, 90), (0, 0, 0, 0))
    sdi = ImageDraw.Draw(side_text_img)
    sdi.text((10, 26), "ATHARV VISION", font=font_side, fill=(255, 255, 255, 190))
    side_text_img = side_text_img.rotate(90, expand=True)

    sx = w - side_text_img.size[0] - 8
    sy = (h - side_text_img.size[1]) // 2
    side.alpha_composite(side_text_img, dest=(max(0, sx), max(0, sy)))
    overlay = Image.alpha_composite(overlay, side)

    # --- Embedded ID/serial bottom-left ---
    id_box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    idd = ImageDraw.Draw(id_box)

    label = embed_text
    bbox = idd.textbbox((0, 0), label, font=font_id)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad = 10
    x = pad
    y = h - th - pad

    pill_w = tw + 18
    pill_h = th + 10
    idd.rounded_rectangle(
        [x - 6, y - 4, x - 6 + pill_w, y - 4 + pill_h],
        radius=10,
        fill=(0, 0, 0, 110),
    )
    idd.text((x, y), label, font=font_id, fill=(255, 255, 255, 230))

    overlay = Image.alpha_composite(overlay, id_box)
    return Image.alpha_composite(base, overlay)


def pil_to_jpg_bytes(img_rgba: Image.Image, quality: int = 90) -> bytes:
    rgb = Image.new("RGB", img_rgba.size, (255, 255, 255))
    rgb.paste(img_rgba, mask=img_rgba.split()[-1])
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# -----------------------------
# PDF layout helpers (best UI)
# -----------------------------
def _fit_in_box(iw: int, ih: int, bw: float, bh: float) -> Tuple[float, float]:
    """Fit image inside box preserving aspect ratio."""
    scale = min(bw / iw, bh / ih)
    return iw * scale, ih * scale


def _truncate_text(c, text: str, max_width: float, font_name="Helvetica", font_size=10) -> str:
    """Truncate text with ... so it fits max_width in PDF."""
    c.setFont(font_name, font_size)
    if c.stringWidth(text, font_name, font_size) <= max_width:
        return text

    ell = "..."
    max_w = max_width - c.stringWidth(ell, font_name, font_size)
    if max_w <= 0:
        return ell

    out = ""
    for ch in text:
        if c.stringWidth(out + ch, font_name, font_size) > max_w:
            break
        out += ch
    return out + ell


def _draw_header(c, page_w, page_h, margin_x, margin_top, title_left: str, title_right: str):
    # Clean header: title left, page label right
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin_x, page_h - margin_top + 4, title_left)

    c.setFont("Helvetica", 10)
    right_w = c.stringWidth(title_right, "Helvetica", 10)
    c.drawString(page_w - margin_x - right_w, page_h - margin_top + 6, title_right)

    # thin line under header
    y_line = page_h - margin_top - 6
    c.setLineWidth(0.6)
    c.setStrokeGray(0.7)
    c.line(margin_x, y_line, page_w - margin_x, y_line)


def _draw_slot_frame(c, x, y, w, h, radius=10):
    # subtle rounded frame (best “UI” look)
    c.setLineWidth(0.8)
    c.setStrokeGray(0.82)
    c.setFillGray(0.98)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)

    # inner border
    c.setLineWidth(0.6)
    c.setStrokeGray(0.88)
    c.roundRect(x + 1.3, y + 1.3, w - 2.6, h - 2.6, radius - 2, stroke=1, fill=0)


def build_page_pdf(items: List[Tuple[str, bytes]], title: str) -> bytes:
    """
    ✅ Best PDF UI:
    - 3 full-width vertical slots (rounded cards)
    - image top-aligned in each slot
    - filename tight below image
    - clean header with line
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    margin_x = 28
    margin_top = 34
    margin_bottom = 28

    header_h = 26
    slot_gap = 12
    caption_h = 14
    inner_pad = 10

    usable_h = page_h - margin_top - margin_bottom - header_h - (2 * slot_gap)
    slot_h = usable_h / 3
    slot_w = page_w - (2 * margin_x)

    _draw_header(c, page_w, page_h, margin_x, margin_top, title, "Secure • Watermarked")

    top_y = page_h - margin_top - header_h
    slots = [
        (margin_x, top_y - slot_h, slot_w, slot_h),
        (margin_x, top_y - (2 * slot_h) - slot_gap, slot_w, slot_h),
        (margin_x, top_y - (3 * slot_h) - (2 * slot_gap), slot_w, slot_h),
    ]

    for idx, (filename, jpg_bytes) in enumerate(items[:3]):
        x, y, w, h = slots[idx]

        # Card frame
        _draw_slot_frame(c, x, y, w, h, radius=12)

        # Image box (inside card)
        img_box_x = x + inner_pad
        img_box_y = y + caption_h + inner_pad
        img_box_w = w - 2 * inner_pad
        img_box_h = h - caption_h - 2 * inner_pad

        img_reader = ImageReader(io.BytesIO(jpg_bytes))
        iw, ih = Image.open(io.BytesIO(jpg_bytes)).size
        dw, dh = _fit_in_box(iw, ih, img_box_w, img_box_h)

        # Top-align image within its box
        dx = img_box_x + (img_box_w - dw) / 2
        dy = img_box_y + (img_box_h - dh)

        c.drawImage(img_reader, dx, dy, dw, dh, preserveAspectRatio=True, mask="auto")

        # Filename (tight, truncated)
        c.setFont("Helvetica", 10)
        safe_text = _truncate_text(c, normalize_filename(filename), max_width=w - 2 * inner_pad)
        c.setFillGray(0.15)
        c.drawString(x + inner_pad, y + 4, safe_text)

    c.showPage()
    c.save()
    return buf.getvalue()


def build_all_pages_pdf(page_chunks: List[List[Tuple[str, bytes]]], out_path: str, header_prefix: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    page_w, page_h = A4

    margin_x = 28
    margin_top = 34
    margin_bottom = 28

    header_h = 26
    slot_gap = 12
    caption_h = 14
    inner_pad = 10

    usable_h = page_h - margin_top - margin_bottom - header_h - (2 * slot_gap)
    slot_h = usable_h / 3
    slot_w = page_w - (2 * margin_x)

    total = len(page_chunks)

    for p_idx, items in enumerate(page_chunks, start=1):
        _draw_header(
            c,
            page_w,
            page_h,
            margin_x,
            margin_top,
            f"{header_prefix}",
            f"Page {p_idx}/{total}",
        )

        top_y = page_h - margin_top - header_h
        slots = [
            (margin_x, top_y - slot_h, slot_w, slot_h),
            (margin_x, top_y - (2 * slot_h) - slot_gap, slot_w, slot_h),
            (margin_x, top_y - (3 * slot_h) - (2 * slot_gap), slot_w, slot_h),
        ]

        for idx, (filename, jpg_bytes) in enumerate(items[:3]):
            x, y, w, h = slots[idx]
            _draw_slot_frame(c, x, y, w, h, radius=12)

            img_box_x = x + inner_pad
            img_box_y = y + caption_h + inner_pad
            img_box_w = w - 2 * inner_pad
            img_box_h = h - caption_h - 2 * inner_pad

            img_reader = ImageReader(io.BytesIO(jpg_bytes))
            iw, ih = Image.open(io.BytesIO(jpg_bytes)).size
            dw, dh = _fit_in_box(iw, ih, img_box_w, img_box_h)

            dx = img_box_x + (img_box_w - dw) / 2
            dy = img_box_y + (img_box_h - dh)

            c.drawImage(img_reader, dx, dy, dw, dh, preserveAspectRatio=True, mask="auto")

            c.setFont("Helvetica", 10)
            safe_text = _truncate_text(c, normalize_filename(filename), max_width=w - 2 * inner_pad)
            c.setFillGray(0.15)
            c.drawString(x + inner_pad, y + 4, safe_text)

        c.showPage()

    c.save()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
<style>
:root { --card-bg: rgba(255,255,255,0.06); --border: rgba(255,255,255,0.10); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.6rem; }
.small-muted { opacity: 0.75; font-size: 0.92rem; }

.header-badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(0,0,0,0.18);
  font-size: 0.86rem;
  opacity: 0.9;
}

.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 12px 12px 10px 12px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.20);
}

.caption-tight {
  margin-top: 4px;
  padding: 0px 2px;
  font-size: 0.95rem;
  opacity: 0.92;
  word-break: break-all;
}

.toolbar {
  display:flex; gap:12px; align-items:center;
  padding: 10px 12px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.0rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.markdown(
    """
<div class="small-muted">
3 images per page • vertically stacked in PDF • clean card UI in PDF • filenames provide traceability
</div>
""",
    unsafe_allow_html=True,
)

# Session init
if "files" not in st.session_state:
    st.session_state.files = []  # list of (filename, bytes)
if "metas" not in st.session_state:
    st.session_state.metas = []  # list[PhotoMeta]
if "wm" not in st.session_state:
    st.session_state.wm = {
        "wm_text": DEFAULT_WATERMARK_TEXT,
        "strength": 0.24,
        "angle": 28,
        "pattern_gap": 240,
    }

# Top controls
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.subheader("Upload images")
    uploaded = st.file_uploader(
        "Select images (jpg/png/webp). You can select many at once.",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    st.caption("Tip: For big batches (1000+), keep images under ~8–12 MB each for smoother processing.")

with right:
    st.subheader("Watermark settings")
    st.session_state.wm["wm_text"] = st.text_input("Watermark text", st.session_state.wm["wm_text"])
    st.session_state.wm["strength"] = st.slider("Strength", 0.10, 0.60, float(st.session_state.wm["strength"]), 0.01)
    st.session_state.wm["angle"] = st.slider("Angle", 0, 60, int(st.session_state.wm["angle"]), 1)
    st.session_state.wm["pattern_gap"] = st.slider("Pattern spacing", 160, 420, int(st.session_state.wm["pattern_gap"]), 10)
    st.markdown("<span class='header-badge'>Best look: strength 0.20–0.30 • spacing 220–300</span>", unsafe_allow_html=True)

# Load images
if uploaded:
    st.session_state.files = [(f.name, f.read()) for f in uploaded]
    st.session_state.metas = [
        PhotoMeta(filename=fn, photo_id=str(uuid.uuid4())[:10].upper())
        for (fn, _b) in st.session_state.files
    ]

if not st.session_state.files:
    st.info("Upload images to start.")
    st.stop()

files = st.session_state.files
metas = st.session_state.metas
wm = st.session_state.wm

n = len(files)
total_pages = math.ceil(n / IMAGES_PER_PAGE)

# Cache watermarking
@st.cache_data(show_spinner=False)
def get_watermarked_jpg(file_bytes: bytes, wm_text: str, strength: float, angle: int, pattern_gap: int, embed_text: str) -> bytes:
    img = safe_open_image(file_bytes)
    img = resize_for_speed(img, MAX_RENDER_DIM)
    out = add_hologram_watermark(
        img=img,
        watermark_text=wm_text,
        embed_text=embed_text,
        strength=strength,
        angle=angle,
        pattern_gap=pattern_gap,
    )
    return pil_to_jpg_bytes(out)

st.divider()

# Toolbar
t1, t2, t3, t4 = st.columns([1, 1, 1, 2], gap="large")
with t1:
    page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
with t2:
    st.markdown(f"<div class='toolbar'><b>Total images</b>: {n}</div>", unsafe_allow_html=True)
with t3:
    st.markdown(f"<div class='toolbar'><b>Total pages</b>: {total_pages}</div>", unsafe_allow_html=True)
with t4:
    st.markdown("<div class='toolbar'>Exports: current page PDF • all images ZIP • full PDF</div>", unsafe_allow_html=True)

start = (page - 1) * IMAGES_PER_PAGE
end = min(start + IMAGES_PER_PAGE, n)
indices = list(range(start, end))

# UI preview: vertical stack
for i in indices:
    filename, file_bytes = files[i]
    meta = metas[i]

    embed_text = f"ATHARV VISION | {meta.photo_id} | {filename}"
    jpg_bytes = get_watermarked_jpg(
        file_bytes=file_bytes,
        wm_text=wm["wm_text"],
        strength=float(wm["strength"]),
        angle=int(wm["angle"]),
        pattern_gap=int(wm["pattern_gap"]),
        embed_text=embed_text,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(jpg_bytes, use_container_width=True)
    st.markdown(f"<div class='caption-tight'>{normalize_filename(filename)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

st.divider()

# -----------------------------
# Downloads
# -----------------------------
d1, d2, d3 = st.columns([1, 1, 1], gap="large")

# Current page PDF
with d1:
    page_items: List[Tuple[str, bytes]] = []
    for i in indices:
        filename, file_bytes = files[i]
        meta = metas[i]

        embed_text = f"ATHARV VISION | {meta.photo_id} | {filename}"
        jpg_bytes = get_watermarked_jpg(
            file_bytes=file_bytes,
            wm_text=wm["wm_text"],
            strength=float(wm["strength"]),
            angle=int(wm["angle"]),
            pattern_gap=int(wm["pattern_gap"]),
            embed_text=embed_text,
        )
        page_items.append((filename, jpg_bytes))

    pdf_bytes = build_page_pdf(page_items, title=f"Atharv Vision — Page {page}/{total_pages}")
    st.download_button(
        "Download THIS page PDF",
        data=pdf_bytes,
        file_name=f"atharv_vision_page_{page}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# All images ZIP
with d2:
    st.markdown("<div class='small-muted'>Builds a ZIP with all watermarked images (can be large).</div>", unsafe_allow_html=True)
    if st.button("Build ALL images ZIP", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            zip_path = tmp.name

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for idx in range(n):
                    filename, file_bytes = files[idx]
                    meta = metas[idx]

                    embed_text = f"ATHARV VISION | {meta.photo_id} | {filename}"
                    jpg_bytes = get_watermarked_jpg(
                        file_bytes=file_bytes,
                        wm_text=wm["wm_text"],
                        strength=float(wm["strength"]),
                        angle=int(wm["angle"]),
                        pattern_gap=int(wm["pattern_gap"]),
                        embed_text=embed_text,
                    )

                    safe_name = filename.replace("/", "_").replace("\\", "_")
                    z.writestr(f"{meta.photo_id}_{safe_name}.jpg", jpg_bytes)

                    if idx % 10 == 0 or idx == n - 1:
                        prog.progress((idx + 1) / n)
                        status.write(f"Processing {idx + 1}/{n}...")

            status.success("ZIP ready.")
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download ALL images ZIP",
                    data=f.read(),
                    file_name="atharv_vision_all_watermarked_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        finally:
            try:
                os.remove(zip_path)
            except Exception:
                pass

# Full PDF
with d3:
    st.markdown("<div class='small-muted'>Creates a single PDF with all pages (3 per page). Heavy but supported.</div>", unsafe_allow_html=True)
    if st.button("Build FULL PDF (all pages)", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()

        page_chunks_indices = [
            list(range(p * IMAGES_PER_PAGE, min((p + 1) * IMAGES_PER_PAGE, n)))
            for p in range(total_pages)
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name

        try:
            all_pages_items: List[List[Tuple[str, bytes]]] = []

            for p_idx, idx_list in enumerate(page_chunks_indices, start=1):
                items: List[Tuple[str, bytes]] = []
                for i in idx_list:
                    filename, file_bytes = files[i]
                    meta = metas[i]

                    embed_text = f"ATHARV VISION | {meta.photo_id} | {filename}"
                    jpg_bytes = get_watermarked_jpg(
                        file_bytes=file_bytes,
                        wm_text=wm["wm_text"],
                        strength=float(wm["strength"]),
                        angle=int(wm["angle"]),
                        pattern_gap=int(wm["pattern_gap"]),
                        embed_text=embed_text,
                    )
                    items.append((filename, jpg_bytes))

                all_pages_items.append(items)

                if p_idx % 5 == 0 or p_idx == total_pages:
                    prog.progress(p_idx / total_pages)
                    status.write(f"Building pages {p_idx}/{total_pages}...")

            build_all_pages_pdf(all_pages_items, out_path=pdf_path, header_prefix="Atharv Vision")

            status.success("Full PDF ready.")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download FULL PDF",
                    data=f.read(),
                    file_name="atharv_vision_full.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        finally:
            try:
                os.remove(pdf_path)
            except Exception:
                pass

st.caption(
    "Note: watermarking discourages reuse/printing but cannot fully prevent screenshots or camera capture. "
    "This embeds watermark + serial inside the image pixels."
)
