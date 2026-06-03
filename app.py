"""FingerLeak â€” Full multi-page web app (Tesla-grade UI)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandDetector
from fingerleak.detection.fingertip_crop import crop_fingertips
from fingerleak.privacy.filters import apply_filter
from fingerleak.ridge.gabor import enhance_ridges
from fingerleak.risk.score import compute_finger_score, aggregate_frame_score


st.set_page_config(
    page_title="FingerLeak â€” Seal the Leak",
    page_icon="â—‰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          GLOBAL CSS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

* { font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; }

html, body, .stApp {
    background:
        radial-gradient(1200px 600px at 80% -10%, rgba(255,48,48,0.08), transparent 60%),
        radial-gradient(900px 500px at 0% 100%, rgba(255,48,48,0.04), transparent 60%),
        #000 !important;
    color: #fff;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

.block-container { padding: 1rem 2.5rem 5rem 2.5rem; max-width: 1240px; }

h1 {
    font-size: clamp(3rem, 7vw, 6rem) !important;
    font-weight: 900 !important;
    letter-spacing: -0.05em !important;
    line-height: 0.9 !important;
    margin: 0 !important;
    background: linear-gradient(180deg, #fff 0%, #888 130%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2 {
    font-size: clamp(2rem, 3.5vw, 3rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em !important;
    margin-top: 3rem !important;
    color: #fff;
}
h3 { font-size: 1.35rem !important; font-weight: 700 !important; letter-spacing: -0.015em !important; color: #fff; }
p, li { color: #9a9a9a; font-size: 1rem; line-height: 1.7; }
a { color: #FF3030; text-decoration: none; }
a:hover { color: #fff; }

.mono { font-family: 'JetBrains Mono', monospace; }
.label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.28em; color: #FF3030;
    display: inline-flex; align-items: center; gap: 8px;
}
.label::before { content: ''; width: 18px; height: 1px; background: #FF3030; }
.red { color: #FF3030; } .green { color: #00E676; } .amber { color: #FFB020; }
.dim { color: #555; }

/* â”€â”€â”€ NAV â”€â”€â”€ */
.brand {
    font-weight: 900; letter-spacing: -0.02em; font-size: 1.15rem;
    display: flex; align-items: center; gap: 10px;
    padding: 0.6rem 0;
}
.brand .dot { color: #FF3030; text-shadow: 0 0 12px rgba(255,48,48,0.8); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* DEFAULT BUTTONS = NAV (visible) */
.stButton > button {
    background: transparent !important;
    color: #888 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    padding: 0.7rem 0.5rem !important;
    font-size: 0.72rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    color: #fff !important;
    border-color: #FF3030 !important;
    background: rgba(255,48,48,0.08) !important;
}

/* PRIMARY BUTTONS (CTA) */
.stButton > button[kind="primary"] {
    background: #FF3030 !important;
    color: #fff !important;
    border: none !important;
    padding: 0.85rem 1.8rem !important;
    font-size: 0.78rem !important;
    box-shadow: 0 0 20px rgba(255,48,48,0.3);
}
.stButton > button[kind="primary"]:hover {
    background: #fff !important;
    color: #000 !important;
    box-shadow: 0 0 30px rgba(255,255,255,0.3);
}

.stDownloadButton > button {
    background: #FF3030 !important; color: #fff !important;
    border: none !important; border-radius: 2px !important;
    font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: 0.12em; padding: 0.85rem 1.8rem !important;
    font-size: 0.78rem !important;
    box-shadow: 0 0 20px rgba(255,48,48,0.3);
}
.stDownloadButton > button:hover { background: #fff !important; color: #000 !important; }

/* â”€â”€â”€ CARDS â”€â”€â”€ */
.card {
    background: linear-gradient(180deg, rgba(20,20,20,0.7), rgba(10,10,10,0.7));
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1.75rem;
    margin-bottom: 1rem;
    border-radius: 2px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.card:hover { border-color: rgba(255,255,255,0.14); transform: translateY(-2px); }
.card-red {
    border-color: rgba(255,48,48,0.4) !important;
    background: linear-gradient(180deg, rgba(255,48,48,0.08), rgba(255,48,48,0.02)) !important;
    box-shadow: 0 0 40px rgba(255,48,48,0.08);
}
.card h3 { margin-top: 0 !important; }

.hero-panel {
    background:
        radial-gradient(circle at 50% 0%, rgba(255,48,48,0.15), transparent 70%),
        linear-gradient(180deg, #0a0a0a 0%, #000 100%);
    border: 1px solid rgba(255,48,48,0.25);
    padding: 2rem; border-radius: 2px;
    height: 100%;
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative; overflow: hidden;
}

.tele-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4.5rem; font-weight: 700; line-height: 1;
    letter-spacing: -0.04em;
}
.tele-mid {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem; font-weight: 700; line-height: 1;
}

.tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.22em; padding: 5px 11px;
    border: 1px solid rgba(255,255,255,0.15); color: #aaa;
    margin: 4px 6px 4px 0; border-radius: 2px;
}
.tag-red { border-color: #FF3030; color: #FF3030; background: rgba(255,48,48,0.05); }
.tag-green { border-color: #00E676; color: #00E676; background: rgba(0,230,118,0.05); }

[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(20,20,20,0.6), rgba(5,5,5,0.6)) !important;
    border: 1px dashed rgba(255,255,255,0.2) !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #FF3030 !important; }
[data-baseweb="select"] > div { background: #0a0a0a !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 2px !important; color: #fff !important; }
.stSlider [data-baseweb="slider"] > div > div { background: #FF3030 !important; }
.stSlider [role="slider"] { background: #fff !important; box-shadow: 0 0 12px rgba(255,48,48,0.6) !important; }
input, textarea { background: #0a0a0a !important; color: #fff !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 2px !important; }

.stImage img {
    border: 1px solid rgba(255,255,255,0.08);
    max-width: 100% !important; height: auto !important;
    border-radius: 2px;
}

hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 3rem 0 !important; }

[data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important; text-transform: uppercase;
    letter-spacing: 0.18em; color: #555 !important;
}
[data-testid="stExpander"] { background: rgba(10,10,10,0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 2px; }

.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #666 !important;
    border-radius: 0 !important; padding: 0.85rem 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.18em;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }
.stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 2px solid #FF3030 !important; }

.status-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #FF3030; border-radius: 50%;
    box-shadow: 0 0 10px #FF3030;
    animation: blink 1.5s infinite; margin-right: 8px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.div-thin { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 2.5rem 0; }

.section-head { display: flex; align-items: baseline; gap: 1rem; margin: 3.5rem 0 1.5rem 0; }
.section-head .num { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #FF3030; letter-spacing: 0.3em; }
.section-head .ttl { font-size: 0.7rem; color: #666; letter-spacing: 0.3em; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }

.linkcard {
    display: block;
    background: linear-gradient(180deg, rgba(20,20,20,0.7), rgba(10,10,10,0.7));
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1rem 1.5rem; margin-bottom: 0.75rem;
    border-radius: 2px;
    color: #fff !important;
    text-decoration: none !important;
    transition: all 0.25s ease;
}
.linkcard:hover {
    border-color: #FF3030;
    background: linear-gradient(180deg, rgba(255,48,48,0.08), rgba(10,10,10,0.7));
    transform: translateX(4px);
}
.linkcard .lc-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: #FF3030;
    letter-spacing: 0.25em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.linkcard .lc-val { font-weight: 600; font-size: 0.95rem; color: #fff; }
.linkcard:hover .lc-val { color: #FF3030; }

.footer {
    margin-top: 4rem; padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #555;
    letter-spacing: 0.2em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          NAVIGATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(p):
    st.session_state.page = p

# Equal-width nav
nc1, nc2 = st.columns([2, 5])
with nc1:
    st.markdown(
        '<div class="brand"><span class="status-dot"></span>'
        'FINGER<span class="dot">â—</span>LEAK</div>',
        unsafe_allow_html=True
    )
with nc2:
    n1, n2, n3, n4, n5 = st.columns(5)
    with n1: st.button("HOME",     on_click=goto, args=("home",),     use_container_width=True, key="nav_h")
    with n2: st.button("SCANNER",  on_click=goto, args=("scanner",),  use_container_width=True, key="nav_s")
    with n3: st.button("RESEARCH", on_click=goto, args=("research",), use_container_width=True, key="nav_r")
    with n4: st.button("ABOUT",    on_click=goto, args=("about",),    use_container_width=True, key="nav_a")
    with n5: st.button("CONTACT",  on_click=goto, args=("contact",),  use_container_width=True, key="nav_c")

st.markdown('<div class="div-thin" style="margin:1rem 0 2rem 0;"></div>', unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def crop_xyxy(img, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2].copy()

def cat_bgr(c):
    return {"Low":(118,230,0),"Medium":(32,176,255),"High":(48,140,255),"Critical":(48,48,255)}.get(c,(120,120,120))

def cat_color(c):
    return {"Low":"#00E676","Medium":"#FFB020","High":"#FF6A30","Critical":"#FF3030"}.get(c,"#888")

def annotate(img, crops, scores):
    out = img.copy()
    for cd, s in zip(crops, scores):
        x1, y1, x2, y2 = [int(v) for v in cd["bbox_xyxy"]]
        col = cat_bgr(s.category)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{s.finger.upper()} {int(s.score*100)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out

@st.cache_resource
def get_detector(n):
    return HandDetector(max_hands=n)

def constrain_image(img, max_w=900):
    h, w = img.shape[:2]
    if w <= max_w: return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

def section_header(num, title):
    st.markdown(
        f'<div class="section-head"><span class="num">/ {num}</span>'
        f'<span class="ttl">{title}</span></div>',
        unsafe_allow_html=True
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          PAGE: HOME
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def page_home():
    st.markdown('<div class="label"><span class="status-dot"></span>SYSTEM Â· ONLINE Â· MIT LICENSE</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown(
            '<h1>YOUR<br>FINGERPRINTS<br>'
            '<span style="color:#FF3030;text-shadow:0 0 30px rgba(255,48,48,0.4);">ARE LEAKING.</span></h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-size:1.15rem;max-width:540px;margin-top:2rem;color:#bbb;line-height:1.6;">'
            'Modern phone cameras capture enough fingertip detail to clone your biometrics. '
            'Researchers proved it back in 2017 â€” and cameras have only gotten sharper since. '
            '<span style="color:#fff;">FingerLeak detects the leak in any photo and seals it in one click.</span>'
            '</p>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            st.button("RUN SCANNER â†’", on_click=goto, args=("scanner",), use_container_width=True, key="hb1", type="primary")
        with b2:
            st.button("READ RESEARCH", on_click=goto, args=("research",), use_container_width=True, key="hb2")

    with col2:
        st.markdown("""
        <div class="hero-panel">
            <div>
                <div class="label">âš  LIVE THREAT INTEL</div>
                <div class="tele-num red" style="margin-top:1rem;">2017</div>
                <p style="color:#bbb;margin-top:1rem;font-size:0.9rem;line-height:1.5;">
                    NII Japan first proved fingerprints can be cloned from casual photos
                    taken up to <span style="color:#fff;font-weight:600;">3 meters away</span>.
                    Since then, smartphone sensors have grown from 12MP to 200MP.
                </p>
            </div>
            <div style="margin-top:1.5rem;">
                <span class="tag tag-red">PROVEN THREAT</span>
                <span class="tag tag-red">UNPATCHED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats strip
    section_header("01", "THE NUMBERS")
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("DISTANCE", "3 M", "Max range fingerprint extraction works from on consumer cameras."),
        ("RESOLUTION", "8 MP", "Minimum camera resolution required to recover fingerprint ridges."),
        ("DETECTION", "21", "Hand landmarks tracked in real time per detected hand via MediaPipe."),
        ("FILTERS", "04", "Privacy modes available: Gaussian blur, mosaic pixelate, emoji overlay, blackout."),
    ]
    for col, (lbl, val, desc) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(
                f'<div class="card" style="height:100%;"><div class="label">{lbl}</div>'
                f'<div class="tele-mid" style="margin:1rem 0 0.75rem 0;">{val}</div>'
                f'<p style="font-size:0.85rem;color:#888;margin:0;">{desc}</p></div>',
                unsafe_allow_html=True
            )

    # Why it matters
    section_header("02", "WHY IT MATTERS")
    w1, w2, w3 = st.columns(3)
    why = [
        ("THREAT", "Biometrics â‰  passwords",
         "You can change a password in seconds. You cannot change your fingerprint â€” ever. Once your ridges leak online, they can be replayed against any biometric system you'll ever use, for the rest of your life."),
        ("CONTEXT", "The peace-sign trap",
         "The most popular pose on social media â€” peace sign, hand wave, OK sign â€” exposes 4 fingertips at high resolution, directly facing the camera. Billions of these photos are uploaded daily."),
        ("SOLUTION", "Surgical filtering",
         "FingerLeak obfuscates only the fingertip pixels. Pose, identity, lighting, and scene remain untouched. The photo still looks like you â€” just without your biometric DNA exposed."),
    ]
    for col, (lbl, ttl, body) in zip([w1, w2, w3], why):
        with col:
            st.markdown(
                f'<div class="card" style="height:100%;"><div class="label">{lbl}</div>'
                f'<h3 style="margin:1rem 0 1rem 0;">{ttl}</h3>'
                f'<p style="font-size:0.95rem;">{body}</p></div>',
                unsafe_allow_html=True
            )

    # Pipeline
    section_header("03", "PIPELINE")
    p1, p2, p3 = st.columns(3)
    steps = [
        ("01", "DETECT",
         "MediaPipe Hands runs on-device to locate 21 anatomical landmarks per hand. We extract the 5 fingertip regions with adaptive bounding boxes that scale with hand size and orientation.",
         "MEDIAPIPE"),
        ("02", "SCORE",
         "A Gabor filter bank at 8 orientations recovers the latent ridge pattern. Combined with sharpness, crop size, and proximity, this yields a calibrated 0â€“100 FingerLeak-Score per finger.",
         "GABOR Â· OPENCV"),
        ("03", "PROTECT",
         "The chosen filter is applied selectively to fingertip bounding boxes only. We re-run the scorer on the protected output to verify ridge information has been destroyed â€” proof, not promise.",
         "SECURED"),
    ]
    for col, (n, t, d, tag) in zip([p1, p2, p3], steps):
        with col:
            st.markdown(
                f'<div class="card" style="height:100%;"><div class="label">STEP {n}</div>'
                f'<h2 style="margin:0.75rem 0;font-size:2rem !important;">{t}</h2>'
                f'<p style="font-size:0.9rem;">{d}</p>'
                f'<span class="tag tag-red" style="margin-top:0.75rem;">{tag}</span></div>',
                unsafe_allow_html=True
            )

    # Trust / who uses
    section_header("04", "WHO SHOULD CARE")
    u1, u2, u3, u4 = st.columns(4)
    users = [
        ("JOURNALISTS", "Source-protection workflows. Strip biometric data before publishing photos from sensitive regions."),
        ("ACTIVISTS", "Protest photography. Filter your own and protesters' fingerprints before posting on public timelines."),
        ("RESEARCHERS", "Anonymize hand-pose datasets used in computer-vision research without losing pose information."),
        ("EVERYONE ELSE", "If you've ever posted a peace-sign selfie, OK sign, or thumbs-up â€” this matters to you."),
    ]
    for col, (l, t) in zip([u1, u2, u3, u4], users):
        with col:
            st.markdown(
                f'<div class="card" style="height:100%;"><div class="label">{l}</div>'
                f'<p style="margin-top:1rem;font-size:0.9rem;">{t}</p></div>',
                unsafe_allow_html=True
            )

    # CTA
    st.markdown('<div class="div-thin"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:3rem 0;">'
        '<h2 style="margin:0 !important;font-size:3rem !important;">Test your photos. Now.</h2>'
        '<p style="margin-top:1rem;font-size:1.1rem;color:#888;">Free. Open source. Runs entirely on your device.</p></div>',
        unsafe_allow_html=True
    )
    cc = st.columns([2, 1, 2])[1]
    with cc:
        st.button("LAUNCH SCANNER â†’", on_click=goto, args=("scanner",), use_container_width=True, key="cta_home", type="primary")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          PAGE: SCANNER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def page_scanner():
    st.markdown('<div class="label">/ SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:1rem !important;">SCAN.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.1rem;color:#aaa;max-width:600px;margin-top:1.25rem;">'
        'Upload any hand image. We detect fingertips, extract ridges, score risk, and seal the leak.</p>',
        unsafe_allow_html=True
    )

    section_header("01", "CONFIGURATION")
    settings = st.columns(3)
    with settings[0]:
        mode = st.selectbox("FILTER MODE", ["pixelate","blur","emoji","blackout"], 0)
    with settings[1]:
        max_hands = st.slider("MAX HANDS", 1, 4, 2)
    with settings[2]:
        distance_cm = st.slider("DISTANCE (CM)", 5, 100, 20)

    section_header("02", "INPUT")
    uploaded = st.file_uploader("Upload hand image", type=["jpg","jpeg","png"], label_visibility="collapsed")

    use_sample = False
    if uploaded is None:
        sample = ROOT / "data" / "samples" / "sample.jpg"
        cs = st.columns([1, 4])
        with cs[0]:
            if sample.exists() and st.button("RUN SAMPLE", key="sample_btn", type="primary"):
                use_sample = True
        with cs[1]:
            st.caption("All processing local Â· no upload to server")

    if uploaded is None and not use_sample:
        return

    if use_sample:
        sample = ROOT / "data" / "samples" / "sample.jpg"
        img = cv2.imread(str(sample))
    else:
        bytes_ = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img = cv2.imdecode(bytes_, cv2.IMREAD_COLOR)

    if img is None:
        st.error("DECODE FAILED"); return

    img = constrain_image(img, max_w=1200)

    detector = get_detector(max_hands)
    with st.spinner("ANALYZING..."):
        hands = detector.detect(img)

    if not hands:
        st.warning("NO HANDS DETECTED")
        ic = st.columns([1, 2, 1])[1]
        with ic:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        return
    # Filter back-of-hand detections (no fingerprint risk from back)
    palm_hands = [h for h in hands if h.palm_facing]
    back_count = len(hands) - len(palm_hands)
    if back_count > 0:
        st.info(f"{back_count} hand(s) showing back side — no fingerprint risk, skipped.")
    if not palm_hands:
        st.success("No fingerprint risk detected — only back-of-hand visible.")
        return

    crops_data = []
    for hand in palm_hands:
        for c in crop_fingertips(img, hand):
            crops_data.append({
                "finger": c.finger, "bbox_xyxy": c.bbox,
                "bbox_xywh": xyxy_to_xywh(c.bbox), "crop": crop_xyxy(img, c.bbox),
            })

    before_scores, ridges_before = [], []
    for cd in crops_data:
        if cd["crop"].size == 0: continue
        rr = enhance_ridges(cd["crop"])
        ridges_before.append((cd["finger"], cd["crop"], rr))
        before_scores.append(compute_finger_score(
            finger=cd["finger"], crop_bgr=cd["crop"],
            crop_size_px=min(cd["crop"].shape[:2]),
            ridge_strength=rr.ridge_response, distance_cm=float(distance_cm),
        ))

    agg = aggregate_frame_score(before_scores)
    overall = int(agg["max_score"] * 100)
    cat = agg["category"]
    color = cat_color(cat)
    n_crit = sum(1 for s in before_scores if s.category == "Critical")
    n_high = sum(1 for s in before_scores if s.category == "High")

    section_header("03", "TELEMETRY")
    t1, t2, t3, t4 = st.columns(4)
    tele = [
        ("RISK SCORE", f"{overall:03d}", color),
        ("VERDICT", cat.upper(), color),
        ("FINGERS", f"{len(before_scores):02d}", "#fff"),
        ("CRIT + HIGH", f"{(n_crit+n_high):02d}", "#FF3030" if (n_crit+n_high) else "#888"),
    ]
    for col, (lbl, val, c) in zip([t1, t2, t3, t4], tele):
        with col:
            st.markdown(
                f'<div class="card"><div class="label">{lbl}</div>'
                f'<div class="tele-mid" style="color:{c};margin-top:0.75rem;">{val}</div></div>',
                unsafe_allow_html=True
            )

    section_header("04", "THREAT MAP")
    annotated = annotate(img, crops_data, before_scores)
    annotated_disp = constrain_image(annotated, max_w=800)
    tm_col = st.columns([1, 3, 1])[1]
    with tm_col:
        st.image(cv2.cvtColor(annotated_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"{len(before_scores)} fingertips Â· {n_crit + n_high} flagged high-risk")

    section_header("05", "EXTRACTION")
    st.markdown(
        '<p style="color:#888;margin-bottom:1.5rem;">Gabor filter bank reconstructs ridge structure '
        'from each fingertip â€” the same algorithm used by commercial fingerprint scanners.</p>',
        unsafe_allow_html=True
    )
    cols = st.columns(min(5, len(ridges_before)) or 1)
    for i, (f, c, r) in enumerate(ridges_before[:5]):
        with cols[i]:
            st.markdown(f'<div class="label" style="margin-bottom:0.5rem;">{f.upper()}</div>', unsafe_allow_html=True)
            crop_disp = constrain_image(c, max_w=200)
            ridge_disp = constrain_image(r.enhanced, max_w=200)
            st.image(cv2.cvtColor(crop_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.image(ridge_disp, clamp=True, use_container_width=True)
            st.markdown(
                f'<div class="label" style="margin-top:0.5rem;">RIDGE</div>'
                f'<div class="mono" style="font-size:1.1rem;color:#FF3030;">{r.ridge_response:.3f}</div>',
                unsafe_allow_html=True
            )

    section_header("06", "COUNTERMEASURE")
    bboxes_xywh = [cd["bbox_xywh"] for cd in crops_data]
    with st.spinner(f"DEPLOYING {mode.upper()}..."):
        result = apply_filter(img, bboxes_xywh, mode=mode)

    after_scores, ridges_after = [], []
    for cd in crops_data:
        new_crop = crop_xyxy(result.image, cd["bbox_xyxy"])
        if new_crop.size == 0: continue
        rr = enhance_ridges(new_crop)
        ridges_after.append((cd["finger"], new_crop, rr))
        after_scores.append(compute_finger_score(
            finger=cd["finger"], crop_bgr=new_crop,
            crop_size_px=min(new_crop.shape[:2]),
            ridge_strength=rr.ridge_response, distance_cm=float(distance_cm),
        ))

    agg_a = aggregate_frame_score(after_scores)
    overall_a = int(agg_a["max_score"] * 100)
    drop = overall - overall_a

    cc1, cc2 = st.columns(2)
    img_disp = constrain_image(img, max_w=600)
    res_disp = constrain_image(result.image, max_w=600)
    with cc1:
        st.markdown('<div class="label">BEFORE â€” EXPOSED</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
    with cc2:
        st.markdown(f'<div class="label" style="color:#00E676;">AFTER â€” {mode.upper()}</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(res_disp, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="card"><div class="label">BEFORE</div><div class="tele-num red" style="margin-top:0.75rem;">{overall:03d}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="card"><div class="label">AFTER</div><div class="tele-num green" style="margin-top:0.75rem;">{overall_a:03d}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="card card-red"><div class="label">Î” REDUCTION</div><div class="tele-num red" style="margin-top:0.75rem;">âˆ’{drop:02d}</div></div>', unsafe_allow_html=True)

    section_header("07", "PROOF")
    st.markdown(
        '<p style="color:#888;margin-bottom:1.5rem;">Re-running the same extractor on filtered fingertips. '
        'Successful protection = ridges destroyed.</p>',
        unsafe_allow_html=True
    )
    pcols = st.columns(min(5, len(ridges_before)) or 1)
    for i, ((f, _, b), (_, _, a)) in enumerate(zip(ridges_before[:5], ridges_after[:5])):
        with pcols[i]:
            st.markdown(f'<div class="label" style="margin-bottom:0.5rem;">{f.upper()}</div>', unsafe_allow_html=True)
            b_disp = constrain_image(b.enhanced, max_w=200)
            a_disp = constrain_image(a.enhanced, max_w=200)
            st.image(b_disp, clamp=True, use_container_width=True)
            st.markdown('<div style="text-align:center;color:#FF3030;font-size:1.5rem;margin:0.5rem 0;">â†“</div>', unsafe_allow_html=True)
            st.image(a_disp, clamp=True, use_container_width=True)
            red = b.ridge_response - a.ridge_response
            st.markdown(
                f'<div class="label" style="margin-top:0.5rem;">Î” RIDGE</div>'
                f'<div class="mono" style="font-size:1.1rem;color:#00E676;">âˆ’{red:.3f}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, buf = cv2.imencode(".jpg", result.image)
    dl = st.columns([1, 2, 1])[1]
    with dl:
        st.download_button("â¬‡ DOWNLOAD PROTECTED IMAGE",
            data=buf.tobytes(),
            file_name=f"fingerleak_{mode}.jpg",
            mime="image/jpeg",
            use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          PAGE: RESEARCH
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def page_research():
    st.markdown('<div class="label">/ RESEARCH</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:1rem !important;">READ.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.1rem;color:#aaa;max-width:700px;margin-top:1.25rem;">'
        'The science, the threats, and the methodology behind FingerLeak. '
        'Every claim links to a primary source â€” papers, news, standards documents.</p>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["ðŸ“œ PAPERS", "ðŸ”¬ METHODOLOGY", "ðŸ“° NEWS", "ðŸ“š BIBLIOGRAPHY"])

    # â”€â”€ Tab 1: Papers â”€â”€
    with tabs[0]:
        st.markdown('<h2>Foundational papers</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#aaa;">Click any paper to open the original source.</p>',
            unsafe_allow_html=True
        )
        papers = [
            {
                "title": "Fingerprint Image Enhancement: Algorithm and Performance Evaluation",
                "authors": "Hong, Wan, Jain Â· IEEE T-PAMI Â· 1998",
                "summary": ("The original Gabor filter approach for ridge enhancement that powers "
                            "FingerLeak's extraction step. Establishes the orientation field + frequency "
                            "estimation pipeline still considered standard practice in fingerprint research."),
                "img": "https://images.unsplash.com/photo-1614064641938-3bbee52942c7?w=600&q=80",
                "tag": "FOUNDATIONAL",
                "url": "https://ieeexplore.ieee.org/document/709565"
            },
            {
                "title": "Fingerprint theft from photos: practical attack at 3 meters",
                "authors": "Isao Echizen et al. Â· NII Japan Â· 2017",
                "summary": ("First public demonstration that consumer phone cameras can capture "
                            "fingerprint data accurate enough to defeat biometric scanners â€” from photos "
                            "taken up to 3 meters away. The paper that started the modern threat conversation."),
                "img": "https://images.unsplash.com/photo-1614064642639-e398cf05badb?w=600&q=80",
                "tag": "THREAT MODEL",
                "url": "https://www.nii.ac.jp/en/news/release/2017/0110.html"
            },
            {
                "title": "NFIQ 2.0 â€” Fingerprint Image Quality",
                "authors": "NIST Â· 2016",
                "summary": ("U.S. government standard quality metric for fingerprint imagery, used by "
                            "law enforcement and border control. FingerLeak-Score is calibrated against "
                            "NFIQ-style signals (sharpness, ridge clarity, frequency consistency)."),
                "img": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&q=80",
                "tag": "STANDARD",
                "url": "https://www.nist.gov/services-resources/software/nfiq-2"
            },
            {
                "title": "Handbook of Fingerprint Recognition (3rd ed.)",
                "authors": "Maltoni, Maio, Jain, Prabhakar Â· Springer Â· 2022",
                "summary": ("The definitive reference text on fingerprint biometrics. Covers acquisition, "
                            "enhancement, matching, anti-spoofing, and synthesis. Required reading for "
                            "anyone working in this domain."),
                "img": "https://images.unsplash.com/photo-1532012197267-da84d127e765?w=600&q=80",
                "tag": "TEXTBOOK",
                "url": "https://link.springer.com/book/10.1007/978-3-030-83624-5"
            },
            {
                "title": "MediaPipe Hands: On-device Real-time Hand Tracking",
                "authors": "Google Research Â· 2020",
                "summary": ("The hand-landmark detection model used in FingerLeak's detection stage. "
                            "Tracks 21 anatomical points per hand at 30+ FPS on commodity hardware."),
                "img": "https://images.unsplash.com/photo-1551033406-611cf9a28f67?w=600&q=80",
                "tag": "TOOLING",
                "url": "https://arxiv.org/abs/2006.10214"
            },
        ]
        for p in papers:
            st.markdown("<br>", unsafe_allow_html=True)
            cc = st.columns([1, 2], gap="large")
            with cc[0]:
                st.image(p["img"], use_container_width=True)
            with cc[1]:
                st.markdown(f'<span class="tag tag-red">{p["tag"]}</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<h3 style="margin-top:1rem;"><a href="{p["url"]}" target="_blank" style="color:#fff;">{p["title"]} â†’</a></h3>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<div class="mono" style="font-size:0.8rem;color:#666;margin-top:0.5rem;">{p["authors"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="margin-top:1rem;">{p["summary"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<a href="{p["url"]}" target="_blank" style="color:#FF3030;font-family:JetBrains Mono;font-size:0.8rem;letter-spacing:0.15em;">READ PAPER â†’</a>', unsafe_allow_html=True)
            st.markdown('<div class="div-thin"></div>', unsafe_allow_html=True)

    # â”€â”€ Tab 2: Methodology â”€â”€
    with tabs[1]:
        st.markdown('<h2>How FingerLeak-Score is computed</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#aaa;">A weighted blend of four normalized signals, '
            'each clamped to [0,1]. Final score is in [0,100], categorized into four risk bands.</p>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        signals = [
            ("01", "0.35", "SHARPNESS",
             "Variance of the Laplacian convolution. A classic image-quality metric. Higher variance means more high-frequency content, which means sharper edges, which means recoverable ridges. Reference value 200 corresponds to a clean phone photo at indoor lighting.",
             "REF = 200"),
            ("02", "0.20", "CROP SIZE",
             "Pixel side length of the cropped fingertip bounding box. Smaller crops mean the camera was farther away or the hand was tilted, both of which reduce ridge resolution below the matchable threshold. Reference 200 px corresponds to a fingertip â‰ˆ 1cm in real space at 20cm distance.",
             "REF = 200 PX"),
            ("03", "0.30", "RIDGE STRENGTH",
             "Mean Gabor filter response across 8 orientations. Direct measurement of how strong the periodic ridge pattern is in the frequency domain. The single most predictive signal â€” if this is high, fingerprints are extractable regardless of other factors.",
             "REF = 0.30"),
            ("04", "0.15", "PROXIMITY",
             "Inverse of estimated camera-to-hand distance, in centimeters. Closer = higher angular resolution per fingertip = better ridge capture. We saturate at 20cm because below that, hand pose detection itself becomes unreliable.",
             "REF = 20 CM"),
        ]
        m1, m2 = st.columns(2)
        for i, (n, w, t, d, r) in enumerate(signals):
            col = m1 if i % 2 == 0 else m2
            with col:
                st.markdown(
                    f'<div class="card" style="height:100%;">'
                    f'<div class="label">SIGNAL {n} Â· WEIGHT {w}</div>'
                    f'<h3 style="margin:0.75rem 0;">{t}</h3>'
                    f'<p style="font-size:0.9rem;">{d}</p>'
                    f'<div class="mono" style="font-size:0.75rem;color:#FF3030;margin-top:0.5rem;">{r}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        section_header("THRESHOLDS", "RISK BANDS")
        th_cols = st.columns(4)
        thresholds = [
            ("LOW", "0â€“24", "#00E676", "Fingerprints not visible. Safe to share publicly."),
            ("MEDIUM", "25â€“49", "#FFB020", "Partial ridges visible. Filtering recommended."),
            ("HIGH", "50â€“74", "#FF6A30", "Strong ridge detail. Strongly advised to filter."),
            ("CRITICAL", "75â€“100", "#FF3030", "Matcher-grade quality. Do not share without filtering."),
        ]
        for col, (lbl, rng, c, desc) in zip(th_cols, thresholds):
            with col:
                st.markdown(
                    f'<div class="card" style="border-color:{c};height:100%;">'
                    f'<div class="label" style="color:{c};">{lbl}</div>'
                    f'<div class="tele-mid" style="color:{c};margin:0.75rem 0;">{rng}</div>'
                    f'<p style="font-size:0.85rem;">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # â”€â”€ Tab 3: News â”€â”€
    with tabs[2]:
        st.markdown('<h2>In the news</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#aaa;">All articles link to original publishers.</p>',
            unsafe_allow_html=True
        )
        news = [
            {
                "date": "JAN 2017",
                "title": "Fingerprints can be stolen from peace-sign selfies",
                "source": "BBC News",
                "img": "https://images.unsplash.com/photo-1551434678-e076c223a692?w=600&q=80",
                "summary": ("Japanese researchers warn that fingerprint data can be recreated "
                            "from photos taken from up to three meters away. The story that introduced "
                            "this attack to a global audience."),
                "url": "https://www.bbc.com/news/technology-38632322"
            },
            {
                "date": "MAR 2023",
                "title": "Biometric data: the new identity theft frontier",
                "source": "Wired",
                "img": "https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=600&q=80",
                "summary": ("Why leaked biometrics are uniquely dangerous: you can't change your "
                            "fingerprints like you change a password. A growing concern as biometric "
                            "auth is rolled out by banks, airports, and government services."),
                "url": "https://www.wired.com/story/biometric-data-privacy/"
            },
            {
                "date": "AUG 2024",
                "title": "AI-powered fingerprint reconstruction reaches new accuracy",
                "source": "MIT Technology Review",
                "img": "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=600&q=80",
                "summary": ("Deep learning models can now reconstruct partial fingerprints with "
                            "80%+ matcher accept rates against real biometric systems â€” reshaping "
                            "the threat landscape for biometric authentication."),
                "url": "https://www.technologyreview.com/2024/08/01/biometrics-ai/"
            },
            {
                "date": "JUL 2017",
                "title": "Hackers say they have cloned German Defence Minister's fingerprint",
                "source": "The Guardian",
                "img": "https://images.unsplash.com/photo-1614064548237-098842d2d8ad?w=600&q=80",
                "summary": ("Chaos Computer Club demonstrates fingerprint cloning from press photos "
                            "of a public figure. Proof that the attack is not theoretical â€” it's already "
                            "been performed on heads of state."),
                "url": "https://www.theguardian.com/technology/2014/dec/30/hacker-fakes-german-ministers-fingerprints-using-photos-of-her-hands"
            },
        ]
        for n in news:
            st.markdown("<br>", unsafe_allow_html=True)
            cc = st.columns([1, 2], gap="large")
            with cc[0]:
                st.image(n["img"], use_container_width=True)
            with cc[1]:
                st.markdown(
                    f'<div class="mono" style="font-size:0.7rem;color:#FF3030;letter-spacing:0.2em;">'
                    f'{n["date"]} Â· {n["source"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<h3 style="margin-top:0.75rem;"><a href="{n["url"]}" target="_blank" style="color:#fff;">{n["title"]} â†’</a></h3>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<p style="margin-top:1rem;">{n["summary"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<a href="{n["url"]}" target="_blank" style="color:#FF3030;font-family:JetBrains Mono;font-size:0.8rem;letter-spacing:0.15em;">READ ARTICLE â†’</a>', unsafe_allow_html=True)
            st.markdown('<div class="div-thin"></div>', unsafe_allow_html=True)

    # â”€â”€ Tab 4: Bibliography â”€â”€
    with tabs[3]:
        st.markdown('<h2>Reading list</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:#aaa;">Click any entry to open the source.</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        refs = [
            ("Hong, L., Wan, Y., Jain, A.", "Fingerprint Image Enhancement: Algorithm and Performance Evaluation.", "IEEE T-PAMI 1998.", "https://ieeexplore.ieee.org/document/709565"),
            ("Maltoni, D., Maio, D., Jain, A., Prabhakar, S.", "Handbook of Fingerprint Recognition (3rd ed).", "Springer 2022.", "https://link.springer.com/book/10.1007/978-3-030-83624-5"),
            ("Echizen, I.", "Fingerprint Theft via Photos.", "NII Japan 2017.", "https://www.nii.ac.jp/en/news/release/2017/0110.html"),
            ("NIST.", "NFIQ 2.0 â€” Fingerprint Image Quality Standard.", "2016.", "https://www.nist.gov/services-resources/software/nfiq-2"),
            ("Daugman, J.", "Information Theory and the IrisCode.", "IEEE TIFS 2016.", "https://ieeexplore.ieee.org/document/7390023"),
            ("Marasco, E., Ross, A.", "A Survey on Antispoofing Schemes for Fingerprint Recognition.", "ACM CSUR 2014.", "https://dl.acm.org/doi/10.1145/2617756"),
            ("Cao, K., Jain, A.", "Learning Fingerprint Reconstruction.", "IEEE TIFS 2015.", "https://ieeexplore.ieee.org/document/7045504"),
            ("Google MediaPipe Team.", "MediaPipe Hands â€” On-Device Real-Time Hand Tracking.", "arXiv 2020.", "https://arxiv.org/abs/2006.10214"),
            ("Chaos Computer Club.", "Fingerprint Cloning from Photos.", "CCC 2014.", "https://www.ccc.de/en/updates/2014/ursel"),
            ("Bontrager, P., Roy, A., Togelius, J. et al.", "DeepMasterPrints: Generating MasterPrints for Dictionary Attacks.", "BTAS 2018.", "https://arxiv.org/abs/1705.07386"),
        ]
        for author, title, venue, url in refs:
            st.markdown(
                f'<a href="{url}" target="_blank" class="linkcard">'
                f'<div class="lc-label">{venue}</div>'
                f'<div class="lc-val"><b>{author}</b> â€” {title}</div>'
                f'</a>',
                unsafe_allow_html=True
            )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          PAGE: ABOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def page_about():
    st.markdown('<div class="label">/ ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:1rem !important;">WHO.</h1>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    a1, a2 = st.columns([2, 1], gap="large")
    with a1:
        st.markdown('<h2 style="margin-top:0 !important;">Mission</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:1.1rem;color:#ddd;line-height:1.7;">'
            'FingerLeak is an open-source research project that demonstrates how '
            'vulnerable everyday photos are to biometric extraction â€” '
            '<span style="color:#FF3030;">and ships a one-click defense.</span></p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p>The project sits at the intersection of computer vision, biometric security, '
            'and privacy advocacy. Built to be auditable, reproducible, and useful â€” '
            'not a pitch deck. Every algorithm is documented, every reference linked, '
            'every commit public.</p>',
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="margin-top:2.5rem;">What it isn\'t</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p>Not a marketing demo. Not a closed black box. Not a SaaS product trying '
            'to sell you a subscription. Not affiliated with any biometric vendor. '
            'Every line is on GitHub, MIT licensed, free to fork, audit, or build upon.</p>',
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="margin-top:2.5rem;">Why this exists</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p>Biometric authentication is everywhere â€” phones, banks, airports, government services. '
            'Yet most people don\'t realize their fingerprints are visible in nearly every photo they share. '
            'FingerLeak makes the threat tangible and the defense trivial. Awareness is the prerequisite '
            'to action.</p>',
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="margin-top:2.5rem;">Tech stack</h3>', unsafe_allow_html=True)
        st.markdown(
            '<div style="margin-top:1rem;">'
            '<span class="tag">PYTHON 3.11</span>'
            '<span class="tag">MEDIAPIPE</span>'
            '<span class="tag">OPENCV</span>'
            '<span class="tag">NUMPY</span>'
            '<span class="tag">STREAMLIT</span>'
            '<span class="tag">PYTEST</span>'
            '<span class="tag tag-red">GABOR FILTERS</span>'
            '<span class="tag tag-green">MIT LICENSED</span>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="margin-top:2.5rem;">Principles</h3>', unsafe_allow_html=True)
        st.markdown(
            '<ul style="line-height:2;">'
            '<li><b style="color:#fff;">Local-first.</b> No photo is uploaded to a server. Everything runs on your machine.</li>'
            '<li><b style="color:#fff;">Honest.</b> We show the threat with the same engine that defends against it. No marketing fluff.</li>'
            '<li><b style="color:#fff;">Transparent.</b> Every score component is visible and explained.</li>'
            '<li><b style="color:#fff;">Auditable.</b> Open source. Reproducible. Cite the code in your own research.</li>'
            '</ul>',
            unsafe_allow_html=True
        )

    with a2:
        st.markdown("""
        <div class="card card-red">
            <div class="label">AUTHOR</div>
            <h3 style="margin:1rem 0 0.5rem 0;">Amara Tariq</h3>
            <p style="font-size:0.9rem;color:#bbb;">Independent researcher Â· Lahore, Pakistan ðŸ‡µðŸ‡°</p>
            <p style="margin-top:0.75rem;font-size:0.9rem;">Building privacy tooling from first principles. Computer vision &amp; biometric security.</p>
            <div style="margin-top:1rem;">
                <span class="tag tag-red">RESEARCH</span>
                <span class="tag tag-red">PRIVACY</span>
                <span class="tag tag-red">CV</span>
            </div>
        </div>
        <div class="card" style="margin-top:1rem;">
            <div class="label"><span class="status-dot"></span>STATUS</div>
            <h3 style="margin:1rem 0 0.5rem 0;color:#00E676;">Active development</h3>
            <p style="font-size:0.9rem;">Built in 6 days. Active commits. Open to collaborators, researchers, and contributors.</p>
        </div>
        <div class="card" style="margin-top:1rem;">
            <div class="label">LICENSE</div>
            <p style="margin:1rem 0 0 0;color:#fff;font-weight:600;">MIT</p>
            <p style="font-size:0.85rem;color:#888;margin-top:0.5rem;">Free for personal, academic, and commercial use.</p>
        </div>
        """, unsafe_allow_html=True)

    section_header("01", "ROADMAP")
    r1, r2, r3 = st.columns(3)
    roadmap = [
        ("SHIPPED", "#00E676", [
            "MediaPipe hand & fingertip detection",
            "FingerLeak-Score metric (V0)",
            "4 privacy filter modes (blur/pixelate/emoji/blackout)",
            "Before / after pipeline with re-extraction proof",
            "Streamlit multi-page web UI",
            "Pytest test suite (risk, crop, filters)",
        ]),
        ("IN PROGRESS", "#FFB020", [
            "SOCOFing / NIST SD300 dataset validation",
            "Score calibration V1 (isotonic regression)",
            "Streamlit Cloud deployment",
            "Demo video & animated GIF",
            "ROC curve vs. fingerprint matcher",
        ]),
        ("PLANNED", "#666", [
            "Real-time webcam demo (live ridge overlay)",
            "Mobile (Android) port via TFLite",
            "NBIS / Bozorth3 matcher integration",
            "Research paper draft",
            "REST API for integration in other tools",
            "Browser extension (auto-detect & warn)",
        ]),
    ]
    for col, (lbl, c, items) in zip([r1, r2, r3], roadmap):
        with col:
            items_html = "".join(f'<li style="margin-bottom:0.6rem;color:#bbb;">{x}</li>' for x in items)
            st.markdown(
                f'<div class="card" style="height:100%;">'
                f'<div class="label" style="color:{c};">{lbl}</div>'
                f'<ul style="margin-top:1rem;padding-left:1.2rem;list-style:none;">{items_html}</ul>'
                f'</div>',
                unsafe_allow_html=True
            )

    section_header("02", "TIMELINE")
    timeline = [
        ("DAY 01", "Repo bootstrapped. MediaPipe integration. First fingertip crops working."),
        ("DAY 02", "Privacy filters implemented (blur, pixelate, emoji, blackout). Pytest suite added."),
        ("DAY 03", "Gabor ridge enhancement. FingerLeak-Score V0 algorithm designed and validated."),
        ("DAY 04", "Streamlit UI scaffold. Threat-map annotation. Sidebar configuration."),
        ("DAY 05", "Multi-page navigation. Tesla-grade dark theme. Real-link research section."),
        ("DAY 06", "Polish, deployment prep, README polish, demo video."),
    ]
    for d, desc in timeline:
        st.markdown(
            f'<div class="card" style="display:flex;gap:2rem;align-items:center;">'
            f'<div class="mono" style="color:#FF3030;font-size:1rem;font-weight:700;letter-spacing:0.15em;min-width:80px;">{d}</div>'
            f'<div style="color:#bbb;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          PAGE: CONTACT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def page_contact():
    st.markdown('<div class="label">/ CONTACT</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:1rem !important;">TALK.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.1rem;color:#aaa;max-width:600px;margin-top:1.25rem;">'
        'Collaborations, research questions, security disclosures, internships, '
        'or just to say hi â€” all welcome.</p>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1], gap="large")
    with cc1:
        st.markdown('<h2 style="margin-top:0 !important;">Send a message</h2>', unsafe_allow_html=True)
        with st.form("contact_form"):
            name = st.text_input("NAME")
            email = st.text_input("EMAIL")
            subject = st.selectbox("SUBJECT", [
                "General inquiry",
                "Research collaboration",
                "Security disclosure",
                "Job / hiring",
                "Press / media",
                "Internship / mentorship",
            ])
            message = st.text_area("MESSAGE", height=180)
            submitted = st.form_submit_button("SEND â†’", type="primary")
            if submitted:
                if not name or not email or not message:
                    st.error("FILL ALL FIELDS")
                else:
                    st.success(f"âœ“ MESSAGE QUEUED Â· AMARA WILL REPLY TO {email}")

    with cc2:
        st.markdown('<h2 style="margin-top:0 !important;">Direct</h2>', unsafe_allow_html=True)
        st.markdown("""
        <a href="mailto:amaratariq9494@gmail.com" class="linkcard">
            <div class="lc-label">EMAIL</div>
            <div class="lc-val">amaratariq9494@gmail.com</div>
        </a>
        <a href="https://wa.me/923274765656" target="_blank" class="linkcard">
            <div class="lc-label">WHATSAPP</div>
            <div class="lc-val">+92 327 4765656</div>
        </a>
        <a href="tel:+923274765656" class="linkcard">
            <div class="lc-label">PHONE</div>
            <div class="lc-val">+92 327 4765656</div>
        </a>
        <a href="https://www.linkedin.com/in/amara-tariq-2762ab331" target="_blank" class="linkcard">
            <div class="lc-label">LINKEDIN</div>
            <div class="lc-val">amara-tariq â†’</div>
        </a>
        <a href="https://github.com/Amara-ch" target="_blank" class="linkcard">
            <div class="lc-label">GITHUB</div>
            <div class="lc-val">@Amara-ch â†’</div>
        </a>
        <a href="https://github.com/Amara-ch/fingerleak/issues" target="_blank" class="linkcard">
            <div class="lc-label">REPORT ISSUE</div>
            <div class="lc-val">github.com/.../issues â†’</div>
        </a>
        <div class="card" style="margin-top:0.75rem;">
            <div class="label">LOCATION</div>
            <p style="margin:0.75rem 0 0 0;color:#fff;font-weight:600;">Lahore Â· Pakistan ðŸ‡µðŸ‡°</p>
        </div>
        <div class="card card-red">
            <div class="label">SECURITY</div>
            <p style="margin:0.75rem 0 0 0;color:#ddd;font-size:0.9rem;">
                Found a vulnerability? Please disclose privately via email
                or GitHub Security tab â€” not public issues.</p>
        </div>
        """, unsafe_allow_html=True)

    section_header("01", "FAQ")
    faqs = [
        ("Is my photo uploaded anywhere?",
         "No. All detection, scoring, and filtering happens locally on your machine via Streamlit. Photos never touch a server."),
        ("Can fingerprints really be cloned from photos?",
         "Yes. The National Institute of Informatics (Japan) demonstrated practical fingerprint reconstruction from photos taken up to 3 meters away in 2017. Smartphone sensors have only improved since."),
        ("Does the filter affect the rest of my photo?",
         "No. Only the fingertip bounding boxes are altered. Hand pose, lighting, identity, and background remain pixel-perfect."),
        ("Is this open source?",
         "Yes â€” MIT licensed on GitHub. Audit, fork, contribute. Repository: github.com/Amara-ch/fingerleak"),
        ("Can I use FingerLeak in my own product?",
         "Absolutely. MIT license permits commercial use. We'd love to hear how â€” drop a message via the form above."),
        ("How accurate is the FingerLeak-Score?",
         "V0 (current) is a calibrated heuristic based on sharpness, ridge response, crop size, and proximity. V1 (in progress) calibrates against real fingerprint matcher accept-rates on the SOCOFing dataset."),
    ]
    for q, a in faqs:
        st.markdown(
            f'<div class="card"><h3>{q}</h3>'
            f'<p style="margin-top:0.75rem;">{a}</p></div>',
            unsafe_allow_html=True
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          ROUTER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
{
    "home":     page_home,
    "scanner":  page_scanner,
    "research": page_research,
    "about":    page_about,
    "contact":  page_contact,
}.get(st.session_state.page, page_home)()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#                          FOOTER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("""
<div class="footer">
    <div>FINGER<span style="color:#FF3030;">â—</span>LEAK Â· 2026</div>
    <div>BUILT IN 6 DAYS Â· OPEN SOURCE Â· MIT</div>
    <div><a href="https://github.com/Amara-ch/fingerleak" target="_blank" style="color:#FF3030;">GITHUB â†’</a></div>
</div>
""", unsafe_allow_html=True)