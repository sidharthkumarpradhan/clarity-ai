import streamlit as st
import numpy as np
import time
import os
import sys
import cv2
from PIL import Image
import io

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

LOGO_PATH = os.path.join(WORKSPACE, "attached_assets", "fairfieldUniversityLogo_1772082834920.png")
MIRNET_WEIGHTS = os.path.join(WORKSPACE, "models", "mirnet_finetuned.pth")
ZERODCE_WEIGHTS = os.path.join(WORKSPACE, "models", "zero_dce.pth")

HF_MIRNET_URL = "https://huggingface.co/dblasko/mirnet-low-light-img-enhancement/resolve/main/mirnet_finetuned.pth"
ZERODCE_URL = "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"

MODELS = {
    "mirnet": {
        "name": "MIRNet",
        "full_name": "Multi-scale Residual Block Network",
        "description": "Deep learning model using multi-scale residual blocks and dual attention for low-light image enhancement. Processes features at multiple resolutions for high-quality restoration.",
        "paper": "Learning Enriched Features for Real Image Restoration and Enhancement (ECCV 2020)",
        "speed": "Slow (~30s on CPU)",
        "quality": "High",
        "type": "Deep Learning",
    },
    "zero_dce": {
        "name": "Zero-DCE",
        "full_name": "Zero-Reference Deep Curve Estimation",
        "description": "Lightweight model that estimates image-specific tonal curves for dynamic range adjustment. Uses zero-reference learning - no paired training data needed.",
        "paper": "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)",
        "speed": "Fast (~0.5s on CPU)",
        "quality": "Good",
        "type": "Deep Learning",
    },
    "clahe": {
        "name": "CLAHE",
        "full_name": "Contrast Limited Adaptive Histogram Equalization",
        "description": "Classic computer vision technique that enhances local contrast while limiting noise amplification. Divides image into tiles and applies adaptive equalization.",
        "paper": "Adaptive Histogram Equalization and Its Variations (1987)",
        "speed": "Instant (<0.1s)",
        "quality": "Moderate",
        "type": "Traditional CV",
    },
    "histogram_eq": {
        "name": "Histogram Equalization",
        "full_name": "Global Histogram Equalization",
        "description": "Spreads out the most frequent intensity values in an image to enhance overall contrast. Simple but effective baseline for uniform illumination correction.",
        "paper": "Digital Image Processing (Gonzalez & Woods)",
        "speed": "Instant (<0.1s)",
        "quality": "Basic",
        "type": "Traditional CV",
    },
    "gamma_correction": {
        "name": "Gamma Correction",
        "full_name": "Adaptive Gamma Correction",
        "description": "Applies a power-law transformation to brighten dark images. Automatically estimates optimal gamma value based on image brightness statistics.",
        "paper": "Power-Law (Gamma) Transformations",
        "speed": "Instant (<0.1s)",
        "quality": "Basic",
        "type": "Traditional CV",
    },
}


def download_weights(url, path, name):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    progress = st.progress(0, text=f"Downloading {name} weights...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=120)
        total = int(resp.headers.get("Content-Length", 0))
        data = b""
        block_size = 1024 * 256
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            data += chunk
            if total > 0:
                progress.progress(len(data) / total, text=f"Downloading {name} weights... {len(data) / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB")
        with open(path, "wb") as f:
            f.write(data)
        progress.empty()
        return True
    except Exception as e:
        progress.empty()
        st.error(f"Failed to download {name} weights: {e}")
        return False


@st.cache_resource
def load_mirnet():
    import torch
    from server.model.MIRNet.model import MIRNet

    if not os.path.exists(MIRNET_WEIGHTS):
        if not download_weights(HF_MIRNET_URL, MIRNET_WEIGHTS, "MIRNet"):
            return None

    device = torch.device("cpu")
    model = MIRNet().to(device)
    checkpoint = torch.load(MIRNET_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_zero_dce():
    import torch
    from server.model.ZeroDCE.model import ZeroDCE

    if not os.path.exists(ZERODCE_WEIGHTS):
        if not download_weights(ZERODCE_URL, ZERODCE_WEIGHTS, "Zero-DCE"):
            return None

    device = torch.device("cpu")
    model = ZeroDCE().to(device)
    checkpoint = torch.load(ZERODCE_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def enhance_mirnet(image_np, max_size=400):
    import torch
    import torchvision.transforms as T

    model = load_mirnet()
    if model is None:
        return None, "Failed to load MIRNet model"

    pil_img = Image.fromarray(image_np)
    original_size = pil_img.size

    transform = T.Compose([T.Resize(max_size), T.ToTensor()])
    img_tensor = transform(pil_img).unsqueeze(0)

    if img_tensor.shape[2] % 8 != 0:
        img_tensor = img_tensor[:, :, :-(img_tensor.shape[2] % 8), :]
    if img_tensor.shape[3] % 8 != 0:
        img_tensor = img_tensor[:, :, :, :-(img_tensor.shape[3] % 8)]

    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    output = output.clamp(0, 1)
    elapsed = time.time() - start

    output_img = T.ToPILImage()(output.squeeze(0).cpu())
    output_img = output_img.resize(original_size, Image.LANCZOS)
    return np.array(output_img), elapsed


def enhance_zero_dce(image_np):
    import torch
    import torchvision.transforms as T

    model = load_zero_dce()
    if model is None:
        return None, "Failed to load Zero-DCE model"

    pil_img = Image.fromarray(image_np)
    original_size = pil_img.size

    max_dim = 1200
    w, h = original_size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    transform = T.ToTensor()
    img_tensor = transform(pil_img).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    elapsed = time.time() - start

    output_img = T.ToPILImage()(output.squeeze(0).cpu())
    if output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.LANCZOS)
    return np.array(output_img), elapsed


def enhance_clahe(image_np, clip_limit=3.0, tile_size=8):
    start = time.time()
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    elapsed = time.time() - start
    return result, elapsed


def enhance_histogram_eq(image_np):
    start = time.time()
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    y_eq = cv2.equalizeHist(y_channel)

    merged = cv2.merge((y_eq, cr_channel, cb_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    elapsed = time.time() - start
    return result, elapsed


def enhance_gamma(image_np):
    start = time.time()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 10:
        gamma = 0.2
    elif mean_brightness < 50:
        gamma = 0.4
    elif mean_brightness < 100:
        gamma = 0.6
    elif mean_brightness < 128:
        gamma = 0.8
    else:
        gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    result = cv2.LUT(image_np, table)
    elapsed = time.time() - start
    return result, elapsed


ENHANCE_FUNCTIONS = {
    "mirnet": enhance_mirnet,
    "zero_dce": enhance_zero_dce,
    "clahe": enhance_clahe,
    "histogram_eq": enhance_histogram_eq,
    "gamma_correction": enhance_gamma,
}


def get_image_bytes(image_np, format="PNG"):
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


st.set_page_config(
    page_title="ClarityAI - AI-Powered Image Enhancement",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { }
    .main-header {
        background: linear-gradient(135deg, #a01c2a 0%, #7c1520 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }
    .model-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .model-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-dl { background: #dbeafe; color: #1e40af; }
    .badge-cv { background: #dcfce7; color: #166534; }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .credit-section {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
    div[data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=220)
    st.markdown("---")

    st.markdown("### Model Selection")

    mode = st.radio(
        "Enhancement Mode",
        ["Single Model", "Compare Models"],
        help="Single mode enhances with one model. Compare mode shows side-by-side results from multiple models.",
    )

    if mode == "Single Model":
        selected_model = st.selectbox(
            "Choose Model",
            list(MODELS.keys()),
            format_func=lambda x: f"{MODELS[x]['name']} ({MODELS[x]['type']})",
        )

        info = MODELS[selected_model]
        st.markdown(f"""
        **{info['full_name']}**

        {info['description']}

        - **Speed:** {info['speed']}
        - **Quality:** {info['quality']}
        - **Type:** {info['type']}
        - **Paper:** _{info['paper']}_
        """)
    else:
        compare_models = st.multiselect(
            "Select Models to Compare",
            list(MODELS.keys()),
            default=["mirnet", "zero_dce", "clahe"],
            format_func=lambda x: f"{MODELS[x]['name']}",
        )
        if len(compare_models) < 2:
            st.warning("Select at least 2 models to compare.")

    st.markdown("---")

    st.markdown("### Advanced Settings")
    if mode == "Single Model" and selected_model == "clahe":
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 3.0, 0.5)
        clahe_tile = st.slider("CLAHE Tile Size", 2, 16, 8, 2)
    else:
        clahe_clip = 3.0
        clahe_tile = 8

    max_resolution = st.selectbox(
        "MIRNet Max Processing Resolution",
        [200, 300, 400, 512],
        index=2,
        help="Higher resolution = better quality but slower. Affects MIRNet only.",
    )

    st.markdown("---")

    st.markdown("""
    <div class="credit-section">
        <p style="font-weight: 600; color: #a01c2a; margin-bottom: 0.5rem;">Presented by</p>
        <p style="font-weight: 600; margin: 0.15rem 0;">Sidharth Kumar Pradhan</p>
        <p style="font-weight: 600; margin: 0.15rem 0;">Naqibahmed Kadri</p>
        <hr style="margin: 0.75rem 0; border-color: #fecaca;">
        <p style="font-size: 0.8rem; color: #666; margin-bottom: 0.15rem;">Guided by</p>
        <p style="font-weight: 600; margin: 0.15rem 0;">Dr. Sidike Paheding</p>
        <p style="font-size: 0.8rem; color: #888;">School of Engineering & Computing<br>Fairfield University</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>ClarityAI</h1>
    <p>AI-Powered Low-Light Image Enhancement</p>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
with col_info1:
    st.metric("Models", "5")
with col_info2:
    st.metric("DL Models", "2")
with col_info3:
    st.metric("CV Methods", "3")
with col_info4:
    st.metric("MIRNet", "ECCV 2020")
with col_info5:
    st.metric("Zero-DCE", "CVPR 2020")


st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a dark or low-light image to enhance. Supports JPEG, PNG, BMP, and WebP formats.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.markdown("### Original Image")
    st.image(image_np, caption=f"Original - {uploaded_file.name} ({image_np.shape[1]}x{image_np.shape[0]})", use_container_width=True)

    mean_brightness = np.mean(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY))
    brightness_label = "Very Dark" if mean_brightness < 50 else "Dark" if mean_brightness < 100 else "Medium" if mean_brightness < 150 else "Bright"

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.metric("Resolution", f"{image_np.shape[1]}x{image_np.shape[0]}")
    with bc2:
        st.metric("Mean Brightness", f"{mean_brightness:.0f}/255")
    with bc3:
        st.metric("Assessment", brightness_label)

    st.markdown("---")

    if mode == "Single Model":
        if st.button(f"Enhance with {MODELS[selected_model]['name']}", type="primary", use_container_width=True):
            with st.spinner(f"Enhancing with {MODELS[selected_model]['name']}... This may take a moment."):
                try:
                    if selected_model == "mirnet":
                        result, elapsed = enhance_mirnet(image_np, max_size=max_resolution)
                    elif selected_model == "clahe":
                        result, elapsed = enhance_clahe(image_np, clip_limit=clahe_clip, tile_size=clahe_tile)
                    else:
                        result, elapsed = ENHANCE_FUNCTIONS[selected_model](image_np)

                    if result is not None:
                        st.success(f"Enhancement complete in {elapsed:.2f}s using {MODELS[selected_model]['name']}")

                        st.markdown("### Enhanced Result")

                        view_mode = st.radio("View Mode", ["Side by Side", "Slider", "Enhanced Only"], horizontal=True)

                        if view_mode == "Side by Side":
                            col_orig, col_enh = st.columns(2)
                            with col_orig:
                                st.image(image_np, caption="Original", use_container_width=True)
                            with col_enh:
                                st.image(result, caption=f"Enhanced ({MODELS[selected_model]['name']})", use_container_width=True)
                        elif view_mode == "Slider":
                            slider_val = st.slider("Original vs Enhanced", 0, 100, 50, help="Slide to compare")
                            blend = cv2.addWeighted(image_np, 1 - slider_val / 100, result, slider_val / 100, 0)
                            st.image(blend, caption=f"Blend: {100 - slider_val}% Original / {slider_val}% Enhanced", use_container_width=True)
                        else:
                            st.image(result, caption=f"Enhanced ({MODELS[selected_model]['name']})", use_container_width=True)

                        rc1, rc2, rc3 = st.columns(3)
                        with rc1:
                            orig_brightness = np.mean(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY))
                            enh_brightness = np.mean(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
                            st.metric("Brightness Change", f"{enh_brightness:.0f}", delta=f"{enh_brightness - orig_brightness:+.0f}")
                        with rc2:
                            st.metric("Processing Time", f"{elapsed:.2f}s")
                        with rc3:
                            st.metric("Model", MODELS[selected_model]['name'])

                        img_bytes = get_image_bytes(result)
                        st.download_button(
                            label="Download Enhanced Image",
                            data=img_bytes,
                            file_name=f"enhanced_{selected_model}_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png",
                            type="primary",
                            use_container_width=True,
                        )
                    else:
                        st.error("Enhancement failed. Please try again.")
                except Exception as e:
                    st.error(f"Enhancement error: {str(e)}")

    else:
        if len(compare_models) >= 2:
            if st.button("Run Comparison", type="primary", use_container_width=True):
                results = {}
                progress = st.progress(0, text="Starting comparison...")

                for i, model_id in enumerate(compare_models):
                    progress.progress(
                        (i) / len(compare_models),
                        text=f"Processing with {MODELS[model_id]['name']}..."
                    )

                    try:
                        if model_id == "mirnet":
                            result, elapsed = enhance_mirnet(image_np, max_size=max_resolution)
                        elif model_id == "clahe":
                            result, elapsed = enhance_clahe(image_np, clip_limit=clahe_clip, tile_size=clahe_tile)
                        else:
                            result, elapsed = ENHANCE_FUNCTIONS[model_id](image_np)

                        if result is not None:
                            enh_brightness = np.mean(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
                            results[model_id] = {
                                "image": result,
                                "time": elapsed,
                                "brightness": enh_brightness,
                            }
                    except Exception as e:
                        st.warning(f"{MODELS[model_id]['name']} failed: {str(e)}")

                progress.progress(1.0, text="Comparison complete!")
                time.sleep(0.5)
                progress.empty()

                if results:
                    st.markdown("### Comparison Results")

                    st.markdown("#### Metrics Overview")
                    metric_cols = st.columns(len(results) + 1)
                    with metric_cols[0]:
                        st.markdown("**Original**")
                        st.metric("Brightness", f"{mean_brightness:.0f}")
                        st.metric("Time", "-")

                    for i, (model_id, data) in enumerate(results.items()):
                        with metric_cols[i + 1]:
                            st.markdown(f"**{MODELS[model_id]['name']}**")
                            st.metric("Brightness", f"{data['brightness']:.0f}", delta=f"{data['brightness'] - mean_brightness:+.0f}")
                            st.metric("Time", f"{data['time']:.2f}s")

                    st.markdown("#### Visual Comparison")

                    num_results = len(results) + 1
                    cols = st.columns(min(num_results, 3))

                    with cols[0]:
                        st.image(image_np, caption="Original", use_container_width=True)

                    for i, (model_id, data) in enumerate(results.items()):
                        col_idx = (i + 1) % min(num_results, 3)
                        with cols[col_idx]:
                            st.image(data["image"], caption=f"{MODELS[model_id]['name']} ({data['time']:.2f}s)", use_container_width=True)

                    if num_results > 3:
                        extra_cols = st.columns(min(num_results - 2, 3))
                        items = list(results.items())[2:]
                        for i, (model_id, data) in enumerate(items):
                            with extra_cols[i % 3]:
                                st.image(data["image"], caption=f"{MODELS[model_id]['name']} ({data['time']:.2f}s)", use_container_width=True)

                    st.markdown("#### Download Results")
                    dl_cols = st.columns(len(results))
                    for i, (model_id, data) in enumerate(results.items()):
                        with dl_cols[i]:
                            img_bytes = get_image_bytes(data["image"])
                            st.download_button(
                                label=f"Download {MODELS[model_id]['name']}",
                                data=img_bytes,
                                file_name=f"enhanced_{model_id}_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                                mime="image/png",
                                use_container_width=True,
                            )

else:
    st.info("Upload a low-light image to get started. The AI models will enhance brightness, contrast, and detail recovery.")

    st.markdown("### Available Models")

    for model_id, info in MODELS.items():
        badge_class = "badge-dl" if info["type"] == "Deep Learning" else "badge-cv"
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>{info['name']}</strong>
                <span class="model-badge {badge_class}">{info['type']}</span>
            </div>
            <p style="color: #666; font-size: 0.85rem; margin: 0.5rem 0 0 0;">{info['description']}</p>
            <p style="color: #999; font-size: 0.75rem; margin: 0.25rem 0 0 0;">Speed: {info['speed']} | Quality: {info['quality']}</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem 0;">
    <p>ClarityAI - AI-Powered Low-Light Image Enhancement</p>
    <p>Powered by MIRNet & Zero-DCE via PyTorch | CLAHE, Histogram Equalization & Gamma Correction via OpenCV</p>
    <p>School of Engineering & Computing, Fairfield University</p>
</div>
""", unsafe_allow_html=True)
