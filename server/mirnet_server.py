import sys
import os
import json
import logging
import time
import numpy as np
import cv2
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [enhance] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger('enhance')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRNET_WEIGHTS = os.path.join(WORKSPACE, 'models', 'mirnet_finetuned.pth')
ZERODCE_WEIGHTS = os.path.join(WORKSPACE, 'models', 'zero_dce.pth')

HF_MIRNET_URL = "https://huggingface.co/dblasko/mirnet-low-light-img-enhancement/resolve/main/mirnet_finetuned.pth"
ZERODCE_URL = "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"

sys.path.insert(0, WORKSPACE)


def download_weights(url, path, name):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Downloading {name} weights from {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=120)
        data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"Downloaded {name} weights: {len(data)} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to download {name} weights: {e}")
        return False

_models = {}
_device = None


def get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cpu")
    return _device


def load_mirnet():
    if "mirnet" in _models:
        logger.info("Using cached MIRNet model")
        return _models["mirnet"]

    import torch
    from server.model.MIRNet.model import MIRNet

    if not os.path.exists(MIRNET_WEIGHTS):
        if not download_weights(HF_MIRNET_URL, MIRNET_WEIGHTS, "MIRNet"):
            raise FileNotFoundError(f"MIRNet weights not found and download failed")

    logger.info(f"Loading MIRNet model from {MIRNET_WEIGHTS}")
    start = time.time()

    device = get_device()
    model = MIRNet().to(device)
    checkpoint = torch.load(MIRNET_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _models["mirnet"] = model
    logger.info(f"MIRNet loaded in {time.time() - start:.2f}s")
    return model


def load_zero_dce():
    if "zero_dce" in _models:
        logger.info("Using cached Zero-DCE model")
        return _models["zero_dce"]

    import torch
    from server.model.ZeroDCE.model import ZeroDCE

    if not os.path.exists(ZERODCE_WEIGHTS):
        if not download_weights(ZERODCE_URL, ZERODCE_WEIGHTS, "Zero-DCE"):
            raise FileNotFoundError(f"Zero-DCE weights not found and download failed")

    logger.info(f"Loading Zero-DCE model from {ZERODCE_WEIGHTS}")
    start = time.time()

    device = get_device()
    model = ZeroDCE().to(device)
    checkpoint = torch.load(ZERODCE_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    _models["zero_dce"] = model
    logger.info(f"Zero-DCE loaded in {time.time() - start:.2f}s")
    return model


def enhance_mirnet(input_path, output_path, max_size=400):
    import torch
    import torchvision.transforms as T

    model = load_mirnet()
    device = get_device()

    logger.info(f"MIRNet processing: {input_path}")
    start = time.time()

    original = Image.open(input_path).convert('RGB')
    original_width, original_height = original.size

    transform = T.Compose([T.Resize(max_size), T.ToTensor()])
    img_tensor = transform(original).unsqueeze(0).to(device)

    if img_tensor.shape[2] % 8 != 0:
        img_tensor = img_tensor[:, :, :-(img_tensor.shape[2] % 8), :]
    if img_tensor.shape[3] % 8 != 0:
        img_tensor = img_tensor[:, :, :, :-(img_tensor.shape[3] % 8)]

    infer_start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    output = output.clamp(0, 1)
    infer_time = time.time() - infer_start

    output_img = T.ToPILImage()(output.squeeze(0).cpu())
    output_img = output_img.resize((original_width, original_height), Image.LANCZOS)
    output_img.save(output_path, 'PNG')

    total_time = time.time() - start
    logger.info(f"MIRNet done: inference={infer_time:.2f}s, total={total_time:.2f}s")

    return {
        "success": True,
        "model": "mirnet",
        "input_dimensions": f"{original_width}x{original_height}",
        "output_path": output_path,
        "output_size": os.path.getsize(output_path),
        "inference_time": round(infer_time, 2),
        "total_time": round(total_time, 2),
    }


def enhance_zero_dce(input_path, output_path):
    import torch
    import torchvision.transforms as T

    model = load_zero_dce()
    device = get_device()

    logger.info(f"Zero-DCE processing: {input_path}")
    start = time.time()

    original = Image.open(input_path).convert('RGB')
    original_width, original_height = original.size

    max_dim = 1200
    proc_img = original
    if max(original_width, original_height) > max_dim:
        scale = max_dim / max(original_width, original_height)
        proc_img = original.resize((int(original_width * scale), int(original_height * scale)), Image.LANCZOS)

    transform = T.ToTensor()
    img_tensor = transform(proc_img).unsqueeze(0).to(device)

    infer_start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    infer_time = time.time() - infer_start

    output_img = T.ToPILImage()(output.squeeze(0).cpu())
    if output_img.size != (original_width, original_height):
        output_img = output_img.resize((original_width, original_height), Image.LANCZOS)
    output_img.save(output_path, 'PNG')

    total_time = time.time() - start
    logger.info(f"Zero-DCE done: inference={infer_time:.2f}s, total={total_time:.2f}s")

    return {
        "success": True,
        "model": "zero_dce",
        "input_dimensions": f"{original_width}x{original_height}",
        "output_path": output_path,
        "output_size": os.path.getsize(output_path),
        "inference_time": round(infer_time, 2),
        "total_time": round(total_time, 2),
    }


def enhance_clahe(input_path, output_path):
    logger.info(f"CLAHE processing: {input_path}")
    start = time.time()

    img = cv2.imread(input_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, result)

    total_time = time.time() - start
    h, w = img.shape[:2]
    logger.info(f"CLAHE done: {total_time:.4f}s")

    return {
        "success": True,
        "model": "clahe",
        "input_dimensions": f"{w}x{h}",
        "output_path": output_path,
        "output_size": os.path.getsize(output_path),
        "inference_time": round(total_time, 4),
        "total_time": round(total_time, 4),
    }


def enhance_histogram_eq(input_path, output_path):
    logger.info(f"Histogram EQ processing: {input_path}")
    start = time.time()

    img = cv2.imread(input_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, result)

    total_time = time.time() - start
    h, w = img.shape[:2]
    logger.info(f"Histogram EQ done: {total_time:.4f}s")

    return {
        "success": True,
        "model": "histogram_eq",
        "input_dimensions": f"{w}x{h}",
        "output_path": output_path,
        "output_size": os.path.getsize(output_path),
        "inference_time": round(total_time, 4),
        "total_time": round(total_time, 4),
    }


def enhance_gamma(input_path, output_path):
    logger.info(f"Gamma correction processing: {input_path}")
    start = time.time()

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    result = cv2.LUT(img, table)
    cv2.imwrite(output_path, result)

    total_time = time.time() - start
    h, w = img.shape[:2]
    logger.info(f"Gamma done: gamma={gamma:.1f}, brightness={mean_brightness:.0f}, {total_time:.4f}s")

    return {
        "success": True,
        "model": "gamma_correction",
        "input_dimensions": f"{w}x{h}",
        "output_path": output_path,
        "output_size": os.path.getsize(output_path),
        "inference_time": round(total_time, 4),
        "total_time": round(total_time, 4),
    }


ENHANCE_FUNCTIONS = {
    "mirnet-low-light": enhance_mirnet,
    "zero-dce": enhance_zero_dce,
    "clahe": enhance_clahe,
    "histogram-eq": enhance_histogram_eq,
    "gamma-correction": enhance_gamma,
}


def main():
    logger.info("Multi-model inference server starting (stdin/stdout JSON protocol)")

    try:
        load_mirnet()
        load_zero_dce()
        logger.info("All deep learning models loaded, ready for requests")
    except Exception as e:
        logger.warning(f"Some models failed to preload: {e}")

    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            input_path = request.get("input_path")
            output_path = request.get("output_path")
            model_id = request.get("model", "mirnet-low-light")

            if not input_path or not output_path:
                result = {"success": False, "error": "Missing input_path or output_path"}
            elif not os.path.exists(input_path):
                result = {"success": False, "error": f"Input file not found: {input_path}"}
            elif model_id not in ENHANCE_FUNCTIONS:
                result = {"success": False, "error": f"Unknown model: {model_id}"}
            else:
                result = ENHANCE_FUNCTIONS[model_id](input_path, output_path)

        except json.JSONDecodeError as e:
            result = {"success": False, "error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
            result = {"success": False, "error": str(e)}

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
