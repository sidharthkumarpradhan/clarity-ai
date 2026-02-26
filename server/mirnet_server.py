import sys
import os
import json
import logging
import time
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [mirnet] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger('mirnet')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(WORKSPACE, 'models', 'mirnet_finetuned.pth')

sys.path.insert(0, WORKSPACE)

_model = None
_device = None

def load_model():
    global _model, _device
    if _model is not None:
        logger.info("Using cached MIRNet model")
        return _model, _device

    import torch
    from server.model.MIRNet.model import MIRNet

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}. Please download mirnet_finetuned.pth from HuggingFace.")

    logger.info(f"Loading MIRNet model from {WEIGHTS_PATH}")
    start = time.time()

    _device = torch.device("cpu")
    _model = MIRNet().to(_device)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    elapsed = time.time() - start
    logger.info(f"MIRNet model loaded in {elapsed:.2f}s")
    return _model, _device

def enhance_image(model, device, input_path, output_path, max_size=400):
    import torch
    import torchvision.transforms as T

    logger.info(f"Processing image: {input_path}")
    start = time.time()

    original = Image.open(input_path).convert('RGB')
    original_width, original_height = original.size
    logger.info(f"Input dimensions: {original_width}x{original_height}")

    transform = T.Compose([
        T.Resize(max_size),
        T.ToTensor(),
    ])

    img_tensor = transform(original).unsqueeze(0).to(device)

    if img_tensor.shape[2] % 8 != 0:
        img_tensor = img_tensor[:, :, :-(img_tensor.shape[2] % 8), :]
    if img_tensor.shape[3] % 8 != 0:
        img_tensor = img_tensor[:, :, :, :-(img_tensor.shape[3] % 8)]

    logger.info(f"Input tensor shape: {list(img_tensor.shape)}")

    infer_start = time.time()
    with torch.no_grad():
        output = model(img_tensor)

    output = output.clamp(0, 1)
    infer_elapsed = time.time() - infer_start
    logger.info(f"Inference completed in {infer_elapsed:.2f}s")

    output_img = T.ToPILImage()(output.squeeze(0).cpu())
    output_img = output_img.resize((original_width, original_height), Image.LANCZOS)
    output_img.save(output_path, 'PNG')

    output_size = os.path.getsize(output_path)
    total_elapsed = time.time() - start
    logger.info(f"Enhanced image saved: {output_path} ({output_size} bytes)")
    logger.info(f"Total processing time: {total_elapsed:.2f}s")

    return {
        "success": True,
        "input_dimensions": f"{original_width}x{original_height}",
        "output_path": output_path,
        "output_size": output_size,
        "inference_time": round(infer_elapsed, 2),
        "total_time": round(total_elapsed, 2)
    }

def main():
    logger.info("MIRNet inference server starting (stdin/stdout JSON protocol)")

    try:
        model, device = load_model()
        logger.info("Model loaded, ready for requests")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

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

            if not input_path or not output_path:
                result = {"success": False, "error": "Missing input_path or output_path"}
            elif not os.path.exists(input_path):
                result = {"success": False, "error": f"Input file not found: {input_path}"}
            else:
                result = enhance_image(model, device, input_path, output_path)

        except json.JSONDecodeError as e:
            result = {"success": False, "error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
            result = {"success": False, "error": str(e)}

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
