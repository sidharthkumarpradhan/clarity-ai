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
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger('mirnet')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(WORKSPACE, 'models', 'mirnet_finetuned.pth')

sys.path.insert(0, WORKSPACE)

def load_model():
    import torch
    from server.model.MIRNet.model import MIRNet

    logger.info(f"Loading MIRNet model from {WEIGHTS_PATH}")
    start = time.time()

    device = torch.device("cpu")
    model = MIRNet().to(device)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    elapsed = time.time() - start
    logger.info(f"MIRNet model loaded in {elapsed:.2f}s")
    return model, device

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        result = {"success": False, "error": "Usage: mirnet_inference.py <input_path> <output_path>"}
        print(json.dumps(result))
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        result = {"success": False, "error": f"Input file not found: {input_path}"}
        print(json.dumps(result))
        sys.exit(1)

    try:
        model, device = load_model()
        result = enhance_image(model, device, input_path, output_path)
        print(json.dumps(result))
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
        result = {"success": False, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)
