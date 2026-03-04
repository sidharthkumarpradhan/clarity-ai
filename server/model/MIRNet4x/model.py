"""MIRNet 4x Super-Resolution model using Keras/TensorFlow.
Downloads model weights from HuggingFace Space on first run.
"""
import os
import zipfile
import shutil
import numpy as np
from PIL import Image

# HuggingFace Space model URL
MODEL_URL = "https://huggingface.co/spaces/sidharthpradhan/4x-mirnet-enhancer/resolve/main/model/4x%20mimarnet.keras"
MODEL_FILENAME = "4x_mimarnet.keras"

INPUT_PATCH = 128
OUTPUT_PATCH = 512
SCALE = OUTPUT_PATCH // INPUT_PATCH  # 4
MAX_SIDE = 256  # cap longest dimension before tiling


def download_model(save_path: str) -> bool:
    """Download the 4x MIRNet model weights from HuggingFace."""
    import urllib.request
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(save_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"[MIRNet4x] Download failed: {e}")
        return False


class MIRNet4x:
    """4x MIRNet Keras model wrapper for YCbCr tiled inference."""

    def __init__(self, weights_path: str):
        import tensorflow as tf

        @tf.keras.utils.register_keras_serializable(package="Custom")
        class Conv2DTransposeCompat(tf.keras.layers.Conv2DTranspose):
            def __init__(self, *args, groups=1, **kwargs):
                super().__init__(*args, **kwargs)
            @classmethod
            def from_config(cls, config):
                config.pop("groups", None)
                return cls(**config)

        @tf.keras.utils.register_keras_serializable(package="Custom")
        class ManhattanSelfAttention(tf.keras.layers.Layer):
            def __init__(self, d_model=128, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
            def build(self, input_shape):
                c = input_shape[-1]
                self.Wq = self.add_weight(name="Wq", shape=(c, self.d_model), initializer="glorot_uniform")
                self.bq = self.add_weight(name="bq", shape=(self.d_model,), initializer="zeros")
                self.Wk = self.add_weight(name="Wk", shape=(c, self.d_model), initializer="glorot_uniform")
                self.bk = self.add_weight(name="bk", shape=(self.d_model,), initializer="zeros")
                self.Wv = self.add_weight(name="Wv", shape=(c, c), initializer="glorot_uniform")
                self.bv = self.add_weight(name="bv", shape=(c,), initializer="zeros")
                super().build(input_shape)
            def call(self, x):
                b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
                xf = tf.reshape(x, [b, h * w, c])
                Q = tf.matmul(xf, self.Wq) + self.bq
                K = tf.matmul(xf, self.Wk) + self.bk
                V = tf.matmul(xf, self.Wv) + self.bv
                attn = tf.nn.softmax(
                    tf.matmul(Q, K, transpose_b=True) / tf.cast(self.d_model, tf.float32) ** 0.5, axis=-1
                )
                return tf.reshape(tf.matmul(attn, V), [b, h, w, c]) + x
            def get_config(self):
                return {**super().get_config(), "d_model": self.d_model}

        custom_objects = {
            "ManhattanSelfAttention": ManhattanSelfAttention,
            "Conv2DTranspose": Conv2DTransposeCompat,
        }

        # Determine if file is a zip (.keras) or legacy h5
        is_zip = zipfile.is_zipfile(weights_path)
        if is_zip:
            dst = "/tmp/mirnet4x_model.keras"
        else:
            dst = "/tmp/mirnet4x_model.h5"

        if not os.path.exists(dst):
            shutil.copy2(weights_path, dst)

        try:
            self.model = tf.keras.models.load_model(
                dst, compile=False, safe_mode=False, custom_objects=custom_objects
            )
        except Exception:
            # Fallback: try as h5
            dst_h5 = "/tmp/mirnet4x_model_fb.h5"
            if not os.path.exists(dst_h5):
                shutil.copy2(weights_path, dst_h5)
            self.model = tf.keras.models.load_model(
                dst_h5, compile=False, safe_mode=False, custom_objects=custom_objects
            )

    def enhance(self, image_np: np.ndarray) -> np.ndarray:
        """Run 4x super-resolution on a numpy RGB image array.
        Returns enhanced numpy RGB array.
        """
        import tensorflow as tf

        pil_rgb = Image.fromarray(image_np)
        orig_w, orig_h = pil_rgb.size

        # Downscale if needed
        if max(orig_w, orig_h) > MAX_SIDE:
            scale_factor = MAX_SIDE / max(orig_w, orig_h)
            proc_w = max(1, int(orig_w * scale_factor))
            proc_h = max(1, int(orig_h * scale_factor))
            pil_rgb = pil_rgb.resize((proc_w, proc_h), Image.LANCZOS)
        proc_w, proc_h = pil_rgb.size

        # YCbCr split
        ycbcr = pil_rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y_arr = np.array(y, dtype=np.float32) / 255.0
        H, W = y_arr.shape

        # Pad to tile multiple
        pad_h = (-H) % INPUT_PATCH
        pad_w = (-W) % INPUT_PATCH
        y_pad = np.pad(y_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        PH, PW = y_pad.shape

        # Tiled inference
        canvas = np.zeros((PH * SCALE, PW * SCALE), dtype=np.float32)
        for r in range(0, PH, INPUT_PATCH):
            for c in range(0, PW, INPUT_PATCH):
                patch = y_pad[r:r + INPUT_PATCH, c:c + INPUT_PATCH][np.newaxis, :, :, np.newaxis]
                out = self.model(tf.constant(patch, dtype=tf.float32), training=False).numpy().squeeze()
                canvas[r * SCALE:(r + INPUT_PATCH) * SCALE, c * SCALE:(c + INPUT_PATCH) * SCALE] = out

        # Crop to actual output
        y_out = Image.fromarray((np.clip(canvas[:H * SCALE, :W * SCALE], 0, 1) * 255).astype(np.uint8), mode="L")
        out_img = Image.merge("YCbCr", [
            y_out,
            cb.resize((W * SCALE, H * SCALE), Image.LANCZOS),
            cr.resize((W * SCALE, H * SCALE), Image.LANCZOS),
        ]).convert("RGB")
        return np.array(out_img)
