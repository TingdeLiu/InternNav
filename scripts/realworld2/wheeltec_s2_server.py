#!/usr/bin/env python3
"""
Wheeltec S2 Server — Qwen3-VL 视觉语言导航解析服务

在 GPU 服务器上运行，接收 Jetson 端发来的 RGB 图像 + 导航指令，
返回目标像素坐标 (u, v) 和导航控制符号序列。

API:
  GET  /health          → {"status": "ok", "model": "..."}
  POST /s2_step         → {"target", "point_2d_norm", "point_2d_pixel", "navigation", "raw"}

启动:
  python scripts/realworld2/wheeltec_s2_server.py \\
      --model_path Qwen/Qwen3-VL-7B-Instruct \\
      --port 8890 \\
      --device auto

依赖 (在 internnav conda 环境中安装):
  pip install flask transformers>=4.57.0 qwen-vl-utils
  pip install flash-attn --no-build-isolation  # 可选，无则自动降级
"""

import argparse
import io
import json
import re
import sys
import traceback

from flask import Flask, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
# Role
You are a high-precision robot vision navigation and object detection system.
Your task is to analyze images and convert user instructions into target coordinates
and navigation control symbols.

# Output Rules (Detection)
1. Localization: When the user specifies a navigation target, identify its exact
   position in the image.
2. Normalized Coordinates: Use integer values ranging from [0, 1000].
3. JSON Format: Strictly follow this format: {"target": "target_name", "point_2d": [x, y]}.
   - 'x' is horizontal, 'y' is vertical.
4. Negative Constraint: If no target is mentioned by the user, OR the target is not
   visible in the image, you MUST return: {"target": null, "point_2d": null}.

# Output Rules (Navigation)
Strictly map movements to symbols:
1. Left:     One "←" per 15° (e.g., 30° = "←←").
2. Right:    One "→" per 15° (e.g., 60° = "→→→→").
3. Forward:  One "↑" per 0.5 meters (e.g., 1m = "↑↑").
4. Backward: One "↓" per 0.5 meters (e.g., 2m = "↓↓↓↓").
5. Stop:     Output "stop" only if a stop command is included.
6. Execute:  Output "start" when an action begins.

# Constraints
- Output Sequence: Always output the JSON block first, followed by navigation
  symbols on a new line.
- Strictly No Filler: Output ONLY the defined symbols or JSON strings.
  No explanations, no markdown blocks (```json), and no conversational text.
- Vision-First: Coordinates must be based on actual visual evidence.
  Do not guess or use placeholder values like [500, 500] if the target is missing.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = None
processor = None
cfg = None  # argparse namespace, set in main()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str) -> None:
    global model, processor

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[S2] Loading processor from {model_path} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device)

    # Try flash_attention_2, fall back to sdpa (PyTorch scaled-dot-product)
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, attn_implementation=attn_impl, **load_kwargs
            )
            print(f"[S2] Loaded with attn_implementation={attn_impl}", flush=True)
            break
        except Exception as exc:
            print(f"[S2] {attn_impl} unavailable: {exc}", flush=True)
    else:
        raise RuntimeError("Could not load model with any attention implementation.")

    model.eval()
    print("[S2] Model ready.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(image_bytes: bytes, instruction: str) -> str:
    """Call Qwen3-VL and return the raw text output."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "resized_width": cfg.resize_w,
                    "resized_height": cfg.resize_h,
                },
                {"type": "text", "text": instruction},
            ],
        },
    ]

    # Standard Qwen3-VL inference path via process_vision_info
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    target_device = next(model.parameters()).device
    inputs = inputs.to(target_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Strip prompt tokens, decode only newly generated tokens
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────
_NAV_PATTERN = re.compile(r"[←→↑↓]+|stop|start")
_JSON_PATTERN = re.compile(r"\{[^{}]+\}", re.DOTALL)


def parse_output(raw: str) -> dict:
    """
    Parse the two-line model output:
      Line 1: {"target": "chair", "point_2d": [320, 240]}
      Line 2: ↑↑←

    Returns:
      target          – string or None
      point_2d_norm   – [x, y] in [0, 1000] or None
      point_2d_pixel  – [u, v] in actual image pixels or None
      navigation      – concatenated nav symbols, e.g. "↑↑←←" or "stop"
      raw             – original model output (for debugging)
    """
    target = None
    point_2d_norm = None
    point_2d_pixel = None

    # ── JSON block ────────────────────────────────────────────────────────────
    json_match = _JSON_PATTERN.search(raw)
    if json_match:
        try:
            data = json.loads(json_match.group())
            target = data.get("target")            # None or string
            norm = data.get("point_2d")            # None or [x, y]

            if norm is not None and isinstance(norm, (list, tuple)) and len(norm) == 2:
                nx, ny = float(norm[0]), float(norm[1])
                point_2d_norm = [int(nx), int(ny)]

                # Convert [0, 1000] → pixel coords; clamp to valid range
                u = int(nx / 1000.0 * cfg.image_width)
                v = int(ny / 1000.0 * cfg.image_height)
                u = max(0, min(cfg.image_width - 1, u))
                v = max(0, min(cfg.image_height - 1, v))
                point_2d_pixel = [u, v]

        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # leave as None, raw output still returned for debugging

    # ── Navigation symbols ────────────────────────────────────────────────────
    nav_parts = _NAV_PATTERN.findall(raw)
    navigation = "".join(nav_parts)   # e.g. "↑↑←←" or "stop" or ""

    return {
        "raw": raw,
        "target": target,
        "point_2d_norm": point_2d_norm,    # [0, 1000] scale
        "point_2d_pixel": point_2d_pixel,  # actual pixel (u, v)
        "navigation": navigation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": cfg.model_path if cfg else "not loaded"})


@app.route("/s2_step", methods=["POST"])
def s2_step():
    """
    POST /s2_step
    Form fields:
      image       – image file (JPEG / PNG)
      instruction – natural language navigation instruction

    Response JSON:
      target          – detected target name or null
      point_2d_norm   – [x, y] in [0, 1000] or null
      point_2d_pixel  – [u, v] in pixel coords (cfg.image_width × cfg.image_height) or null
      navigation      – nav symbol string, e.g. "↑↑←←" / "stop" / ""
      raw             – raw model text output (for debugging)
    """
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "missing form field: image"}), 400

    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "missing form field: instruction"}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "image file is empty"}), 400

    try:
        raw_output = run_inference(image_bytes, instruction)
        result = parse_output(raw_output)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def _round32(n: int) -> int:
    """Round up to the nearest multiple of 32 (required by Qwen3-VL)."""
    return ((n + 31) // 32) * 32


def main():
    global cfg

    parser = argparse.ArgumentParser(
        description="Wheeltec S2 Server — Qwen3-VL navigation instruction parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", default="Qwen/Qwen3-VL-7B-Instruct",
        help="HuggingFace model ID or local checkpoint directory",
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--device", default="auto",
        help="device_map value: 'auto', 'cuda:0', 'cuda:1', 'cpu', …",
    )
    # Camera / image dimensions (used for pixel coordinate conversion)
    parser.add_argument("--image_width",  type=int, default=640,
                        help="Robot camera width in pixels (Astra S default: 640)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Robot camera height in pixels (Astra S default: 480)")
    # Resolution fed to the model (multiples of 32; can be smaller for faster inference)
    parser.add_argument("--resize_w", type=int, default=640,
                        help="Image width passed to Qwen3-VL (rounded up to multiple of 32)")
    parser.add_argument("--resize_h", type=int, default=480,
                        help="Image height passed to Qwen3-VL (rounded up to multiple of 32)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate per request")

    cfg = parser.parse_args()

    # Enforce multiples of 32
    cfg.resize_w = _round32(cfg.resize_w)
    cfg.resize_h = _round32(cfg.resize_h)

    print(f"[S2] Config: {vars(cfg)}", flush=True)

    load_model(cfg.model_path, cfg.device)

    print(f"[S2] Listening on http://{cfg.host}:{cfg.port}", flush=True)
    # threaded=False: single-GPU model is not thread-safe; requests queue naturally
    app.run(host=cfg.host, port=cfg.port, threaded=False)


if __name__ == "__main__":
    main()
