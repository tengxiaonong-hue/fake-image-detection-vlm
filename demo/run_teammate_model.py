"""Simple command-line sanity check for the team's original V1 demo adapter."""
import argparse
import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA"
PROMPT = """Classify this image as REAL or AI-GENERATED.
Return only one label:
REAL
AI-GENERATED"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path")
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")
    print(f"Adapter: {ADAPTER}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER).to(device)
    model.eval()
    with Image.open(args.image) as file:
        image = ImageOps.exif_transpose(file).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=8, do_sample=False
        )
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(
        generated_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    print(f"Raw output: {result}")

if __name__ == "__main__":
    main()
