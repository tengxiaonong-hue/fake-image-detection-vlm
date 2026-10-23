"""Chat-style Streamlit UI for the team's FakeBench LoRA model."""
from io import BytesIO
from html import escape
from base64 import b64encode
import streamlit as st
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
EFFECT_PROMPT = """The authenticity result is: {prediction}.
Now explain the specific visual evidence in the image that supports this result.
Mention only evidence that is actually visible. Keep the explanation concise."""
CAUSE_PROMPT = """Inspect this image for visual evidence relevant to whether it is authentic or AI-generated.
Consider object structure, geometry, text, textures, lighting, shadows, reflections, repeated patterns, semantic consistency, and local artefacts.
Mention only evidence that is actually visible.

Return your answer in exactly this format:
EVIDENCE: concise visual analysis
LABEL: REAL or AI-GENERATED"""

@st.cache_resource(show_spinner="Loading the model. This may take a while on the first run…")
def load_model():
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base, ADAPTER).to(device)
    model.eval()
    return model, processor, device

def generate(model, processor, device, image, prompt, token_limit):
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
    return processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

def predict(image, explanation_mode):
    model, processor, device = load_model()
    image = ImageOps.exif_transpose(image).convert("RGB")
    label = generate(model, processor, device, image, PROMPT, 8)
    explanations = {}
    if label in {"REAL", "AI-GENERATED"} and explanation_mode != "Classification only":
        # Explanations come from the UNADAPTED base model and are not
        # faithful explanations of the LoRA classifier's internal reasoning.
        with model.disable_adapter():
            if explanation_mode in {"Predict then explain", "Show both explanations"}:
                explanations["Predict then explain"] = generate(
                    model, processor, device, image,
                    EFFECT_PROMPT.format(prediction=label), 128
                )
            if explanation_mode in {"Analyse then predict", "Show both explanations"}:
                explanations["Analyse then predict"] = generate(
                    model, processor, device, image, CAUSE_PROMPT, 160
                )
    return label, explanations

st.set_page_config(page_title="Fake Image Detection", page_icon="🖼️", layout="centered")
st.markdown(
    """<style>
    .user-turn { display: flex; justify-content: flex-end; align-items: flex-start;
                 gap: 0.75rem; margin: 1.25rem 0 2rem; }
    .user-content { display: flex; flex-direction: column; align-items: flex-end;
                    max-width: calc(100% - 3rem); }
    .user-photo { display: block; max-width: 100%; width: 320px; height: auto;
                  border-radius: 0.75rem; }
    .user-filename { color: #999; font-size: 0.8rem; margin: 0.25rem 0 0.5rem; }
    .user-question { background: #1c477f; color: white; border-radius: 1.4rem;
                     padding: 0.65rem 1rem; width: fit-content; overflow-wrap: anywhere; }
    .user-avatar { width: 2.5rem; height: 2.5rem; flex: none; display: grid;
                   place-items: center; border-radius: 0.65rem;
                   background: #1c477f; font-size: 1.35rem; }
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] {
        background: transparent; padding-left: 0;
    }
    </style>""", unsafe_allow_html=True,
)
st.title("Explainable Fake Image Detection")
st.caption("Upload an image to see the model's prediction and an optional visual explanation.")
st.caption("Your message appears in the chat. The model currently performs image authenticity detection only.")

if "history" not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.header("Analysis settings")
    mode = st.selectbox("Analysis mode",
        ["Predict then explain", "Classification only", "Analyse then predict", "Show both explanations"])
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

def show_user(image_bytes, filename, message):
    with Image.open(BytesIO(image_bytes)) as image:
        mime = Image.MIME.get(image.format, "image/png")
    source = f"data:{mime};base64,{b64encode(image_bytes).decode('ascii')}"
    question = escape(message or "Is this image real or AI-generated?")
    st.markdown(
        f'<div class="user-turn"><div class="user-content">'
        f'<img class="user-photo" src="{source}" alt="Uploaded image">'
        f'<div class="user-filename">{escape(filename)}</div>'
        f'<div class="user-question">{question}</div></div>'
        f'<div class="user-avatar" aria-label="User">👤</div></div>',
        unsafe_allow_html=True,
    )

def show_pair(item):
    show_user(item["image"], item["name"], item["text"])
    with st.chat_message("assistant"):
        if item["label"] == "REAL":
            st.markdown("### Prediction: Real (REAL)")
        elif item["label"] == "AI-GENERATED":
            st.markdown("### Prediction: AI-generated (AI-GENERATED)")
        else:
            st.markdown(f"### Raw model output: {item['label']}")
        for title, explanation in item["explanations"].items():
            st.markdown(f"**{title}**")
            st.write(explanation)
        if item["explanations"]:
            st.caption("Explanations are generated separately by the base model. They can be wrong and do not reveal the classifier's reasoning.")

for item in st.session_state.history:
    show_pair(item)

submission = st.chat_input(
    "Attach an image and send it for analysis",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)
if submission:
    if not submission.files:
        st.info("Please attach an image using the plus button before sending.")
    else:
        uploaded = submission.files[0]
        image_bytes = uploaded.getvalue()
        try:
            show_user(image_bytes, uploaded.name, submission.text)
        except Exception as exc:
            st.error(f"Could not open image: {exc}")
            st.stop()
        with st.chat_message("assistant"):
            try:
                with Image.open(BytesIO(image_bytes)) as file:
                    image = file.copy()
                with st.spinner("Analysing the image…"):
                    label, explanations = predict(image, mode)
                item = {"name": uploaded.name, "image": image_bytes, "text": submission.text,
                        "label": label, "explanations": explanations}
                st.session_state.history.append(item)
                if label == "REAL":
                    st.markdown("### Prediction: Real (REAL)")
                elif label == "AI-GENERATED":
                    st.markdown("### Prediction: AI-generated (AI-GENERATED)")
                else:
                    st.markdown(f"### Raw model output: {label}")
                for title, explanation in explanations.items():
                    st.markdown(f"**{title}**")
                    st.write(explanation)
                if explanations:
                    st.caption("Explanations are generated separately by the base model. They can be wrong and do not reveal the classifier's reasoning.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
st.caption("Model predictions can be incorrect.")
