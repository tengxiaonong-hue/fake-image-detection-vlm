# Streamlit Fake Image Detection Demo (team UI)

This folder contains the **team-provided Streamlit front end**, recovered from the `fake_image_demo` archive. It offers image upload in a chat interface, binary Real/AI-generated prediction, optional natural-language explanations, session history, and a clear-chat control.

> **Important distinction:** The code in this folder currently uses the **original V1 LoRA adapter**, `NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA`. It is **not** the V2 checkpoint used for the 93.99% FakeBench Test-ID, 91.00% FakeBench Test-OOD, or 81.60% OpenFake results. Those are independent HPC evaluations with different preprocessing. Do **not** claim these accuracy values are from this Streamlit app.

## Files

| File | Function |
|---|---|
| `chat_image_aligned.py` | Streamlit chat interface, inference, optional explanation display |
| `run_teammate_model.py` | Single-image command-line inference for sanity checking |
| `requirements.txt` | Small UI-specific dependency list (version ranges, **not** a lockfile) |
| `.gitignore` | Prevent accidental local environment and model-weight commits |

## Reproduce locally

Prerequisites: **Python 3.11 recommended**, working PyTorch installation compatible with your GPU, enough memory to load Qwen2.5-VL-3B, network access to Hugging Face to download model files, and any needed HF authentication/accepted model terms. Tested working environment versions were **not recorded in the supplied archive**; the following is a reproducible *procedure*, not a guarantee of byte-for-byte deterministic output.

```bash
git clone https://github.com/tengxiaonong-hue/fake-image-detection-vlm.git
cd fake-image-detection-vlm
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# Install PyTorch for your machine first: https://pytorch.org/get-started/locally/
python -m pip install -r demo/requirements.txt
python -m streamlit run demo/chat_image_aligned.py
```

Visit the local Streamlit address printed in the terminal (normally `http://localhost:8501`). Upload one JPEG, PNG, or WebP image and choose one of the modes. To run without the UI:

```bash
python demo/run_teammate_model.py --image /absolute/path/to/example.jpg
```

**Record versions for actual reproducibility:**

```bash
python --version
python -m pip freeze > demo_environment_freeze.txt
python - <<'PY'
import torch, transformers, peft, streamlit
print("torch",torch.__version__,"transformers",transformers.__version__,
      "peft",peft.__version__,"streamlit",streamlit.__version__)
print("CUDA available",torch.cuda.is_available())
PY
```

Retain the **exact base-model and LoRA adapter commit revisions** from Hugging Face when reporting a fully pinned replication. The present code uses repository names rather than immutable commit hashes.

## Inference details (as implemented)

1. `AutoProcessor` prepares uploaded RGB/EXIF-corrected image and fixed prompt `Classify this image as REAL or AI-GENERATED`.
2. `Qwen/Qwen2.5-VL-3B-Instruct` loads alongside the **V1** PEFT adapter `NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA`.
3. Generation is greedy (`do_sample=False`, up to eight new tokens); raw outputs should be read as model predictions, **not ground truth**.
4. If explanations are requested, `with model.disable_adapter():` generates text using the **untuned base model**, separately from the classification adapter. Two modes correspond to *Predict then explain* (EtC-style) and *Analyse then predict* (CtE-style).
5. The web UI holds chat history in Streamlit session state. It does not itself provide a production server API, authentication, evaluation harness, or permanent prediction log.

**Explanation caveat:** These descriptions are optional, unvalidated generated text; they do not reveal the LoRA classifier's actual internal evidence and may contain hallucinations.

**Image-preprocessing caveat:** Demo preprocessing applies EXIF orientation and RGB conversion only. It **does not** apply V2's 224×224 resize/crop/JPEG normalization. Merely changing `ADAPTER` to the V2 HF repo is **not enough** to reproduce the V2 benchmark: one must align processor parameters, resize/crop/JPEG pipeline, and inference protocol first.

## Validation / limitations

- The supplied ZIP included a local `.venv`; that machine-specific environment is intentionally excluded from Git history.
- The Python source has been preserved as a team demo, with the CLI's displayed messages translated to English. No live Streamlit GPU inference was run by this upload process.
- To assert that the app works on another host, capture `pip freeze`, device details, actual HF model revisions, a known sample prediction, and optionally a screen recording.
- For research-score reproduction use the **separate** HPC evaluation scripts in `../src/` and `../scripts/`, not this interactive interface.
