# 🎓 Fine Tuning Example: Qwen3-4B on GSM8K

A minimal, end to end example of fine tuning **Qwen3-4B** on the **GSM8K** math word problem dataset using **Unsloth** and **LoRA**. The notebook measures accuracy before and after training so you can see exactly what fine tuning buys you. 🚀

## 📌 What This Project Does

1. 📥 Loads `unsloth/Qwen3-4B` in 4 bit quantization
2. 📊 Evaluates the stock model on 200 GSM8K test problems (baseline)
3. 🏋️ Fine tunes with LoRA adapters on 2,000 training examples via TRL's `SFTTrainer`
4. 🔁 Re evaluates the same 200 test problems
5. 💾 Saves the trained LoRA adapters and runs a sample question

## 🗂️ Project Structure

```
Fine_Turning_Example/
├── qwen_gsm8k_finetune.ipynb   # Main notebook: train + eval + demo
└── README.md                   # You are here
```

## ⚙️ Configuration

All settings live in a single `CONFIG` dict at the top of the notebook:

| Setting | Value | Notes |
|---|---|---|
| 🧠 Model | `unsloth/Qwen3-4B` | 4 bit loaded |
| 📏 Max sequence length | 2048 | |
| 🎯 LoRA rank / alpha | 16 / 16 | dropout 0.0 |
| 📚 Train samples | 2,000 | from GSM8K train split |
| 🧪 Eval samples | 200 | from GSM8K test split |
| 📦 Batch size | 2 | grad accum 4 (effective 8) |
| 📈 Learning rate | 2e-4 | linear schedule, 10 warmup steps |
| 🔁 Epochs | 1 | |
| 🎲 Seed | 3407 | |

## 🚀 Getting Started

### Requirements

* A GPU (built and tested on a free **Google Colab T4** ✅)
* Python 3

### Run It

1. Open `qwen_gsm8k_finetune.ipynb` in [Google Colab](https://colab.research.google.com/) with a GPU runtime (Runtime > Change runtime type > T4 GPU)
2. Run the cells top to bottom. Dependencies install automatically:

```bash
pip install -q unsloth
pip install -q --no-deps "trl>=0.11" "datasets>=2.19"
```

3. Watch the baseline vs fine tuned accuracy report at the end 📊

## 🧪 How Evaluation Works

* The model is prompted as a math tutor and must end with `\boxed{answer}`
* Answers are extracted by regex (boxed first, then `####`, then last number fallback)
* Numeric comparison with a tolerance of `1e-4`
* Deterministic decoding (`do_sample=False`) keeps results reproducible 🔒

## 📦 Output

Trained adapters and tokenizer are saved to:

```
outputs/qwen3-4b-gsm8k-lora/
```

## 🛠️ Built With

* [Unsloth](https://github.com/unslothai/unsloth) ⚡ fast, memory efficient fine tuning
* [TRL](https://github.com/huggingface/trl) 🤗 `SFTTrainer`
* [GSM8K](https://huggingface.co/datasets/openai/gsm8k) 🔢 grade school math benchmark
* [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B) 🧠 base model

## 💡 Ideas to Try Next

* 🔢 Bump `train_samples` or `epochs` for higher accuracy
* 🎯 Sweep `lora_r` and `learning_rate`
* 🧩 Try a bigger model like `unsloth/Qwen3-8B`
* 📤 Merge and push the adapters to the Hugging Face Hub
