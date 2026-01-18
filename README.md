# VIde-AI-Assignment

# Empathetic Best-Friend Chatbot (QLoRA + Multi-Objective SFT)

> Fine-tuning an open-weights LLM to behave as an empathetic, safety-aware best-friend chatbot with measurable gains on empathy benchmarks.

**Author:** Aditya Pal  
**Reviewer / Tagged:** @ahan-2000  

---

## 🔍 Objective

This project fine-tunes an open-weights LLM to respond with **empathy, emotional awareness, and safety compliance**, following the **Vibe Hiring Round Task** specification.

Key goals:
- Improve empathetic response quality over the base model
- Handle high-risk emotional content safely (e.g., self-harm ideation)
- Demonstrate measurable gains and clear ablations

---

## 🧠 Model & Compute

- **Base Model:** Qwen2.5-7B-Instruct  
- **Fine-Tuning:** QLoRA (4-bit, PEFT)  
- **Context Window:** Default model limit  
- **Hardware:** Single-GPU (Colab-compatible)

No full-precision SFT is used.

---

## 📚 Data Sources

Multi-task mixture with temperature-based sampling:

| Dataset | Purpose |
|------|------|
| EmpatheticDialogues | Empathy-aligned dialogue SFT |
| ESConv | Emotional support strategy supervision |
| GoEmotions | Emotion classification auxiliary head |

### Dataset Mixing (Temperature Sampling)

Sampling probability for dataset i:

p_i = (n_i ^ alpha) / sum_j (n_j ^ alpha)

where:
- n_i = number of training examples in dataset i
- alpha ∈ (0, 1] controls balance (lower = more uniform)


---

## 🧩 Model Architecture

Shared decoder with multiple heads:

- **LM Head** → Assistant token generation  
- **Emotion Head** → User emotion classification  
- **Strategy Head** → Support strategy prediction  

Auxiliary heads operate on the final hidden state.

---

### Training Objective (Multi-Objective SFT Loss)

Total loss:

L_SFT =
  lambda_LM * L_NLL
+ lambda_emo * L_emo
+ lambda_strat * L_strat
+ lambda_safe * L_safe

where:
- L_NLL   = negative log-likelihood of assistant tokens
- L_emo   = emotion classification cross-entropy
- L_strat = strategy classification cross-entropy
- L_safe  = safety KL-divergence loss

Chosen weights (dev-tuned):
- lambda_LM = 1.0
- lambda_emo = 0.3
- lambda_strat = 0.3
- lambda_safe = 0.1


---

## 🔐 Safety Handling

High-risk signals (e.g., *“I feel like hurting myself”*) trigger an **inference-time safety controller**:

- No probing questions  
- No fabricated memories  
- No invalidating language  
- Single, calm escalation encouraging external support  

Crisis responses:
- Acknowledge pain
- Name emotion
- Encourage trusted help / professional support
- Avoid directives or actionable advice

---

## 🧪 Evaluation

### Primary Metric
- **EQ-Bench 3** (raw + normalized comparison)

### Secondary Checks
- GoEmotions macro-F1 (aux head consistency)
- Safety red-team prompts


---

## 🗣️ Qualitative Examples

The model consistently:
- Names emotions explicitly
- Uses warm, non-judgmental language
- Escalates safely for high-risk inputs
- Maintains conversational tone for positive updates (e.g., job placements)

---

## ♻️ Reproducibility

- Deterministic seeds
- Logged loss curves
- Clear separation of base / SFT / SFT+DPO checkpoints
- No proprietary or private data used

---

## 📌 Notes for Reviewers

- All safety behavior is enforced **without hard-coding responses**
- Empathy improvements persist across domains
- Designed to be lightweight and reproducible

Reviewer: **@ahan-2000**.

---

## ✅ Status

✔ Modeling complete  
✔ Safety validated  
✔ Evaluation ready  
✔ Hiring-round submission ready
