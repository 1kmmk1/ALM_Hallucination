# ALM_Hallucination

## Overview

Audio-Language Models (ALMs) frequently exhibit **"attribute binding hallucinations"** — a critical failure where they incorrectly associate sounds with their sources.

For example, a model may misinterpret *"a man crying and a woman laughing"* as *"a man laughing and a woman crying."*

## The Evaluation Gap

Existing discriminative evaluation methods are **insufficient**. Models achieve near-perfect accuracy when the prompt matches the audio (Congruent), but fail completely when attributes are swapped (Incongruent) — even though all sounds still exist in the audio.

| Model | Congruent Acc. (%) | Incongruent Acc. (%) |
|-------|:------------------:|:--------------------:|
| Qwen2-Audio-Instruct | 99.74 | 1.06 |
| GAMA | 97.87 | 0.00 |
| Phi4-MM-Instruct | 100.00 | 3.46 |
| SALMONN-7B | 100.00 | 0.00 |

> *Congruent*: "Is there a man crying and a woman laughing?" → Correct answer: **Yes**  
> *Incongruent*: "Is there a man laughing and a woman crying?" → Correct answer: **No**

This reveals that models only check for **co-existence** of sounds, not their correct binding.

## Dataset

📦 **Synthetic Dataset**: [Download Link](https://drive.google.com/file/d/1ClFBMZyeTalhqqYcawuehr6YumY8rz7G/view?usp=sharing)

- 376 high-quality synthetic audio samples
- Controlled object-attribute pairs (e.g., Man/Woman × Crying/Laughing)
- PAM score filtered (threshold: 0.75) for audio quality
