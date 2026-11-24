# Steering vs. Fine-Tuning for Safety in LLMs

**Team:** Jia Li, Yanchi Li, Shenghe Shao, Suyang Ma

## Abstract

This project compares difference-of-means activation steering, learned activation steering, and LoRA fine-tuning for improving the safety of small open-weight language models under limited computational budgets. We evaluate how effectively each method prevents harmful responses while preserving performance on benign tasks, and we measure the computational cost of each. We hypothesize that learned steering generalizes better to unseen jailbreaks than difference-of-means steering, and achieves safety comparable to LoRA at a lower cost. 

## Introduction and Background

Open-weight language models allow users to modify both model parameters and internal representations. This flexibility supports inexpensive adaptation, but also raises questions about how reliably safety behavior can be controlled. We compare activation steering with LoRA fine-tuning to understand when a single steering vector is sufficient and when weight updates are needed.

**Alignment background.** Christiano et al. \[1\] showed that agents can learn from human preference comparisons instead of hand-written rewards, and InstructGPT \[2\] applied this to LLMs through SFT followed by RLHF. Since full RLHF is expensive, we focus on cheaper supervised safety tuning. Bianchi et al. \[3\] showed that a small amount of safety data already helps a lot, but too much causes over-refusal, so we measure both.

**Efficient fine-tuning.** LoRA \[4\] trains low-rank weight updates on a frozen model and serves as our weight-update baseline. Prompt tuning \[5\] goes further and learns only a few input embeddings; notably, it gets closer to full fine-tuning as models grow, which suggests that the relative strength of lightweight methods may depend on model size.

**Representation steering**. Representation Engineering (RepE) \[6\] proposes reading and controlling high-level concepts directly in a model's activations. CAA \[7\] constructs a steering vector from mean activation differences between contrasting examples and adds it at inference time. Arditi et al. \[8\] identify a direction that mediates refusal, but adding it can also induce refusal on harmless requests. Subramani et al. \[9\] optimize latent steering vectors on frozen models for target-sentence reconstruction and sentiment transfer; this motivates our optimization approach, rather than establishing that a shared safety vector will generalize across requests. Circuit Breakers \[10\] trains model representations to reroute harmful trajectories and improve robustness to jailbreaks.

**Our angle**. Alfarra et al. \[11\] compare lightweight safety methods for on-device deployment, including soft prompts, LoRA, and steering vectors. Our project focuses on how safety and benign-task performance change as the available training budget increases. We compare **difference-of-means steering**, a **learned steering vector**, and **LoRA SFT** under matched data exposure and FLOP budgets. Our intended contribution is a controlled empirical comparison on a small model.

## Research Questions and Methods

We ask three questions. First, how does learned activation steering compare with LoRA safety fine-tuning in safety, capability, and cost? Second, does learned steering outperform difference-of-means steering on held-out jailbreak prompts? Third, do any of these interventions degrade general capability or cause over-refusal? Time permitting, we also ask whether the answers change with model size.

**Model and scope.** We plan to start with Qwen3.5-0.8B, while taking Qwen3.5-2B as an optional extension.We plan to use a subset of harmful and benign prompt-response pairs from WildJailbreak \[12\] as the training data of SFT and the demonstration to RepE.

For learned steering, we freeze model weights and optimize one vector shared across all requests. LoRA uses the same supervised objective, targeting supported attention projection layers after inspecting the architecture. 

**Evaluation.** We evaluate every setting on the same benchmarks. For safety, we report attack success rate on HarmBench \[13\] and on a held-out set of jailbreak prompts. For over-refusal, we use XSTest-style \[14\] benign prompts that look harmful. For general capability, we use MMLU \[15\] and MT-Bench \[16\], with others added if time allows. 

**Compute comparison**. Since methods run on different members' machines, budgets are measured in total training FLOPs rather than GPU time; our primary comparison uses equal FLOP budgets at three artificially determined levels. We will report training time, peak allocated GPU memory, trainable parameter count, and inference latency at fixed input/output length. Direction extraction and hyperparameter search costs will be reported separately, with a fixed search budget per method. 

## Compute Resources

We have three lightweight laptops and one desktop with an NVIDIA RTX 4070 SUPER (12 GB VRAM). Training and timing comparisons will run on the desktop. 

## References

\[1\] P. Christiano et al. Deep Reinforcement Learning from Human Preferences. NeurIPS, 2017\. arXiv:1706.03741.

\[2\] L. Ouyang et al. Training Language Models to Follow Instructions with Human Feedback. NeurIPS, 2022\. arXiv:2203.02155.

\[3\] F. Bianchi et al. Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions. ICLR, 2024\. arXiv:2309.07875.

\[4\] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022\. arXiv:2106.09685.

\[5\] B. Lester, R. Al-Rfou, and N. Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP, 2021\. arXiv:2104.08691.

\[6\] A. Zou et al. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405, 2023\.

\[7\] N. Rimsky et al. Steering Llama 2 via Contrastive Activation Addition. ACL, 2024\. arXiv:2312.06681.

\[8\] A. Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS, 2024\. arXiv:2406.11717.

\[9\] N. Subramani, N. Suresh, and M. E. Peters. Extracting Latent Steering Vectors from Pretrained Language Models. Findings of ACL, 2022\. arXiv:2205.05124.

\[10\] A. Zou et al. Improving Alignment and Robustness with Circuit Breakers. NeurIPS, 2024\. arXiv:2406.04313.

\[11\] M. Alfarra, C. Pinneri, D. Kianfar, M. Almousa, and C. Louizos. Distilling Safe LLM Systems via Soft Prompts for On Device Settings. arXiv:2606.09388, 2026\.

\[12\] L. Jiang et al. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models. NeurIPS, 2024\. arXiv:2406.18510. 

\[13\] M. Mazeika et al. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. ICML, 2024\. arXiv:2402.04249.

\[14\] Röttger, Paul, et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. arXiv, 2023, arXiv:2308.01263.

\[15\] D. Hendrycks et al. Measuring Massive Multitask Language Understanding. ICLR, 2021\. arXiv:2009.03300.

\[16\] L. Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023\. arXiv:2306.05685.