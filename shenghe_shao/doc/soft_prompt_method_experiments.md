# Soft Prompt Tuning: Method and Experiments

English version first, Chinese version below.

---

## English

### Method

Soft prompt tuning adapts a frozen language model by learning a small set of continuous input
embeddings instead of updating any model weights. We prepend $n$ trainable vectors
$P \in \mathbb{R}^{n \times d}$ to the embedded input of every request, where $d$ is the model's
hidden size. All base-model parameters stay fixed, so the only learned component is $P$. With
$n = 20$ and $d = 1024$, this amounts to about 20K trainable parameters.

Each request is formatted with the model's standard chat template, and the soft prompt is placed
in front of the entire formatted sequence. We train $P$ with the same supervised objective used for
LoRA: token-level cross-entropy on the target response, with prompt positions masked out. Gradients
propagate through the frozen network to the input layer and update only $P$. Following common
practice, $P$ is initialized from the token embeddings of a short natural-language instruction,
"You are a helpful assistant." At inference time the same learned prompt is prepended to every
input, so the method adds no per-request cost beyond $n$ extra input positions.

### Experiments

**Setup.** We use Qwen3.5-0.8B with greedy decoding and thinking mode disabled. The training set
contains 4,000 prompt–response pairs from WildJailbreak, balanced across vanilla and adversarial,
harmful and benign requests. Harmful prompts are paired with refusals and benign prompts with
helpful answers. We train for a single epoch with AdamW and save checkpoints after 1,000, 2,000 and
4,000 examples, which serve as three data budgets. Training compute is estimated as
$C \approx 4NT$, where $N$ is the number of parameters involved in each forward pass and $T$ is the
number of processed tokens; the factor of 4 reflects that only activation gradients, not weight
gradients, are computed for the frozen model. The three checkpoints correspond to roughly 1.2, 2.2
and 4.5 PFLOPs.

We measure safety by the attack success rate on HarmBench and on the held-out WildJailbreak
evaluation set, whose jailbreaks are unseen during training. Over-refusal is measured on benign
prompts that resemble harmful ones, using XSTest and the benign portion of the WildJailbreak
evaluation set. General capability is measured with MMLU and MT-Bench. Responses are labeled for
refusal and harmfulness by an LLM judge, DeepSeek-V4.1-Flash, which also grades MT-Bench answers.

**Hyperparameter selection.** We compared four configurations on a held-out development split of
WildJailbreak, varying the initialization text, the learning rate and the number of soft tokens.
All four showed the same pattern: a sharp drop in attack success rate together with a rise in
over-refusal, and the differences between them were within noise. We selected the neutral
initialization because it performed best at the smallest budget and does not bias the prompt toward
refusing. Test sets were evaluated only once, after this choice was fixed.

**Results.** Figures 1–3 show how safety, over-refusal and general capability change with training
compute. The leftmost point in each plot is the base model.

![Safety vs. training compute](../figures/test_safety_vs_flops.png)

*Figure 1: Attack success rate versus training compute. Lower is better.*

![Over-refusal vs. training compute](../figures/test_overrefusal_vs_flops.png)

*Figure 2: Over-refusal rate on benign prompts versus training compute. Lower is better.*

![Capability vs. training compute](../figures/test_capability_vs_flops.png)

*Figure 3: MMLU accuracy and MT-Bench score versus training compute. Higher is better.*

As Figure 1 shows, soft prompt tuning is highly effective at blocking harmful requests. With only 20K trainable
parameters, the attack success rate on HarmBench falls from 32% to 7% after the first 1,000
examples and reaches zero at the full budget. The improvement carries over to unseen jailbreaks,
where the attack success rate drops from 42% to 3%. Since HarmBench is drawn from a different
source than the training data, the learned behavior is not confined to the training distribution.

Figure 2 shows that this safety comes at a substantial cost in over-refusal. On benign requests written in a
jailbreak-like style, the refusal rate climbs from 18% to 74% after 1,000 examples and to 91% at the
full budget. On plainly worded benign prompts from XSTest the increase is much smaller, from 5% to
18%. Together with a manual inspection of refused examples, this suggests that the soft prompt
largely learns to refuse based on the surface form of a request, such as role-play framing, rather
than on whether its content is actually harmful. A single prompt prepended to the input appears to
have too little capacity to make this finer distinction.

Most of the change occurs within the first 1,000 examples. Beyond that point the safety gains
saturate while over-refusal continues to grow, so additional compute yields diminishing returns.
As Figure 3 shows, general knowledge is largely preserved: MMLU accuracy stays within a few points of the base model
and moves in both directions, which points to changes in answer formatting rather than in knowledge.
Conversational quality declines moderately, with the MT-Bench score falling from 3.58 to 2.86.

**Limitations.** Results come from a single training seed, and the LLM judge has not been validated
against human labels. The compute figures are analytic estimates, and MT-Bench scores from our judge
are not directly comparable to published numbers.

