---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-size: 24px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .box { padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
  h1 { color: #2c3e50; }
  h2 { color: #34495e; }
  strong { color: #e74c3c; }
---

# Simple and Effective Masked Diffusion Language Models

<div style="text-align: center;">

> [**Paper Link:** https://arxiv.org/abs/2406.07524](https://arxiv.org/pdf/2406.07524)
</div>

<div style="text-align: center;"> 

**Presenters:** Qihang Zhang, Donglin Yang

</div>
<br>
<div style="text-align: center;">2025, Dec 3rd</div>

---

## Overview: The Gap in Discrete Diffusion

*   **Context:** Diffusion models dominate image generation but lag behind Autoregressive (AR) models in text (Log-Likelihood gap).
*   **The Solution:** MDLM (Masked Diffusion Language Model).
*   **Key Insight:** Simplify discrete diffusion to a **weighted average of Masked Language Modeling (MLM) losses**.
*   **Result:** New State-of-the-Art (SOTA) among diffusion models, approaching AR perplexity.

---

## **Continuous vs. Discrete Diffusion Models**

| Feature | Continuous Diffusion (DDPM) | Masked Discrete Diffusion (MDLM) |
| :--- | :--- | :--- |
| **Data Space** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Continuous $x \in \mathbb{R}^d$</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Discrete tokens $x \in \mathcal{V}$ (one-hot)</div> |
| **Noise Source** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Masking state $m$ (`[MASK]`)</div> |
| **Forward Process** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Gradually add Gaussian noise until signal is destroyed.</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Gradually replace tokens with `[MASK]` (absorbing state).</div> |
| **Reverse Process** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Predict the **mean** or **noise** to denoise steps.</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Predict **original tokens** $x_0$ to "unmask".</div> |


---

### 1. The Forward Process: Comparison

| Feature | Continuous (Standard DDPM) | MDLM (Masked Diffusion) |
| :--- | :--- | :--- |
| **Core Mechanism** | Additive Gaussian Noise | Interpolation / Masking |
| **Transition Function** | $$q(\mathbf{z}\_t\|\mathbf{z}\_{t-1}) = \mathcal{N}(\mathbf{z}\_t; \sqrt{1-\beta_t}\mathbf{z}\_{t-1}, \beta_t \mathbf{I})$$| $$q(\mathbf{z}\_t\|\mathbf{z}\_s) = \text{Cat}(\mathbf{z}\_t; \mathbf{Q}\_{t\mid s}^\top z_s) =  \text{Cat}(\mathbf{z}\_t; \alpha_{t\mid s} \mathbf{z}\_{s} + (1-\alpha_{t\mid s})\mathbf{m})$$ $$Q_{t\mid s}=\alpha_{t\mid s}\mathbf{I}+ (1-\alpha_{t\mid s})\mathbf{1}\mathbf{m}^\top, \alpha_{t\mid s}=\frac{\alpha_t}{\alpha_s}$$|
| **Marginal Distribution**$q(\mathbf{z}\_t\|\mathbf{x})$ | $$\mathcal{N}(\mathbf{z}\_t; \sqrt{\bar{\alpha}_t}\mathbf{x}, (1-\bar{\alpha}_t)\mathbf{I})$$ | $$\text{Cat}(\mathbf{z}\_t; \alpha_t \mathbf{x} + (1-\alpha_t)\mathbf{m})$$ |
| **State Space** | **Continuous** ($\mathbf{z}\_t \in \mathbb{R}^D$)<br>Value shifts gradually; info is obscured by noise variance. | **Discrete (Absorbing)** ($\mathbf{z}\_t \in \{\mathbf{x}, \mathbf{m}\}$)<br>Token is either clean or strictly `[MASK]`. Once masked, info is lost (absorbing). |
| **Schedule** | Variance schedule $\beta_t$ (or $\alpha_t$) controls noise level. | $\alpha_t$ is strictly decreasing ($1 \to 0$); represents probability of token being unmasked. |

---

## 2. The Forward Posterior: Derivation

We need $q(\mathbf{z}\_s | \mathbf{z}\_t, \mathbf{x})$ where $s < t$ (less noisy).
From D3PM (Austin et al.), the posterior for discrete diffusion is given by Bayes rule involving the transition matrices $Q$:

$$
q(\mathbf{z}_s|\mathbf{z}_t, \mathbf{x}) = \text{Cat}\left\( \mathbf{z}_s; \frac{Q _{t|s}\mathbf{z} _t \odot Q _s^\top \mathbf{x}}{\mathbf{z} _t^\top Q _t^\top \mathbf{x}} \right) 
$$

Where:
*   $Q_{t|s} = \alpha_{t|s}\mathbf{I} + (1-\alpha_{t|s})\mathbf{1}\boldsymbol{\pi}^\top$ (Transition from $s \to t$)
*   $\odot$ is the element-wise (Hadamard) product.
*   $\boldsymbol{\pi}$ is the stationary distribution (the noise prior).

---
We substitute the definition of $Q$ into the posterior,

$$
q(\mathbf{z}_s|\mathbf{z}_t, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}_s; \frac{[\alpha\_{t|s}\mathbf{I} + (1-\alpha\_{t|s})\mathbf{1}\boldsymbol{\pi}^\top]\mathbf{z} _t \odot [\alpha\_s\mathbf{I} + (1-\alpha\_s)\mathbf{1}\boldsymbol{\pi}^\top]^\top \mathbf{x}}{\mathbf{z}_t^\top[\alpha\_t\mathbf{I} + (1-\alpha\_t)\mathbf{1}\boldsymbol{\pi}^\top]^\top \mathbf{x}}\right\)
$$
$$
= \text{Cat}\left\(\mathbf{z}_s; \frac{[\alpha _{t|s}\mathbf{z} _t + (1-\alpha _{t|s})\mathbf{1}(\boldsymbol{\pi}^\top \mathbf{z} _t)] \odot [\alpha _s\mathbf{x} + (1-\alpha _s)\boldsymbol{\pi}]}{\alpha\_t \mathbf{z}_t^\top\mathbf{x}+(1-\alpha_t)\mathbf{z}_t^\top\boldsymbol{\pi}}\right\)
$$

Now, we apply this to **Masked Diffusion**.
*   The prior $\boldsymbol{\pi} = \mathbf{m}$ (the Mask token).
*   State space is discrete: $\mathbf{z}_t \in \{\mathbf{x}, \mathbf{m}\}$.

We analyze two cases: **Unmasked** and **Masked**.

---

## Case 1: The Unmasked Token ($\mathbf{z}_t = \mathbf{x}$)

If the token at time $t$ is **not** masked ($\mathbf{z}_t = \mathbf{x}$), it must have been unmasked at time $s$ (since masking is absorbing).
Substitute $\mathbf{z}\_t = \mathbf{x}$ and $\boldsymbol{\pi} = \mathbf{m}$, and
note that $\mathbf{x} \odot \mathbf{m} = 0$ (orthogonal vectors) and $\mathbf{x}^\top \mathbf{m} = 0$.

$$
q(\mathbf{z}\_s|\mathbf{z}\_t=\mathbf{x}, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{x} + (1-\alpha_{t|s})\mathbf{1}\overbrace{\mathbf{m}^\top \mathbf{x}}^{0}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{\alpha_t \mathbf{x}^\top \mathbf{x}+(1-\alpha_t)\mathbf{x}^\top\mathrm{m}}\right\)
$$

$$
= \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{x}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{\alpha_t}\right\)=\text{Cat}\left\(\mathbf{z}\_s;\mathbf{x}\right\)=\text{Cat}\left\(\mathbf{z}\_s;\mathbf{z}\_t\right\)
$$
---

## Case 2: The Masked Token ($\mathbf{z}_t = \mathbf{m}$)

If the token is currently masked, it was either **already masked** at time $s$, or it **became masked** between $s$ and $t$. Substitute $\mathbf{z}_t = \mathbf{m}$ and $\boldsymbol{\pi} = \mathbf{m}$, and note that $\mathbf{m}^\top \mathbf{m} = 1$ and $\mathbf{m} \odot \mathbf{m} = \mathbf{m}$.

$$
q(\mathbf{z}\_s|\mathbf{z}\_t=\mathbf{x}, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{m} + (1-\alpha_{t|s})\mathbf{1}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{(1-\alpha_t)}\right\)
$$

$$
= \text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x}}{1-\alpha_t}\right\)
$$

---

## Summary: The Posterior $q(\mathbf{z}\_s | \mathbf{z}\_t, \mathbf{x})$

Combining both cases, we arrive at the closed-form posterior used for training:

$$
q(\mathbf{z}\_s|\mathbf{z}\_t, \mathbf{x}) = 
\text{Cat}(\mathbf{z}\_s; \mathbf{z}\_t) \quad \text{if } \quad \mathbf{z}\_t \neq \mathbf{m}
$$

$$
q(\mathbf{z}\_s|\mathbf{z}\_t, \mathbf{x}) = 
\text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x}}{1-\alpha_t}\right\) \quad \text{if} \quad \mathbf{z}\_t = \mathbf{m} 
$$


**Key Insight:** If the token is masked, the posterior is weighted by the noise schedule difference. If the token is unmasked, it will not be changed.


---

## 3. The Reverse Process: Parameterization

How do we model $p_\theta(\mathbf{z}\_s | \mathbf{z}\_t)$ to approximate the posterior?

**SUBS Parameterization** 
Network $x_\theta(\mathbf{z}\_t, t)$ predicts clean $x$.

$$
p_\theta(\mathbf{z}\_s|\mathbf{z}\_t) = 
\text{Cat}(\mathbf{z}\_s; \mathbf{z}\_t) \quad \text{if } \quad \mathbf{z}\_t \neq \mathbf{m}
$$

$$
p_\theta(\mathbf{z}\_s|\mathbf{z}\_t) = 
\text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x_\theta(\mathbf{z}_t)}}{1-\alpha_t}\right\) \quad \text{if} \quad \mathbf{z}\_t = \mathbf{m} 
$$

**Logic:** If token is visible, keep it. If masked, use network prediction.
---

## Deep Dive: The SUBS Parameterization

The authors introduce **SUBS** (Substitution) to enforce logic:

1.  **Zero Masking Probabilities:**
    $\langle x_\theta(z_t, t), m \rangle = 0$.
    *   The network never predicts `[MASK]` as the original token.
    *   *Implementation:* Set logit for `[MASK]` to $-\infty$.

2.  **Carry-Over Unmasking:**
    If $z_t$ is unmasked, $x_\theta$ is ignored/overwritten by $z_t$.
    *   *Result:* Unmasked tokens remain unchanged during reverse diffusion.

---

## 4. The Objective Function (Rao-Blackwellized Likelihood Bounds)

The discrete-time diffusion loss for finite $T$.

$$
\mathcal{L}\_{\text{diffusion}} = \sum_{i=1}^{T} \mathbb{E}\_{q} \left[ \text{D}\_{\text{KL}} \left( q(\mathbf{z}\_{s(i)} | \mathbf{z}\_{t(i)}, \mathbf{x}) \middle\| p\_{\theta}(\mathbf{z}\_{s(i)} | \mathbf{z}\_{t(i)}) \right) \right]
$$
$$
= \sum\_{i=1}^{T} \mathbb{E}\_{q} \left[ \frac{\alpha_{t(i)} - \alpha_{s(i)}}{1 - \alpha_{t(i)}} \log \langle \mathbf{x}\_{\theta}(\mathbf{z}\_{t(i)}), \mathbf{x} \rangle \right]
$$

*   **Weighted Cross-Entropy.**
*   Only computed on masked tokens.

---

## Deriving the Objective

The discrete diffusion loss is $\mathbb{E}_q [ D\_{KL}(q(z_s|z_t, x) || p_\theta(z_s|z_t)) ]$.

1.  **Case 1: $z_t = x$ (Unmasked)**
    *   $q(z_s|z_t, x)$ is deterministic (stay unmasked).
    *   $p_\theta$ copies $z_t$ (Carry-Over).
    *   $D_{KL} = 0$. (Loss is zero).

2.  **Case 2: $z_t = m$ (Masked)**
    *   Simplifies to a standard Cross Entropy term scaled by diffusion noise schedule:
    $$ \text{Loss} \propto - \log p_\theta(x | z_t \text{ is masked}) $$

---

## 5. Continuous Time Extension ($T \to \infty$)

The authors extend this to continuous time for better performance (Eq 10).

$$ \mathcal{L}\_{\text{NELBO}}^\infty = \mathbb{E}\_{q} \int_{0}^{1} \frac{\alpha'\_t}{1-\alpha_t} \log \langle x_\theta(z_t, t), x \rangle dt $$

*   **Interpretation:**
    *   $\frac{\alpha'_t}{1-\alpha_t}$: Weighting function based on noise schedule.
    *   $\log \langle \dots \rangle$: Standard Cross Entropy .
    
---

## 6. Masked Diffusion Language Models

Next, the authors apply masked diffusion to language modeling over sequences $\mathbf{x}^{1:L}$ of $L$ tokens. The forward noising process is applied independently accross a sequence, and $p_\theta({\mathbf{z}_s^{1:L}\mid \mathbf{z}_t^{1:L}})=\Pi\_{l=1}^{L} p_\theta(\mathbf{z}^l_s\mid  \mathbf{z}_t^{1:L})$.

$$ \mathcal{L}\_{\text{NELBO}}^\infty = \mathbb{E}\_{q} \int_{0}^{1} \frac{\alpha'\_t}{1-\alpha_t} \sum_{\ell=1}^{L} \log \langle x_\theta^\ell(z_t^{1:L}, t), x^\ell \rangle dt $$

* **Note**: Although the loss imposes a loss on all tokens, **unmasked tokens don’t contribute to the loss**, as they are copied over by the denoising network due to “carry-over unmasking”.
    
**Conclusion:** MDLM training is simply **Weighted Masked Language Modeling**. It establishes a connection between generative diffusion models and **encoder-only BERT models**.

---


## 7.  Summary of MDLM Method

1.  **Forward:** Independent tokens flip to `[MASK]` over time.
2.  **Reverse:** Predict $x_0$ directly using a Transformer (DiT/BERT).
3.  **Parameterization (SUBS):**
    *   Never predict mask.
    *   Trust unmasked tokens (don't re-predict them).
4.  **Training:**
    *   Sample time $t$.
    *   Mask input $x$ based on $\alpha_t$.
    *   Minimize Cross Entropy on masked tokens weighted by $\frac{\alpha'_t}{1-\alpha_t}$.

---

## MDLM vs Standard MLM

*   **Standard BERT (MLM):**
    *   Mask 15% of tokens once.
    *   Predict at masked positions.
    
*   **MDLM:**
    *   Masking rate varies from 0% to 100% (controlled by $t$).
    *   Loss is weighted.
    *   **Generation:** Can generate text from pure noise (all masks) by iteratively unmasking.

---

# Thank You