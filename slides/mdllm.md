---
title: "Masked Diffusion Language Models"
---

# Masked Diffusion Language Models

<div style="text-align: center;">

> [**Paper Link:** Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)

</div>

<div style="text-align: center;"> 

**Presenter:** Qihang Zhang

</div>
<br>
<div style="text-align: center;">Dec 3rd, 2025</div>

---

## Today's Agenda

1. Background & Motivation
2. Problem Statement
3. Masked Diffusion Framework
4. Model Architecture
5. Training Procedure
6. Experimental Results
7. Key Findings & Conclusions

---

## Background: Language Modeling Approaches

**Autoregressive Models (e.g., GPT)**
- Generate tokens sequentially, left-to-right
- Strong performance but fixed generation order
- Cannot easily revise or refine earlier tokens

**Masked Language Models (e.g., BERT)**
- Bidirectional context understanding
- Good for understanding tasks, less for generation

**Diffusion Models**
- Proven success in image generation (DALL-E, Stable Diffusion)
- Iterative refinement process
- Natural fit for non-autoregressive generation

---

## Motivation: Why Diffusion for Language?

**Key Advantages:**
- **Flexibility**: Can generate in any order, not just left-to-right
- **Refinement**: Iterative improvement of outputs
- **Controllability**: Better control over generation process
- **Parallelization**: Potential for faster generation

**Challenge:**
- Language is discrete, diffusion was designed for continuous spaces
- Need effective way to apply diffusion to discrete tokens

---

## Problem Statement

**Goal**: Develop a diffusion-based language model that:
1. Handles discrete token spaces effectively
2. Matches or exceeds autoregressive model performance
3. Enables flexible, non-autoregressive generation
4. Allows for controllable text generation

**Key Question**: How do we adapt continuous diffusion processes to discrete language tokens?

---

## Masked Diffusion Framework

**Core Idea**: Use masking as the noise process

**Forward Process (Adding Noise)**:
- Gradually mask out tokens in the text
- Replace real tokens with [MASK] tokens
- Controlled by a noise schedule

**Reverse Process (Denoising)**:
- Model learns to predict masked tokens
- Iteratively unmask tokens
- Generate complete text from fully masked state

---

## Masking Schedule

**Forward Process**:
$$q(x_t | x_0) = \prod_{i=1}^{n} q(x_t^{(i)} | x_0^{(i)}, t)$$

where tokens are masked with probability determined by the schedule.

**Key Parameters**:
- Number of diffusion steps $T$
- Masking rate at each step $t$
- Can use cosine, linear, or learned schedules

**Properties**:
- At $t=0$: Original text (no masks)
- At $t=T$: Fully masked text
- Intermediate steps: Partially masked

---

## Model Architecture

**Base Architecture**: Transformer encoder-decoder or decoder-only

**Key Components**:
1. **Token Embeddings**: Convert tokens to continuous representations
2. **Positional Encodings**: Maintain sequence position information
3. **Time Step Embeddings**: Encode current diffusion step $t$
4. **Transformer Layers**: Process masked sequences
5. **Output Layer**: Predict tokens for masked positions

**Time Conditioning**: Model needs to know which diffusion step it's at

---

## Training Procedure

**Objective**: Learn to predict original tokens from masked versions

**Training Algorithm**:
1. Sample a text sequence $x_0$ from dataset
2. Sample a diffusion step $t \sim \text{Uniform}(1, T)$
3. Create masked version $x_t$ by masking tokens
4. Model predicts original tokens: $p_\theta(x_0 | x_t, t)$
5. Compute loss (e.g., cross-entropy) on masked positions
6. Backpropagate and update parameters

**Loss Function**:
$$\mathcal{L} = \mathbb{E}_{x_0, t} [-\log p_\theta(x_0 | x_t, t)]$$

where $p_\theta(x_0 | x_t, t) = f_\theta(x_t, t)$ is the model's prediction.

---

## Inference: Sampling Procedure

**Generation Process**:
1. Start with fully masked sequence: $x_T = [\text{MASK}]^n$
2. For $t = T, T-1, ..., 1$:
   - Predict tokens for masked positions
   - Sample or select tokens for some positions
   - Keep remaining positions masked
3. Output final unmasked sequence $x_0$

**Sampling Strategies**:
- **Deterministic**: Always take argmax prediction
- **Stochastic**: Sample from predicted distribution
- **Hybrid**: Mix of both approaches

---

## Key Innovation: Semi-Autoregressive Generation

**Flexibility in Generation Order**:
- Not restricted to left-to-right generation
- Can unmask tokens in any order
- Learn optimal unmasking order from data

**Advantages**:
- Parallel generation of multiple tokens
- Ability to refine uncertain predictions
- More natural for certain tasks

---

## Experimental Setup

**Datasets**:
- Text generation benchmarks (e.g., WikiText, C4)
- Machine translation tasks
- Conditional generation tasks

**Baselines**:
- GPT-style autoregressive models
- Other non-autoregressive models
- Previous diffusion language models

**Metrics**:
- Perplexity
- BLEU score (for translation)
- Generation quality (human evaluation)
- Inference speed

---

## Results: Generation Quality

**Key Findings**:
- Competitive or superior perplexity compared to autoregressive models
- High-quality text generation
- Better performance on long sequences
- Improved coherence and consistency

**Comparison with Baselines**:
- Outperforms previous diffusion-based language models
- Approaches autoregressive model performance
- Significantly better than other non-autoregressive methods

---

## Results: Generation Speed

**Inference Efficiency**:
- Faster than autoregressive models for long sequences
- Parallelizable token prediction
- Trade-off between quality and speed via number of steps

**Speedup Analysis**:
- $O(T)$ generation steps (where $T$ = number of diffusion steps)
- vs $O(n)$ for autoregressive (where $n$ = sequence length)
- Can adjust $T$ for speed-quality trade-off
- Particularly efficient with modern hardware (GPUs/TPUs)

---

## Results: Controllability

**Advantages for Controlled Generation**:
- Easy to incorporate constraints
- Can fix certain tokens and generate around them
- Supports iterative refinement
- Better for infilling tasks

**Applications**:
- Text infilling and editing
- Constrained generation
- Style transfer
- Semantic preservation

---

## Ablation Studies

**Key Investigations**:
1. **Masking Schedule**: Impact of different noise schedules
2. **Number of Steps**: Quality vs efficiency trade-off
3. **Sampling Strategy**: Deterministic vs stochastic
4. **Architecture Choices**: Model size and depth effects

**Findings**:
- Cosine schedule generally works well
- More steps improve quality with diminishing returns
- Optimal strategy depends on application
- Larger models benefit more from diffusion approach

---

## Qualitative Analysis

**Generation Examples**:
- Coherent long-form text
- Semantically consistent outputs
- Diverse generations with different sampling
- Natural language flow

**Failure Modes**:
- Occasional repetition
- Challenges with very long contexts
- Some grammatical inconsistencies in early steps

---

## Key Contributions

1. **Masked Diffusion Framework**: Effective adaptation of diffusion to discrete text
2. **Competitive Performance**: Matches autoregressive model quality
3. **Flexible Generation**: Non-autoregressive, order-agnostic generation
4. **Practical Applications**: Speed advantages and controllability benefits

---

## Limitations and Future Work

**Current Limitations**:
- Still requires multiple inference steps
- Memory overhead during training
- Optimization challenges for very large models

**Future Directions**:
- Learned masking schedules
- Integration with retrieval mechanisms
- Applications to multimodal generation
- Improved sampling algorithms
- Distillation for faster inference

---

## Impact and Applications

**Potential Use Cases**:
- Text editing and revision
- Interactive writing assistants
- Code generation and refactoring
- Creative writing tools
- Document completion

**Broader Impact**:
- New paradigm for language generation
- Inspiration for future non-autoregressive models
- Opens new research directions

---

## Conclusion

**Summary**:
- Masked diffusion provides effective framework for language modeling
- Achieves competitive quality with autoregressive models
- Offers unique advantages in flexibility and controllability
- Opens new possibilities for language generation

**Key Takeaway**: Diffusion models are not just for images - they can be powerful tools for discrete language generation with the right framework.

---

## Questions?

<div style="text-align: center; margin-top: 100px;">

Thank you for your attention!

**Paper**: [arxiv.org/abs/2406.07524](https://arxiv.org/abs/2406.07524)

</div>

---

## Additional Resources

**Related Papers**:
- Diffusion-LM (2022)
- Analog Bits (2023)
- SUNDAE (2023)
- Discrete Diffusion Models (2021)

**Code and Implementations**:
- Check the paper's GitHub repository
- PyTorch diffusion libraries
- Hugging Face model implementations

**Further Reading**:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Understanding Diffusion Models" (Various tutorials)
