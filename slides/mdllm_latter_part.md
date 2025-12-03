---
title: "Masked Diffusion Language Models Latter Part"
customTheme: "https://comping-style.qihang-zhang.com/stylesheets/slides.css"
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

1. Sampling/Inference Process
2. Experimental Results
3. Key Findings & Conclusions

---

## Sampling / Inference Process

---

## Experimental Setting: 

**PPL**: Perplexity evaluates how well the model predicts the reference tokens:

$$
\mathrm{PPL} = \exp \left(\mathbb{E} _{x \sim \mathbb{P} _{data}} - \frac{1}{N(x)} \sum _{i=1}^{N(x)} \log P(x _i \mid x _{<i})\right),
$$

Where:
+ $N(x)$ is the number of tokens of the sequence $x$.
+ PPL is indeed the exponential of the average negative log-likelihood per token.

---
## Calculate PPL for AR and Diffusion Models

- **AR Models**: We can directly compute PPL.
- **Diffusion Models**: We approximate PPL.
    <!-- TODO: estimate the bound of PPL-->

---
## 

---
## Table 1: Main Results on Language Modeling Benchmarks

<div style="font-size: 28px; line-height: 1.35;">
  <div style="margin: 0 0 12px 0;">
    Table 1: Test perplexities (PPL; &darr;) on LM1B. &dagger;Reported in He et al. [26]. Best diffusion value is bolded.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left;">
    <thead>
      <tr>
        <th style="width: 26%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 44%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 15%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Parameters</th>
        <th style="width: 15%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-style: italic;" rowspan="2">Autoregressive</td>
        <td>Transformer-X Base [13]</td>
        <td style="text-align: center;">0.46B</td>
        <td style="text-align: center;">23.5</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">OmniNet<sub>T</sub> [61]</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">100M</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">21.5</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-top: 1px solid #000;" rowspan="5">Diffusion</td>
        <td style="border-top: 1px solid #000;">BERT-Mouth [64]&dagger;</td>
        <td style="text-align: center; border-top: 1px solid #000;">110M</td>
        <td style="text-align: center; border-top: 1px solid #000;">&le;142.89</td>
      </tr>
      <tr>
        <td>D3PM (absorb) [1]</td>
        <td style="text-align: center;">70M</td>
        <td style="text-align: center;">&le;76.90</td>
      </tr>
      <tr>
        <td>Diffusion-LM [30]&dagger;</td>
        <td style="text-align: center;">80M</td>
        <td style="text-align: center;">&le;118.62</td>
      </tr>
      <tr>
        <td>DiffusionBert [26]</td>
        <td style="text-align: center;">110M</td>
        <td style="text-align: center;">&le;63.78</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">SEDD [33] (33B tokens)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">&le; 32.79</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-top: 1px solid #000;" rowspan="2">Autoregressive<br>(Retrained)</td>
        <td style="border-top: 1px solid #000;">Transformer (33B tokens)</td>
        <td style="text-align: center; border-top: 1px solid #000;">110M</td>
        <td style="text-align: center; border-top: 1px solid #000;">22.32</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Transformer (327B tokens)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">20.86</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-top: 1px solid #000;" rowspan="2">Diffusion<br>(Ours)</td>
        <td style="border-top: 1px solid #000;">MDLM (33B tokens)</td>
        <td style="text-align: center; border-top: 1px solid #000;">110M</td>
        <td style="text-align: center; border-top: 1px solid #000;">&le;27.04</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">MDLM (327B tokens)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">&le;23.00</td>
      </tr>
    </tbody>
  </table>
</div>

---
## Table 2
