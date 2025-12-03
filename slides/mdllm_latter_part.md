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

## Experiments
+ Language Modelling Benchmarks
+ Downstream NLP Tasks
+ DNA Sequence Modelling

---

## Language Modeling Results: `LM1B`

<div style="display: flex; gap: 28px; align-items: flex-start; font-size: 24px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="flex: 0 0 62%;">
    <div style="font-size: 28px; margin: 0 0 12px 0;">
      Table 1: Test perplexities (PPL; &darr;) on LM1B. &dagger;Reported in <a href="https://arxiv.org/abs/2211.15029">Diffusion-bert</a>. Best diffusion value is bolded.
    </div>
    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
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
  <div style="flex: 1;">
    <div style="font-size: 26px; margin: 0 0 12px 0;">Key takeaways</div>
    <ul style="margin: 0; padding-left: 18px; font-size: 24px; line-height: 1.45;">
      <li><strong>LM1B:</strong> <br> One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling <br><br><strong>Origin:</strong> WMT 2011 English News Crawl<br><strong>Feature:</strong> Only short sentenses, and sentenses are shuffled.</li>
      <li>Our MDLM at 327B tokens sets the best diffusion bound so far (&le;23.00), closing most of the gap to AR.</li>
      <li>Scaling tokens helps both AR and diffusion; moving from 33B to 327B cuts PPL notably for each family.</li>
    </ul>
  </div>
</div>

---

## Table 2

<div style="font-size: 28px; line-height: 1.35; max-width: 620px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 2: Test perplexities (PPL; &darr;) on OWT for models trained for 262B tokens. &dagger; denotes retrained models.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
    <thead>
      <tr>
        <th style="width: 55%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 45%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">AR&dagger;</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">17.54</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">SEDD&dagger;</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">&le;24.10</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">&le;23.21</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 3

<div style="font-size: 26px; line-height: 1.35; max-width: 1080px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 3: Zero-shot perplexities (PPL; &darr;) of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 22px;">
    <thead>
      <tr>
        <th style="width: 16%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PTB</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Wikitext</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">LM1B</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Lambada</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">AG News</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Pubmed</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Arxiv</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">AR (Retrained)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">82.05</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">25.75</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">51.25</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">51.28</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">52.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">49.01</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">41.73</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">SEDD (Retrained)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">100.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">34.28</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">68.20</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">49.86</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">62.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">44.53</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">38.48</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">95.26</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">32.83</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">67.01</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">47.52</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">61.15</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">41.89</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">37.37</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 4

<div style="font-size: 26px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 4: GLUE evaluation results. Evaluation measures (&uarr;) are F1 score for QQP and MRPC, Spearman correlations for STS-B, and accuracy for the rest. For MNLI, we report match/mismatch accuracies.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 22px;">
    <thead>
      <tr>
        <th style="width: 16%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">MNLI<br>(m/mm)</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">QQP</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">QNLI</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">SST-2</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">COLA</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">STS-B</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">MRPC</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">RTE</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Avg</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">AR</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">80.94/80.78</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">86.98</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">86.16</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">90.14</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">33.43</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">84.32</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">83.88</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">47.29</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">74.88</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">BERT</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">84.43/85.35</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">88.41</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">90.46</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">92.20</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">54.81</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">88.41</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">89.16</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">61.37</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">81.62</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">+MDLM-FT</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">84.76/85.07</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">88.49</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">90.30</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">92.20</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">57.69</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">87.48</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">90.53</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">62.09</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">82.06</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 5

<div style="font-size: 28px; line-height: 1.35; max-width: 760px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 5: Semi-AR generative perplexity (Gen. PPL; &darr;) for sequences of 2048 tokens.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
    <thead>
      <tr>
        <th style="width: 35%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 32.5%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Gen. PPL (&darr;)</th>
        <th style="width: 32.5%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Sec/Seq (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">SSD-LM</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">35.43</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">2473.9</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">27.18</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">89.3</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 6

<div style="font-size: 26px; line-height: 1.35; max-width: 780px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 6: Test perplexities (PPL; &darr;) of generative fine-tuning of the Caduceus MLM [50] on the HG38 reference genome. Best diffusion model values are bolded. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for fine-tuning.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 22px;">
    <thead>
      <tr>
        <th style="width: 52%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 22%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Params</th>
        <th style="width: 26%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-style: italic;" rowspan="2">Autoregressive (Retrained)</td>
        <td style="text-align: center;">465K</td>
        <td style="text-align: center;">3.067 &plusmn; .010</td>
      </tr>
      <tr>
        <td style="text-align: center; border-bottom: 1px solid #000;">433K</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">3.153 &plusmn; .001</td>
      </tr>
      <tr>
        <td style="font-style: italic;" rowspan="2">Diffusion (Retrained)</td>
        <td style="text-align: center;">507K</td>
        <td style="text-align: center;">&le; 3.240 &plusmn; .005</td>
      </tr>
      <tr>
        <td style="text-align: center; border-bottom: 1px solid #000;">467K</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">&le; 3.216 &plusmn; .003</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 2px solid #000;">Diffusion (Ours)</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">467K</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">&le; 3.199 &plusmn; .010</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 7

<div style="font-size: 26px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 7: Genomic Benchmarks. Top-1 accuracy (&uarr;) across 5-fold cross-validation (CV) for a pre-trained AR Mamba and a pre-trained Caduceus model fine-tuned with different diffusion parameterizations. The best values per task are bolded and the second best are italicized. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for CV.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 20px;">
    <thead>
      <tr>
        <th style="width: 32%; border-top: 2px solid #000; border-bottom: 1px solid #000;">Model Fine-Tuning Objective<br>(Parameter Count)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Mamba AR<br>(465K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus MLM<br>(467K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus Plaid<br>(507K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus SEDD<br>(467K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus MDLM (ours)<br>(467K)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">Mouse Enhancers</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.763 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.810 (&plusmn;0.016)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.745 (&plusmn;0.079)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.784 (&plusmn;0.058)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.795 (&plusmn;0.029)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Coding vs. Intergenomic</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.897 (&plusmn;0.004)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.908 (&plusmn;0.003)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.005)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.003)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human vs. Worm</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.967 (&plusmn;0.002)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.002)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.971 (&plusmn;0.001)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.003)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Enhancers Cohn</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.734 (&plusmn;0.027)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.737 (&plusmn;0.011)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.743 (&plusmn;0.010)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.746 (&plusmn;0.015)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.743 (&plusmn;0.016)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Enhancer Ensembl</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.856 (&plusmn;0.003)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.907 (&plusmn;0.000)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.885 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.905 (&plusmn;0.006)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.899 (&plusmn;0.004)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Regulatory</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.861 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.874 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.868 (&plusmn;0.010)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.828 (&plusmn;0.037)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.868 (&plusmn;0.004)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human OCR Ensembl</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.806 (&plusmn;0.005)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.821 (&plusmn;0.004)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.820 (&plusmn;0.004)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.816 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.823 (&plusmn;0.008)</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">Human NonTATA Promoters</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">0.926 (&plusmn;0.008)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.014)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.007)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.014)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">0.940 (&plusmn;0.007)</td>
      </tr>
    </tbody>
  </table>
</div>

---

## Table 8

<div style="font-size: 28px; line-height: 1.35; max-width: 620px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 8: Test perplexities (PPL; &darr;) for MDLM ablations on LM1B. For the discrete-time models, we use $T = 1000$. Standard deviation is measured over 5 seeds during evaluation.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
    <thead>
      <tr>
        <th style="width: 60%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 40%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&le;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-weight: bold; border-bottom: 1px solid #000;">MDLM <span style="color: #d55e00;">(47)</span></td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">27.04 &plusmn; .01</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">w/o continuous time <span style="color: #d55e00;">(43)</span></td>
        <td style="text-align: center; border-bottom: 1px solid #000;">27.19 &plusmn; .07</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">&amp; w/o carry-over <span style="color: #d55e00;">(41)</span></td>
        <td style="text-align: center; border-bottom: 1px solid #000;">28.56 &plusmn; .15</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 2px solid #000;">&amp; w/o zero masking <span style="color: #d55e00;">(39)</span></td>
        <td style="text-align: center; border-bottom: 2px solid #000;">28.51 &plusmn; .15</td>
      </tr>
    </tbody>
  </table>
</div>
