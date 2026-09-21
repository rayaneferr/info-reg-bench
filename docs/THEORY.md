# Explicit information regularization, made clear

> 🇫🇷 Version française : [THEORIE.md](THEORIE.md)

This document gives the minimum theory needed to read the benchmark. Each section ends with a pointer to
the code. Notation: X the input, Y the target, Z an internal representation, θ the weights,
p = softmax(logits) the prediction, K the number of classes, u the uniform distribution over the K classes.

---

## 1. Regularizing means constraining effective capacity

An over-parameterized model can fit any training set perfectly, including random labels (Zhang et al., 2017).
What makes it generalize is therefore not its raw capacity but the **constraint** that keeps it from using
all of it. Two kinds:

| | **Implicit** regularization | **Explicit** regularization |
|---|---|---|
| Where it comes from | the procedure: SGD, early stopping, initialization, architecture | a term deliberately added to the loss or the model |
| Examples | small-batch SGD, early stopping | weight decay, dropout, label smoothing, VIB |
| Control | indirect | direct, through a hyperparameter (λ, β, ε) |

Explicit regularization always reads

```
L(θ) = CE(Y, Ŷ) + R
```

and the whole question is **what R measures**.

## 2. Weights or information: two ways to constrain

**Weight decay** penalizes the norm of the weights: `R = λ ‖θ‖²`. In Bayesian terms it is a Gaussian prior
on θ (MAP estimation). The constraint is on *where the parameters are*, not on *what they encode*. Two models
with the same norm can store very different amounts of information about the data.

**Information regularization** directly penalizes the amount of information the model stores or transmits.
It uses three quantities:

| Quantity | Definition | What it measures |
|---|---|---|
| Entropy | `H(p) = −Σ_k p_k log p_k` | uncertainty of a prediction. Max = log K (uniform), min = 0 (certain) |
| KL divergence | `KL(p ‖ q) = Σ_k p_k log(p_k / q_k)` | the "cost" of using q instead of p. Asymmetric, ≥ 0 |
| Mutual information | `I(X; Z) = E_x [ KL( p(z\|x) ‖ p(z) ) ]` | how much Z says about X on average. 0 if independent |

The idea: **a model that holds little information about its training set cannot have memorized it.**
This is more than intuition. Several generalization bounds are controlled by a mutual information: between
representation and input (Shamir, Sabato & Tishby, 2010), or between weights and training data
(Xu & Raginsky, 2017; Achille & Soatto, 2018). Less information, smaller guaranteed train/test gap.

## 3. The Information Bottleneck principle (IB)

Tishby, Pereira & Bialek (1999) ask: what is the best representation Z of X for predicting Y? Answer: the one
that **compresses** X as much as possible while **keeping** everything useful for Y.

```
min_θ   I(X ; Z)  −  λ · I(Z ; Y)
        └─ compression    └─ predictiveness
```

Two remarks make this operational:

1. **The prediction term is the cross-entropy.** By definition `I(Z;Y) = H(Y) − H(Y|Z)`, and the
   classifier's cross-entropy is an upper bound on `H(Y|Z)`. Maximizing `I(Z;Y)` amounts to minimizing the
   usual CE.
2. **The compression term is the regularizer.** Rearranging, the IB objective becomes exactly

```
L = CE(Y, Ŷ) + β · I(X ; Z)          with β = 1/λ
```

This is the canonical form of information regularization: the standard CE plus a penalty on the information
the representation keeps about the input. β sets the trade-off. β = 0: no compression, standard model.
Large β: Z keeps almost nothing, the model underfits.

## 4. The Variational Information Bottleneck (VIB)

Problem: `I(X;Z)` requires the marginal `p(z) = ∫ p(z|x) p(x) dx`, intractable for a network.
Alemi et al. (2017) sidestep it with a **variational bound**. For any freely chosen distribution `r(z)`:

```
I(X;Z) = E_x[ KL(q(z|x) ‖ p(z)) ]
       = E_x[ KL(q(z|x) ‖ r(z)) ] − KL(p(z) ‖ r(z))
       ≤ E_x[ KL(q(z|x) ‖ r(z)) ]                     since KL ≥ 0
```

So the true mutual information is replaced by its upper bound, computable per example. The standard choice
makes everything explicit:

- the encoder outputs a Gaussian `q(z|x) = N(μ(x), diag σ²(x))`,
- the prior is `r(z) = N(0, I)`,
- the KL then has a closed form, per dimension `j`:

```
KL( q(z|x) ‖ N(0,I) ) = ½ Σ_j ( μ_j² + σ_j² − log σ_j² − 1 )
```

At training time we sample `z = μ + σ ⊙ ε`, `ε ~ N(0,I)` (reparameterization trick), and the loss is

```
L_VIB = CE(Y, classifier(z)) + β · KL( q(z|x) ‖ N(0,I) )
```

Intuitive reading: the noise `σ` destroys information, the KL pushes `μ` toward 0 and `σ` toward 1 (i.e.
toward "transmit nothing"), and only the CE justifies keeping a channel open on a dimension. Dimensions that
do not help prediction close. It is a **learned, task-dependent** bottleneck.

Mahabadi, Belinkov & Henderson (2021) put this bottleneck between a pretrained encoder and the classification
head during fine-tuning. In low-resource settings it reduces overfitting and makes the model less sensitive to
dataset biases (MNLI's lexical shortcuts, tested on HANS). That is this repo's setup.

> **In the code**: `inforeg/model.py`, class `Classifier`, head `"vib"`. The encoder is an MLP
> `hidden → hidden/2 → 2·z_dim` producing `(μ, log σ²)`. The KL is returned in `outputs["kl"]` and multiplied
> by `beta` in `inforeg/losses.py`. The `val_kl` value in `metrics.json` is the bound on I(X;Z) of the final
> model: **how much information it kept**.

## 5. Regularizing the outputs: label smoothing and confidence penalty

One can also constrain the information in the **prediction** rather than in the representation. A very peaked
output (entropy ≈ 0) on the training set is the classic symptom of memorization.

**Confidence penalty** (Pereyra et al., 2017). Reward entropy:

```
L_CP = CE − β · H(p)
```

Since `H(p) = log K − KL(p ‖ u)`, up to a constant `L_CP = CE + β · KL(p ‖ u)`: the prediction is pulled toward
uniform, in the KL(p ‖ u) direction.

**Label smoothing** (Szegedy et al., 2016). Replace the one-hot target `δ_y` with `(1−ε) δ_y + ε u`:

```
L_LS = (1−ε) · CE(δ_y, p) + ε · CE(u, p)
     = (1−ε) · CE(δ_y, p) + ε · [ H(u) + KL(u ‖ p) ]
```

That is, up to a constant and a factor, `L_LS = CE + ε' · KL(u ‖ p)`.

The two methods are therefore **the same idea with the KL in opposite directions**:

| | Term | Direction | Behaviour |
|---|---|---|---|
| Confidence penalty | `KL(p ‖ u)` | mode-seeking | mostly penalizes classes where p is very large; tolerates zeros |
| Label smoothing | `KL(u ‖ p)` | mass-covering | strongly penalizes any class where p → 0; forbids zeros |

Neither constrains the internal representation: they act on the output layer only. That is what makes them
trivial to implement, and also what limits them: a model can stay over-confident internally and only flatten
its last layer.

> **In the code**: `inforeg/losses.py`. `label_smoothing` goes through `F.cross_entropy(..., label_smoothing=ε)`,
> `confidence_penalty` computes `H(p)` and returns `reg = −β · H`.

## 6. Regularizing the policy: KL to a reference model

In LLM post-training (RLHF, DPO, GRPO) the objective is

```
max_π  E[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

The solution has the closed form `π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )`. The KL bounds the information
the new policy may acquire relative to the starting model: the same mechanism as the VIB, with `π_ref` playing
the role of the prior `r(z)`. Without it the policy collapses onto a few high-reward answers (mode collapse,
reward hacking). Recent work refines the idea: IBRO applies an IB at the level of reasoning tokens,
Forgetting-MarI uses it for unlearning to remove only the *marginal* information contributed by the data to
forget.

> **In the code**: not implemented yet (roadmap). In classification the output head is new, there is no natural
> `π_ref`; this method makes sense on a generation task.

## 7. Regularizing the trajectory: Uniform Information Density (UID)

For a language model, the surprisal of token `t` is `s_t = −log p(x_t | x_<t)`. The UID hypothesis
(psycholinguistics) says an optimal speaker spreads information uniformly across the sentence.
Wei, Meister & Cotterell (2021) turn it into a regularizer:

```
L_UID = MLE + β · Var_t( s_t )
```

It improves perplexity especially with little data, and the lexical diversity of generations. It is not a
compression of information but a constraint on its **temporal distribution**.

> **In the code**: not implemented yet (roadmap), requires a generation task.

## 8. What to expect empirically, and what the benchmark measures

| Expected effect of information regularization | Repo metric | Why |
|---|---|---|
| Less memorization | `gap_nll = val_nll − train_nll` | a model storing little information about the train set cannot be much better on it than on val |
| Better out-of-domain generalization | `hans_acc`, per heuristic | MNLI's lexical shortcuts are non-predictive information the bottleneck should close |
| Better calibration | `val_ece`, reliability diagrams | output penalties act directly on confidence; VIB lowers confidence through noise |
| Compression / performance trade-off | `vib_beta_sweep.png`: `val_acc`, `val_kl` vs β | the IB objective is a trade-off, we should see a bell curve |
| Stronger effect with scarce data | n_train = 1000 vs 5000 | with a lot of data, CE alone suffices to discard non-predictive information |

Two limits to keep in mind when reading the results:

1. **The VIB bound is not I(X;Z).** `val_kl` is an upper bound, and its gap to the true mutual information depends
   on how good the `N(0,I)` prior is. It is for comparing runs to each other, not an absolute measure.
2. **Compression does not always imply generalization.** The IB ↔ generalization link is debated (Saxe et al.,
   2018, show the compression phase is not systematic). The benchmark tests precisely whether the effect is there
   in the low-resource + LoRA regime.

## 9. One sentence per method

- **Weight decay**: "keep the weights small" — a constraint on θ, blind to content.
- **Label smoothing**: "never put zero on a class" — KL(u ‖ p) on the output.
- **Confidence penalty**: "don't be too sure" — KL(p ‖ u) on the output.
- **VIB**: "transmit from the input only what serves the target" — bound on I(X;Z) in the representation.
- **KL to reference**: "move away from the starting model only if the reward justifies it" — same mechanism, on the policy.
- **UID**: "spread information evenly along the sequence" — a constraint on the trajectory.

## References cited here

- Tishby, Pereira, Bialek. *The Information Bottleneck Method.* 1999. [arXiv:physics/0004057](https://arxiv.org/abs/physics/0004057)
- Shamir, Sabato, Tishby. *Learning and generalization with the information bottleneck.* TCS 2010.
- Szegedy et al. *Rethinking the Inception Architecture.* CVPR 2016 (label smoothing). [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Alemi, Fischer, Dillon, Murphy. *Deep Variational Information Bottleneck.* ICLR 2017. [arXiv:1612.00410](https://arxiv.org/abs/1612.00410)
- Pereyra, Tucker, Chorowski, Kaiser, Hinton. *Regularizing Neural Networks by Penalizing Confident Output Distributions.* 2017. [arXiv:1701.06548](https://arxiv.org/abs/1701.06548)
- Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding deep learning requires rethinking generalization.* ICLR 2017. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Xu, Raginsky. *Information-theoretic analysis of generalization capability of learning algorithms.* NeurIPS 2017. [arXiv:1705.07809](https://arxiv.org/abs/1705.07809)
- Achille, Soatto. *Emergence of Invariance and Disentanglement in Deep Representations.* JMLR 2018. [arXiv:1706.01350](https://arxiv.org/abs/1706.01350)
- Saxe et al. *On the Information Bottleneck Theory of Deep Learning.* ICLR 2018.
- McCoy, Pavlick, Linzen. *Right for the Wrong Reasons (HANS).* ACL 2019. [arXiv:1902.01007](https://arxiv.org/abs/1902.01007)
- Mahabadi, Belinkov, Henderson. *Variational Information Bottleneck for Effective Low-Resource Fine-Tuning.* ICLR 2021. [arXiv:2106.05469](https://arxiv.org/abs/2106.05469)
- Wei, Meister, Cotterell. *A Cognitive Regularizer for Language Modeling.* ACL 2021. [arXiv:2105.07144](https://arxiv.org/abs/2105.07144)
- *Revisiting LLM Reasoning via Information Bottleneck (IBRO).* 2025. [arXiv:2507.18391](https://arxiv.org/abs/2507.18391)
- *Forgetting-MarI: LLM Unlearning via Marginal Information Regularization.* 2025. [arXiv:2511.11914](https://arxiv.org/abs/2511.11914)
