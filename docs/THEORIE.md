# La régularisation d'information explicite, en clair

> 🇬🇧 English version: [THEORY.md](THEORY.md)

Ce document donne la théorie minimale pour comprendre ce que mesure le benchmark. Chaque section se termine
par le lien avec le code. Notations : X l'entrée, Y la cible, Z une représentation interne, θ les poids,
p = softmax(logits) la prédiction, K le nombre de classes, u la loi uniforme sur les K classes.

---

## 1. Régulariser : contraindre la capacité effective

Un modèle sur-paramétré peut ajuster parfaitement n'importe quel jeu d'entraînement, y compris des
étiquettes aléatoires (Zhang et al., 2017). Ce qui le fait généraliser n'est donc pas sa capacité brute mais
la **contrainte** qui l'empêche d'utiliser toute cette capacité. On distingue :

| | Régularisation **implicite** | Régularisation **explicite** |
|---|---|---|
| D'où elle vient | de la procédure : SGD, early stopping, initialisation, architecture | d'un terme ajouté volontairement à la perte ou au modèle |
| Exemples | SGD à petit batch, arrêt précoce | weight decay, dropout, label smoothing, VIB |
| Contrôle | indirect | direct, par un hyperparamètre (λ, β, ε) |

La régularisation explicite s'écrit toujours

```
L(θ) = CE(Y, Ŷ) + R
```

et toute la question est **ce que mesure R**.

## 2. Poids ou information : deux façons de contraindre

**Weight decay** pénalise la norme des poids : `R = λ ‖θ‖²`. En termes bayésiens c'est un prior gaussien sur θ
(estimation MAP). La contrainte porte sur *où sont les paramètres*, pas sur *ce qu'ils encodent*. Deux modèles
de même norme peuvent stocker des quantités d'information très différentes sur les données.

La **régularisation d'information** pénalise directement la quantité d'information que le modèle stocke ou
transmet. Elle s'exprime avec trois quantités :

| Quantité | Définition | Ce qu'elle mesure |
|---|---|---|
| Entropie | `H(p) = −Σ_k p_k log p_k` | l'incertitude d'une prédiction. Max = log K (uniforme), min = 0 (certain) |
| Divergence KL | `KL(p ‖ q) = Σ_k p_k log(p_k / q_k)` | le « coût » d'utiliser q à la place de p. Asymétrique, ≥ 0 |
| Information mutuelle | `I(X; Z) = E_x [ KL( p(z\|x) ‖ p(z) ) ]` | combien Z dit sur X en moyenne. 0 si indépendants |

L'idée est : **un modèle qui a peu d'information sur son entraînement ne peut pas l'avoir mémorisé.**
Ce n'est pas qu'une intuition. Plusieurs bornes de généralisation sont contrôlées par une information
mutuelle : entre représentation et entrée (Shamir, Sabato & Tishby, 2010), ou entre poids et données
d'entraînement (Xu & Raginsky, 2017 ; Achille & Soatto, 2018). Moins d'information, plus petit écart
train/test garanti.

## 3. Le principe du goulot d'étranglement informationnel (IB)

Tishby, Pereira & Bialek (1999) posent la question : quelle est la meilleure représentation Z de X pour
prédire Y ? Réponse : celle qui **compresse** X au maximum tout en **conservant** tout ce qui sert à Y.

```
min_θ   I(X ; Z)  −  λ · I(Z ; Y)
        └─ compression    └─ prédictivité
```

Deux remarques rendent ce principe opérationnel :

1. **Le terme de prédiction, c'est la cross-entropy.** Par définition `I(Z;Y) = H(Y) − H(Y|Z)`, et la
   cross-entropy du classifieur est une borne supérieure de `H(Y|Z)`. Maximiser `I(Z;Y)` revient donc à
   minimiser la CE habituelle.
2. **Le terme de compression est le régularisateur.** En réarrangeant, l'objectif IB devient exactement

```
L = CE(Y, Ŷ) + β · I(X ; Z)          avec β = 1/λ
```

C'est la forme canonique de la régularisation d'information : la CE classique plus une pénalité sur
l'information que la représentation garde de l'entrée. β règle le compromis. β = 0 : pas de compression,
modèle standard. β grand : Z ne garde presque rien, le modèle sous-apprend.

## 4. Le Variational Information Bottleneck (VIB)

Problème : `I(X;Z)` demande la loi marginale `p(z) = ∫ p(z|x) p(x) dx`, incalculable pour un réseau.
Alemi et al. (2017) contournent le problème avec une **borne variationnelle**. Pour n'importe quelle loi
`r(z)` choisie librement :

```
I(X;Z) = E_x[ KL(q(z|x) ‖ p(z)) ]
       = E_x[ KL(q(z|x) ‖ r(z)) ] − KL(p(z) ‖ r(z))
       ≤ E_x[ KL(q(z|x) ‖ r(z)) ]                     car KL ≥ 0
```

On remplace donc la vraie information mutuelle par sa borne supérieure, calculable exemple par exemple.
Le choix standard rend tout explicite :

- l'encodeur produit une gaussienne `q(z|x) = N(μ(x), diag σ²(x))`,
- le prior est `r(z) = N(0, I)`,
- la KL a alors une forme fermée, par dimension `j` :

```
KL( q(z|x) ‖ N(0,I) ) = ½ Σ_j ( μ_j² + σ_j² − log σ_j² − 1 )
```

À l'entraînement on échantillonne `z = μ + σ ⊙ ε`, `ε ~ N(0,I)` (astuce de reparamétrisation), et la perte est

```
L_VIB = CE(Y, classifieur(z)) + β · KL( q(z|x) ‖ N(0,I) )
```

Lecture intuitive : le bruit `σ` détruit l'information, la KL pousse `μ` vers 0 et `σ` vers 1 (donc vers
« ne rien transmettre »), et seule la CE justifie de garder un canal ouvert sur une dimension. Les dimensions
qui n'aident pas la prédiction se ferment. C'est un goulot **appris et dépendant de la tâche**.

Mahabadi, Belinkov & Henderson (2021) placent ce goulot entre un encodeur pré-entraîné et la tête de
classification pendant le fine-tuning. En faible ressource, il réduit le sur-apprentissage et rend le
modèle moins sensible aux biais du dataset (raccourcis lexicaux de MNLI, testés sur HANS). C'est le setup
de ce repo.

> **Dans le code** : `inforeg/model.py`, classe `Classifier`, tête `"vib"`. L'encodeur est un MLP
> `hidden → hidden/2 → 2·z_dim` qui sort `(μ, log σ²)`. La KL est renvoyée dans `outputs["kl"]` et
> multipliée par `beta` dans `inforeg/losses.py`. La valeur `val_kl` dans `metrics.json` est la borne sur
> I(X;Z) du modèle final : c'est **combien d'information il a gardé**.

## 5. Régulariser les sorties : label smoothing et pénalité de confiance

On peut aussi contraindre l'information dans la **prédiction** plutôt que dans la représentation. Une
sortie très piquée (entropie ≈ 0) sur le train est le symptôme classique de la mémorisation.

**Pénalité de confiance** (Pereyra et al., 2017). On récompense l'entropie :

```
L_CP = CE − β · H(p)
```

Or `H(p) = log K − KL(p ‖ u)`. Donc, à une constante près, `L_CP = CE + β · KL(p ‖ u)` : on tire la
prédiction vers l'uniforme, au sens KL(p ‖ u).

**Label smoothing** (Szegedy et al., 2016). On remplace la cible one-hot `δ_y` par `(1−ε) δ_y + ε u` :

```
L_LS = (1−ε) · CE(δ_y, p) + ε · CE(u, p)
     = (1−ε) · CE(δ_y, p) + ε · [ H(u) + KL(u ‖ p) ]
```

Soit, à une constante et un facteur près, `L_LS = CE + ε' · KL(u ‖ p)`.

Les deux méthodes sont donc **la même idée avec la KL dans les deux sens** :

| | Terme | Sens | Comportement |
|---|---|---|---|
| Pénalité de confiance | `KL(p ‖ u)` | mode-seeking | pénalise surtout les classes où p est très grand ; laisse des zéros |
| Label smoothing | `KL(u ‖ p)` | mass-covering | pénalise fortement toute classe où p → 0 ; interdit les zéros |

Aucune des deux ne contraint la représentation interne : elles agissent seulement sur la couche de sortie.
C'est ce qui les rend triviales à implémenter et qui limite aussi leur portée : un modèle peut rester
sur-confiant en interne et n'aplatir que sa dernière couche.

> **Dans le code** : `inforeg/losses.py`. `label_smoothing` passe par `F.cross_entropy(..., label_smoothing=ε)`,
> `confidence_penalty` calcule `H(p)` et retourne `reg = −β · H`.

## 6. Régulariser la politique : la KL vers un modèle de référence

Dans le post-training des LLM (RLHF, DPO, GRPO), l'objectif est

```
max_π  E[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

La solution a une forme fermée `π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )`. La KL borne l'information
que la nouvelle politique peut acquérir par rapport au modèle de départ : c'est le même mécanisme que le
VIB, avec `π_ref` dans le rôle du prior `r(z)`. Sans ce terme, la politique s'effondre sur quelques
réponses à haute récompense (mode collapse, « reward hacking »). Des travaux récents affinent cette idée :
IBRO applique un IB au niveau des tokens de raisonnement, Forgetting-MarI l'utilise en unlearning pour
retirer uniquement l'information *marginale* apportée par les données à oublier.

> **Dans le code** : pas encore implémenté (feuille de route). En classification, la tête de sortie est
> neuve, il n'y a pas de `π_ref` naturel ; cette méthode a du sens sur une tâche de génération.

## 7. Régulariser la trajectoire : densité d'information uniforme (UID)

Pour un modèle de langage, la surprise du token `t` est `s_t = −log p(x_t | x_<t)`. L'hypothèse UID
(psycholinguistique) dit qu'un locuteur optimal répartit l'information uniformément dans la phrase.
Wei, Meister & Cotterell (2021) en font un régularisateur :

```
L_UID = MLE + β · Var_t( s_t )
```

Il améliore la perplexité surtout à faible volume de données, et la diversité lexicale des générations.
Ce n'est pas une compression de l'information mais une contrainte sur sa **distribution temporelle**.

> **Dans le code** : pas encore implémenté (feuille de route), nécessite une tâche de génération.

## 8. Ce qu'on attend empiriquement, et ce que mesure le benchmark

| Effet attendu de la régularisation d'information | Métrique du repo | Pourquoi |
|---|---|---|
| Moins de mémorisation | `gap_nll = val_nll − train_nll` | un modèle qui stocke peu d'information sur le train ne peut pas y être beaucoup meilleur qu'en val |
| Meilleure généralisation hors domaine | `hans_acc`, par heuristique | les raccourcis lexicaux de MNLI sont de l'information non prédictive que le goulot doit fermer |
| Meilleure calibration | `val_ece`, diagrammes de fiabilité | les pénalités de sortie agissent directement sur la confiance ; le VIB réduit la confiance via le bruit |
| Compromis compression / performance | `vib_beta_sweep.png` : `val_acc`, `val_kl` en fonction de β | l'objectif IB est un compromis, on doit voir une courbe en cloche |
| Effet plus fort quand les données sont rares | comparaison n_train = 1000 vs 5000 | avec beaucoup de données, la CE seule suffit à écarter l'information non prédictive |

Deux limites à garder en tête en lisant les résultats :

1. **La borne VIB n'est pas I(X;Z).** `val_kl` est une borne supérieure, et son écart à la vraie
   information mutuelle dépend de la qualité du prior `N(0,I)`. Elle sert à comparer des runs entre eux,
   pas comme mesure absolue.
2. **Compression n'implique pas toujours généralisation.** Le lien IB ↔ généralisation est débattu
   (Saxe et al., 2018, montrent que la phase de compression n'est pas systématique). Le benchmark teste
   précisément si l'effet est là dans le régime faible ressource + LoRA.

## 9. Résumé en une phrase par méthode

- **Weight decay** : « garde les poids petits » — contrainte sur θ, aveugle au contenu.
- **Label smoothing** : « ne mets jamais zéro sur une classe » — KL(u ‖ p) sur la sortie.
- **Pénalité de confiance** : « ne sois pas trop sûr » — KL(p ‖ u) sur la sortie.
- **VIB** : « ne transmets de l'entrée que ce qui sert à la cible » — borne sur I(X;Z) dans la représentation.
- **KL vers référence** : « ne t'éloigne du modèle de départ que si la récompense le justifie » — même mécanisme, sur la politique.
- **UID** : « répartis l'information uniformément dans la séquence » — contrainte sur la trajectoire.

## Références citées ici

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
