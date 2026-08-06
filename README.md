# RegretGCD

**RegretGCD learns when a parametric GCD classifier should defer to a transductive prototype classifier.**

The repository previously investigated DeltaSub, a selective subtoken-refinement method. Those experiments found a real oracle opportunity but no label-free patch selector that reliably converted it into improved generalized category discovery. DeltaSub is therefore archived as a negative result. RegretGCD keeps the useful lesson and discards the failing mechanism: instead of asking *which patch should be refined?*, it asks *which of two already available category decisions is less likely to be wrong for this sample?*

This is not a decorative mixture-of-experts layer. It is a bounded, leak-resistant learning-to-defer experiment with a predeclared Gate 1. If the class-held-out router fails the gate, the method is abandoned. Science occasionally benefits from an exit condition, a concept humans otherwise reserve for fire drills.

## Current status

Gate 0 established that the two experts are genuinely complementary on CUB seed 0:

| Method | All | Old | New | H-mean |
|---|---:|---:|---:|---:|
| Matched SelEx parametric expert | 47.8426 | 81.5881 | 14.3986 | 24.4775 |
| Leave-one-out prototype expert | 47.8599 | 74.4799 | 21.4777 | 33.3409 |
| Diagnostic oracle arbitration | 55.5057 | 84.2580 | 27.0103 | 40.9071 |

The oracle improves H-mean by **7.5663 points** over the prototype expert, with a paired-bootstrap interval of **[5.2441, 10.6763] points**. The parametric expert is uniquely correct on 443 samples and the prototype expert on 444 samples. Gate 0 uses test labels and is **only an upper-bound diagnostic**. It is not a deployable method and is never reported as RegretGCD performance.

**Final Gate-1 result: ABANDON.** The labelled known-class unique-winner set contained only one expert outcome. Consequently, the binary regret target was not identifiable without fabricating supervision or using test labels. The router was not fitted, no deployable RegretGCD performance is claimed, and no multi-seed or Aircraft experiments are authorized. Exact diagnostics are stored in `artifacts/regretgcd/cub/seed_0/result.json`.

## Problem formulation

Let the unlabeled evaluation set be

\[
\mathcal U=\{x_i\}_{i=1}^{N},
\]

containing both known and novel categories. RegretGCD has two fixed experts:

- a parametric expert \(E_p\), the matched SelEx classifier;
- a transductive prototype expert \(E_t\), built from multi-view test features without test labels.

The experts output cluster identifiers

\[
\hat y_i^p = E_p(x_i),
\qquad
\hat y_i^t = E_t(x_i).
\]

The router estimates the probability that the prototype expert has lower sample-level regret than the parametric expert:

\[
q_i = \sigma(g_\theta(z_i))
\approx
P(c_{i,t} < c_{i,p}\mid z_i),
\]

where \(z_i\) contains only label-free decision geometry and \(c_{i,e}\) is the error cost of expert \(e\).

At inference, routing is conservative and occurs only when the experts disagree:

\[
\hat y_i =
\begin{cases}
\hat y_i^t,
& \hat y_i^p \neq \hat y_i^t \text{ and } q_i \ge \tau,\\
\hat y_i^p,
& \text{otherwise.}
\end{cases}
\]

The threshold \(\tau\) is selected from class-held-out predictions on labelled training data. Test labels do not train the router, choose features, select \(\tau\), or select an ablation.

## Expert 1: matched parametric classifier

For \(V=4\) deterministic views, the parametric expert produces logits

\[
\ell_{i}^{p,v}\in\mathbb R^C,
\qquad
P_i^{p,v}=\operatorname{softmax}(\ell_i^{p,v}).
\]

The views are:

1. center-cropped original image;
2. low-saturation image;
3. mildly blurred image;
4. mildly dimmed image.

The hard test prediction remains the archived matched SelEx prediction used by Gate 0. Multi-view probabilities are router features, not a silent replacement of the expert.

## Expert 2: transductive leave-one-out prototypes

Let \(h_i^v\in\mathbb R^d\) be the normalized DINOv2 CLS feature for view \(v\). The multi-view feature is

\[
\bar h_i =
\operatorname{norm}\left(
\frac{1}{V}
\sum_{v=1}^{V}
\operatorname{norm}(h_i^v)
\right).
\]

Test samples are grouped by the parametric pseudo-label \(\hat y_i^p\). The prototype for pseudo-class \(c\) is

\[
p_c =
\operatorname{norm}\left(
\sum_{i:\hat y_i^p=c}\bar h_i
\right).
\]

A test sample is excluded from its own assigned prototype:

\[
p_{c,-i} =
\operatorname{norm}\left(
\sum_{j:\hat y_j^p=c,\,j\neq i}\bar h_j
\right).
\]

The view-level prototype score is

\[
s_{ic}^{t,v}=
\frac{\langle h_i^v,p_c^{(-i)}\rangle}{T_t},
\qquad T_t=0.10.
\]

Leave-one-out construction prevents a sample from improving its own prototype score merely by being included in the centroid. Singleton clusters fall back to the full prototype and are counted in the result artifact.

## Regret supervision from labelled known classes

Cluster identifiers are permutation invariant. A Hungarian mapping \(\pi\) is therefore fitted **only on genuinely labelled training examples** using the parametric expert:

\[
\pi^*=
\arg\max_\pi
\sum_{i\in\mathcal L}
\mathbf 1[\pi(\hat y_i^p)=y_i].
\]

Both experts share this pseudo-label space, so the same mapping evaluates their training decisions. Their costs are

\[
c_{i,p}=\mathbf 1[\pi^*(\hat y_i^p)\neq y_i],
\qquad
c_{i,t}=\mathbf 1[\pi^*(\hat y_i^t)\neq y_i].
\]

The signed regret target is

\[
r_i=c_{i,p}-c_{i,t}\in\{-1,0,1\}.
\]

Only unique-winner examples, \(|r_i|=1\), supervise the router. The binary target is

\[
t_i=\mathbf 1[r_i=1],
\]

where \(t_i=1\) means the prototype expert is correct and the parametric expert is wrong. The standardized logistic router minimizes

\[
\mathcal L(\theta)=
-\sum_{i\in\mathcal W}
\left[
 t_i\log q_i+(1-t_i)\log(1-q_i)
\right]
+\lambda\|\theta\|_2^2,
\]

with balanced classes and \(\mathcal W=\{i:|r_i|=1\}\). Logistic regression is intentional. The claim concerns transferable regret estimation, not whether a needlessly ornate MLP can memorize bird species.

## Class-held-out validation

Instance-level random validation is overly optimistic because near-duplicate examples from the same known category may appear in both training and validation. RegretGCD instead partitions **known classes**, not samples, into deterministic folds.

For fold \(k\):

\[
\theta_{-k}
=\operatorname{fit}
\left(
\mathcal L\setminus\mathcal L_k
\right),
\]

and every example from held-out classes \(\mathcal L_k\) receives an out-of-fold regret probability. These probabilities select \(\tau\) and measure unique-winner ROC AUC. A separate instance-held-out router is retained as an ablation to expose any gain from class leakage.

## Router features

All features are scalar, class-permutation invariant, and available without test labels.

### Parametric confidence

\[
\max_c \bar P_{ic}^p,
\qquad
\max_c \bar P_{ic}^p-\max_{c\neq c^*}\bar P_{ic}^p,
\qquad
\frac{H(\bar P_i^p)}{\log C}.
\]

### Prototype confidence and density

The same confidence, margin, and entropy statistics are computed for prototype probabilities. Density features include the top prototype cosine similarity, runner-up similarity, similarity gap, and log pseudo-cluster size.

### Cross-view stability

For each expert, RegretGCD includes view vote agreement and generalized Jensen-Shannon divergence:

\[
\operatorname{JSD}(P_i^1,\ldots,P_i^V)
=
\frac1V\sum_{v=1}^V
D_{KL}(P_i^v\|\bar P_i).
\]

### Cross-expert geometry

Features include expert agreement, symmetric JSD between expert distributions, each expert's probability assigned to the other expert's decision, confidence difference, and entropy difference.

### Knownness

Labelled known-class centroids are computed from training features. Training features use leave-one-out centroids. The router receives only the maximum known-centroid similarity and the top-two similarity gap, not a class identity.

## Gate 1

RegretGCD survives only if the full class-held-out router satisfies every condition against the prototype control:

\[
\Delta\text{All}\ge0,
\qquad
\Delta\text{Old}\ge1.0\text{ point},
\]

\[
\Delta\text{New}\ge-0.5\text{ point},
\qquad
\Delta H\ge1.0\text{ point},
\]

and additionally:

- the fixed-alignment stratified paired-bootstrap lower 95% bound for \(\Delta H\) is positive;
- class-held-out unique-winner AUC is at least 0.55;
- full RegretGCD has strictly higher H-mean than every non-oracle control and ablation.

A pass authorizes three CUB seeds and then three Aircraft seeds. A failure closes RegretGCD. Cars remains `not_run` with reason `dataset unavailable by user choice`.

## Ablations and controls

Gate 1 evaluates all variants from the same cached features:

| Variant | Purpose |
|---|---|
| `matched_selex` | Always use the parametric expert |
| `prototype_control` | Always use the prototype expert |
| `maximum_confidence` | Choose the expert with larger maximum probability |
| `minimum_entropy` | Choose the expert with lower entropy |
| `knownness_only` | Train a router using only known-centroid geometry |
| `global_probability_blend` | Tune a single global mixing coefficient on labelled training data |
| `random_matched_switch_rate` | Randomly switch the same number of disagreements as RegretGCD |
| `instance_holdout_router` | Replace class-held-out validation with sample-held-out validation |
| `without_stability` | Remove view agreement and view JSD |
| `without_density` | Remove prototype density and cluster-size features |
| `without_knownness` | Remove known-centroid features |
| `without_cross_expert` | Remove expert-interaction features |
| `oracle_arbitration_diagnostic` | Test-label upper bound, excluded from the gate |

## Statistical evaluation

Point metrics use the repository's pinned GCD-v2 evaluator with one global Hungarian assignment. The paired confidence interval fixes each method's full-dataset Hungarian mapping, converts predictions to paired correctness indicators, and resamples Old and New strata separately. This avoids the unstable intervals produced when cluster mappings are independently re-estimated inside every bootstrap draw.

For resample \(b\):

\[
\Delta H^{(b)}=
H(A_{old}^{r,b},A_{new}^{r,b})
-
H(A_{old}^{t,b},A_{new}^{t,b}),
\]

where

\[
H(a,b)=\frac{2ab}{a+b}.
\]

## Running Gate 1

Requirements already used by the repository:

- one visible CUDA GPU;
- the pinned DINOv2 source and checkpoint;
- completed matched SelEx artifacts;
- completed Gate-0 prototype artifacts;
- `.venv` with repository dependencies.

Launch the entire bounded experiment from the repository root:

```bash
bash scripts/regretgcd/launch_gate1.sh
```

The launcher detaches from the terminal, redirects stdin from `/dev/null`, and prints exact PID, log, and exit-code paths. Gate 1 is resume-safe:

```text
artifacts/regretgcd/cub/seed_0/
├── labelled_train_cache.npz
├── test_cache.npz
├── router_regretgcd.npz
├── router_regretgcd.json
├── router_*.npz
├── predictions.npz
├── metrics.csv
├── result.json
└── comparison/
    ├── comparison.md
    ├── internal_matched.csv
    └── published_references.csv
```

Inspect the final decision:

```bash
cat artifacts/regretgcd/cub/seed_0/result.json | \
  .venv/bin/python -m json.tool
cat artifacts/regretgcd/cub/seed_0/comparison/comparison.md
```

## SOTA comparison policy

`scripts/regretgcd/compare_sota.py` separates two tables:

1. **matched in-repository methods**, which share samples, expert artifacts, and evaluator;
2. **paper-reported references**, which are not ranked against RegretGCD until their protocols are matched.

The registry currently includes exact DINOv2 values reported by the SubViT and ConGCD papers, plus research metadata for APL, AptGCD, AllGCD, and AFGCD. The current literature increasingly attacks fine-grained GCD through part learning, prompt-based local-global fusion, all-unlabeled contrastive learning, token pruning, visual primitives, and selective subtokenization. RegretGCD is orthogonal: it learns sample-level deferral between a parametric decision and a transductive geometric decision.

A direct SOTA claim is forbidden until the in-repository matched SelEx baseline reproduces the relevant published baseline within a declared tolerance. The existing seed-0 baseline is far below published DINOv2 SelEx references, so combining those values into one ranked table would be numerically easy and scientifically useless.

## Relationship to learning to defer

RegretGCD is related to learning-to-defer and expert-routing work, including consistent deferral estimators and multi-expert routing. It does **not** claim to invent expert arbitration. The defensible research contribution being tested is the GCD-specific combination of:

1. a parametric cluster classifier and a transductive leave-one-out prototype classifier;
2. class-disjoint regret distillation from labelled known classes;
3. class-permutation-invariant decision-geometry features;
4. disagreement-only routing;
5. joint Old/New evaluation under the GCD Hungarian protocol.

No exact prior method with this complete mechanism was found in the papers checked for this implementation. That is a scoped novelty statement, not a proclamation that the literature has been exhaustively conquered.

## Primary references

- Vaze et al., **Generalized Category Discovery**, CVPR 2022: https://arxiv.org/abs/2201.02609
- Wen et al., **Parametric Classification for Generalized Category Discovery: A Baseline Study**, ICCV 2023: https://arxiv.org/abs/2211.11727
- Dai et al., **Adaptive Part Learning for Fine-Grained Generalized Category Discovery**, CVPR 2025: https://arxiv.org/abs/2507.06928
- Zhang et al., **Less Attention is More: Prompt Transformer for Generalized Category Discovery**, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Less_Attention_is_More_Prompt_Transformer_for_Generalized_Category_Discovery_CVPR_2025_paper.html
- Cao et al., **AllGCD**, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Cao_AllGCD_Leveraging_All_Unlabeled_Data_for_Generalized_Category_Discovery_ICCV_2025_paper.html
- Tang et al., **ConGCD**, ICCV 2025: https://arxiv.org/abs/2508.10731
- Zhu et al., **Subtoken Vision Transformer for Fine-grained Recognition**, 2026: https://arxiv.org/abs/2607.09086
- Mozannar and Sontag, **Consistent Estimators for Learning to Defer to an Expert**, ICML 2020: https://proceedings.mlr.press/v119/mozannar20b.html
- Verma et al., **Learning to Defer to Multiple Experts**, AISTATS 2023: https://proceedings.mlr.press/v206/verma23a.html
- Mao, Mohri, and Zhong, **Principled Learning-to-Defer Algorithms for Multiple Experts**, ICML 2025: https://proceedings.mlr.press/v267/mao25a.html

## Archived DeltaSub work

DeltaSub code and artifacts remain in the repository for reproducibility. Its final information-gain selector failed for every tested token budget, and its prototype-conditional selector also failed against the prototype-only control. Those results motivate RegretGCD but do not count as RegretGCD ablations. No further DeltaSub selector experiments are authorized.
