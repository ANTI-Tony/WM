# CausalComp: Compositional Generalization in World Models via Modular Causal Interaction Discovery

**Target:** ICLR 2027 (deadline ~Sep-Oct 2026)
**Format:** 9 pages + unlimited appendix

---

## Abstract (~200 words)

Current world models either learn causal structure but fail at compositional generalization (STICA, OOCDM), or achieve compositionality but lack causal reasoning (DreamWeaver, FIOC-WM). We propose CausalComp, which unifies both: it discovers causal interaction graphs between objects AND learns type-specific dynamics modules, enabling zero-shot generalization to novel object-interaction combinations never seen during training. On our compositional physics benchmark, CausalComp reduces the compositional generalization gap by X% compared to the strongest baseline, while achieving Y% F1 on causal graph discovery.

---

## 1. Introduction (1.5 pages)

**Hook:** World models that can predict "what happens next" are fundamental to planning, reasoning, and decision-making. But current models fail when encountering novel combinations of objects and interactions not seen during training.

**Gap:** Two lines of work have developed independently:
- Causal world models (OOCDM, STICA): learn causal graphs but are monolithic
- Compositional world models (DreamWeaver, FIOC-WM): learn modular representations but no causal structure

**Claim:** Compositional generalization in dynamics prediction REQUIRES causal structure. We formalize this insight and propose CausalComp.

**Contributions:**
1. CausalComp architecture: causal graph discovery + typed interaction modules
2. Theoretical motivation: why causal factorization enables compositional transfer (cite Richens ICLR 2024)
3. Compositional physics benchmark with MCD-style splits
4. Empirical validation: significant reduction in compositional gap vs. all baselines

---

## 2. Related Work (1 page)

### 2.1 Object-Centric World Models
- Slot Attention (Locatello 2020), SlotFormer (ICLR 2023)
- FIOC-WM (NeurIPS 2025), Dyn-O (NeurIPS 2025)

### 2.2 Causal World Models
- Robust Agents Learn Causal World Models (Richens, ICLR 2024 Oral)
- OOCDM (ICML 2024), VCD (NeurIPS 2022 Workshop)
- Causal-JEPA (arXiv 2026)

### 2.3 Compositional Generalization
- DreamWeaver (ICLR 2025), COMBO (ICLR 2025)
- Measuring Compositional Generalization (Keysers, ICLR 2020)
- HOWM (ICML 2022): equivariance framework

### 2.4 At the Intersection
- WM3C (ICLR 2025): language-guided causal components
- Gap: no work simultaneously discovers visual causal graphs + typed modules + demonstrates compositional generalization

---

## 3. Method (3 pages)

### 3.1 Problem Formulation
- Scene = set of objects with attributes (color, shape, size, material)
- Interactions = typed pairwise causal effects (collision, contact, ...)
- Compositional generalization = zero-shot prediction on novel (object_type_A, object_type_B, interaction_type) triples

### 3.2 Object Discovery (Slot Attention / DINOv2)
- Brief description, not our contribution

### 3.3 Causal Interaction Graph Discovery (Contribution 1)
- Edge existence prediction: MLP on pairwise slot features
- Interaction type classification: Gumbel-Softmax routing to M types
- Interventional verification (optional, for refined graphs)

### 3.4 Modular Causal Dynamics (Contribution 2)
- f_self: object self-evolution (gravity, inertia)
- f_inter[τ]: type-specific interaction module (one MLP per type τ)
- Update: s_{t+1} = MLP([f_self(s_t), Σ_j e_ij * w_τ * f_inter[τ](s_j, s_i)])
- Key insight: modules learned for type τ can be REUSED across any object pair exhibiting that interaction → compositional transfer

### 3.5 Training
- Phase 1: reconstruction only (good slots)
- Phase 2: dynamics + graph with optional collision supervision
- Autoregressive rollout with frame-level loss

---

## 4. Experiments (2.5 pages)

### 4.1 Setup
- Synthetic Physics Environment: colored circles with elastic collisions
- CLEVRER (if available) / Physion
- Compositional split: MCD methodology, hold out 40% of attribute-pair collision combinations
- Metrics: Seen MSE, Unseen MSE, Harmonic Mean, Comp Gap, Graph F1

### 4.2 GT-State Results (validates core idea)

| Method | Seen MSE | Unseen MSE | Comp Gap | Graph F1 |
|--------|----------|------------|----------|----------|
| NoGraph | ? | ? | ? | - |
| FullGraph | ? | ? | ? | - |
| SingleModule | ? | ? | ? | ? |
| **CausalComp** | ? | ? | ? | ? |

### 4.3 Learned-Representation Results
- CNN encoder vs DINOv2 encoder
- End-to-end CausalComp performance

### 4.4 Ablation Study
- w/o causal graph → FullGraph
- w/o type modules → SingleModule
- w/o autoregressive rollout
- w/o collision supervision (Phase 2 only)

### 4.5 Qualitative Analysis
- Visualize discovered causal graphs (colliding pairs → high edge prob)
- Visualize interaction type clusters
- Visualize prediction trajectories: GT vs CausalComp vs baselines

---

## 5. Discussion and Limitations (0.5 pages)

- Slot Attention quality limits end-to-end performance
- Collision supervision uses GT; future work: learn from pure observation
- Type number M is a hyperparameter
- Tested on 2D physics; 3D generalization is future work

---

## 6. Conclusion (0.3 pages)

CausalComp demonstrates that combining causal graph discovery with typed interaction modules enables compositional generalization in world models. Our results support the theoretical insight that causal structure is necessary for robust generalization across novel object combinations.

---

## Appendix

- A: Implementation details (architecture, hyperparameters)
- B: Synthetic environment details
- C: Additional compositional split analysis
- D: More visualizations
- E: Physion/PHYRE results (if available)
