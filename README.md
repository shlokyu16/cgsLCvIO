# cgsLCvIO: Living Creature vs. Inanimate Object Classification

A concept-categorization experiment comparing three computational paradigms in cognitive science: **Symbolic (rule-based)**, **Connectionist (neural network)**, and **Bayesian (probabilistic inference)** — plus a **Hybrid model** that combines all three.

The task is simple by design: classify entities as *Living* or *Object* based on five perceptual features — shape, material type, mobility, vocality, and surface hardness. What makes it interesting is the deliberate inclusion of boundary cases (taxidermied animals, animatronic robots, fur rugs) that stress-test each paradigm's assumptions and reveal where rule-rigidity, statistical opacity, and independence violations break down.

This experiment accompanies a reflective article on CGS 001: Introduction to Cognitive Science at UC Davis, and a formal paper examining these paradigms through Marr's three levels of analysis and Bermúdez's argument for cognitive pluralism.

---

## Models

| Model | Approach | Test Accuracy | CV Accuracy |
|---|---|---|---|
| Symbolic | Hand-coded if-else heuristics (PSSH) | 82.6% | — |
| Connectionist | Multi-Layer Perceptron (32→16 neurons) | 92.8% | 92.7% ± 2.8% |
| Bayesian | Naive Bayes (probabilistic inference) | 82.6% | 71.9% ± 9.1% |
| **Hybrid** | Symbolic prior + MLP + Naive Bayes ensemble | **89.9%** | **96.3% ± 1.6%** |

---

## Dataset

274 synthetic entities described across five categorical features. Stratified 75/25 train-test split, near-balanced class distribution (51.8% Living / 48.2% Object). Fuzzy boundary cases represent ~18% of the dataset.

| Feature | Values |
|---|---|
| Shape | Irregular, Regular, Organic |
| Material | Skin, Fur, Wood, Grass, Metal, Plastic, Other |
| Mobility | Yes / No |
| Vocality | Yes / No |
| Surface Hardness | Soft/Smooth, Hard/Rough |

---

## Read More

- **Article (Medium):** [Introduction to Cognitive Science: A Step Closer to My Aspiration](https://lnkd.in/gJ54FvNq)
- **Paper:** *Categorizing the Living: A Comparative Study of Symbolic, Connectionist, and Bayesian Models in Perceptual Concept Classification*

---

## References

- Bermúdez, J. L. (2020). *Cognitive Science: An Introduction to the Science of the Mind* (4th ed.)
- Marr, D. (1982). *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*
- Newell, A., & Simon, H. A. (1976). Computer science as empirical inquiry. *Communications of the ACM*
- Rumelhart, D. E., & McClelland, J. L. (1986). *Parallel Distributed Processing, Vol. 1*
- Searle, J. R. (1980). Minds, brains, and programs. *Behavioral and Brain Sciences*
