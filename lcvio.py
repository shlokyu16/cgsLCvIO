"""
lcvio_v2.py — Living vs. Inanimate Object (LCVIO) Classification
A comparative study of symbolic, connectionist, and Bayesian cognitive models
on a perceptual concept-categorization task.

Author : Shlok Khare
Dataset: Synthetic, generated via generate_dataset() below
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
np.random.seed(16)


# 1. DATASET GENERATION

# Five perceptual features: Shape, Material, Mobility, Vocality, Surface Hardness
# Binary label: Living / Object
# Fuzzy cases are deliberately seeded to mirror real-world perceptual ambiguity
# (e.g., taxidermied animals, animatronic robots, fur rugs, wind chimes).

def generate_dataset(save_path="lcvio.csv"):
    """
    Generates a synthetic concept-categorization dataset of 350+ samples.
    Templates represent cognitive prototypes (Rosch, 1975); fuzzy entries
    represent boundary cases that challenge rule-rigid classifiers.
    """

    # (Shape, Material, Mobility, Vocality, Surface, Label, EntityType, n_samples)
    templates = [
        # ── Clear Living: Mammals ──────────────────────────────────────────
        ("Irregular", "Skin", "Yes", "Yes", "Soft/Smooth", "Living", "Mammal",      20),
        ("Irregular", "Skin", "Yes", "No",  "Soft/Smooth", "Living", "Mammal",      12),
        ("Irregular", "Skin", "No",  "Yes", "Soft/Smooth", "Living", "Mammal",      10),
        ("Irregular", "Fur",  "Yes", "Yes", "Soft/Smooth", "Living", "Mammal",      18),
        ("Irregular", "Fur",  "Yes", "No",  "Soft/Smooth", "Living", "Mammal",      12),
        ("Irregular", "Fur",  "No",  "Yes", "Soft/Smooth", "Living", "Mammal",       8),

        # ── Clear Living: Reptiles & Amphibians ───────────────────────────
        ("Irregular", "Skin", "Yes", "No",  "Hard/Rough",  "Living", "Reptile",     14),
        ("Irregular", "Skin", "No",  "No",  "Hard/Rough",  "Living", "Reptile",      8),

        # ── Clear Living: Plants & Trees ──────────────────────────────────
        ("Organic",   "Wood", "No",  "No",  "Hard/Rough",  "Living", "Tree",        16),
        ("Organic",   "Grass","No",  "No",  "Soft/Smooth", "Living", "Plant",       10),

        # ── Clear Object: Manufactured / Inorganic ────────────────────────
        ("Regular",   "Metal",   "No",  "No",  "Hard/Rough",  "Object", "Tool",     18),
        ("Regular",   "Plastic", "No",  "No",  "Hard/Rough",  "Object", "Tool",     16),
        ("Regular",   "Wood",    "No",  "No",  "Hard/Rough",  "Object", "Furniture",14),
        ("Irregular", "Metal",   "No",  "No",  "Hard/Rough",  "Object", "Tool",     12),
        ("Irregular", "Plastic", "No",  "No",  "Hard/Rough",  "Object", "Tool",     10),
        ("Irregular", "Wood",    "No",  "No",  "Hard/Rough",  "Object", "Furniture", 8),

        # ── Fuzzy Living: Silent / Immobile Animals ───────────────────────
        # Starfish, corals — alive but don't move or vocalize
        ("Irregular", "Skin",  "No",  "No",  "Hard/Rough",  "Living", "Invertebrate", 8),
        ("Irregular", "Other", "No",  "No",  "Soft/Smooth", "Living", "Invertebrate", 6),

        # ── Fuzzy Object: Robots & Animatronics ──────────────────────────
        # Mobile + vocal but clearly inanimate — biggest challenge for symbolic model
        ("Regular",   "Metal",  "Yes", "Yes", "Hard/Rough",  "Object", "Robot",       8),
        ("Regular",   "Other",  "Yes", "Yes", "Hard/Rough",  "Object", "Animatronic", 6),
        ("Irregular", "Metal",  "Yes", "Yes", "Hard/Rough",  "Object", "Robot",       6),

        # ── Fuzzy Object: Taxidermy & Fur Rugs ───────────────────────────
        # Skin/fur material, irregular shape — but inanimate
        ("Irregular", "Skin",  "No",  "No",  "Soft/Smooth", "Object", "Taxidermy",   8),
        ("Irregular", "Fur",   "No",  "No",  "Soft/Smooth", "Object", "FurRug",      8),

        # ── Fuzzy Object: Interactive Toys & Chimes ──────────────────────
        ("Regular",   "Plastic","No",  "Yes", "Hard/Rough",  "Object", "TalkingToy",  6),
        ("Irregular", "Metal",  "Yes", "Yes", "Hard/Rough",  "Object", "WindChime",   6),

        # ── Fuzzy Object: Organic-Looking Sculptures ──────────────────────
        ("Organic",   "Wood",  "No",  "No",  "Hard/Rough",  "Object", "WoodSculpt",  6),
    ]

    rows = []
    for shape, mat, mob, voc, surf, label, etype, n in templates:
        for _ in range(n):
            rows.append({
                "Shape": shape,
                "Skin/Material Type": mat,
                "Mobility": mob,
                "Vocality": voc,
                "Surface Hardness": surf,
                "Label": label,
                "Entity Type": etype,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=16).reset_index(drop=True)
    df.to_csv(save_path, index=False)

    living  = (df["Label"] == "Living").sum()
    objects = (df["Label"] == "Object").sum()
    print(f"Dataset saved to '{save_path}'")
    print(f"Total: {len(df)} samples | Living: {living} ({living/len(df)*100:.1f}%) | Object: {objects} ({objects/len(df)*100:.1f}%)")
    return df



# 2. PREPROCESSING

def preprocess(df):
    """Encode categorical features and label. Returns encoded arrays + encoders."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    feature_cols = ["shape", "skin/material_type", "mobility", "vocality", "surface_hardness"]
    encoders = {}
    for col in feature_cols + ["label"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[[c + "_enc" for c in feature_cols]].values
    y = df["label_enc"].values
    return X, y, encoders, df



# 3. SYMBOLIC MODEL (Rule-Based)

# Implements Physical Symbol System Hypothesis (PSSH) — explicit heuristics.
# Bug-fixed: case-insensitive comparison; labels mapped via encoder classes.

def make_symbolic_classifier(encoders):
    label_classes = list(encoders["label"].classes_)   # e.g. ['Living', 'Object']
    LIVING = label_classes.index("Living")
    OBJECT = label_classes.index("Object")

    def classify(row):
        material = encoders["skin/material_type"].inverse_transform([row[1]])[0].lower()
        mobility = encoders["mobility"].inverse_transform([row[2]])[0].lower()
        vocality = encoders["vocality"].inverse_transform([row[3]])[0].lower()
        shape = encoders["shape"].inverse_transform([row[0]])[0].lower()

        # Rule 1: biological material → living
        if material in ("skin", "fur", "grass"):
            return LIVING
        # Rule 2: organic wood shape → plant
        if shape == "organic" and material == "wood":
            return LIVING
        # Rule 3: mobile AND vocal → living (heuristic)
        if mobility == "yes" and vocality == "yes":
            return LIVING
        return OBJECT

    return classify


# 4. CROSS-VALIDATION HELPER

def cv_accuracy(model, X, y, k=5):
    """Returns mean ± std accuracy from stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=16)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()


# 5. MAIN PIPELINE

def main():
    # ── Generate / Load Dataset ───────────────────────────────────────────────
    df_raw = generate_dataset("lcvio_v2.csv")
    X, y, encoders, df = preprocess(df_raw)

    label_names = list(encoders["label"].classes_)   # ['Living', 'Object']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=16, stratify=y
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class balance (test) — Living: {(y_test==0).sum()} | Object: {(y_test==1).sum()}\n")

    results_summary = {}

    # ── (A) Symbolic Model ────────────────────────────────────────────────────
    symbolic_fn = make_symbolic_classifier(encoders)
    sym_preds   = np.array([symbolic_fn(row) for row in X_test])

    print("=" * 60)
    print("  SYMBOLIC MODEL (Rule-Based / PSSH)")
    print("=" * 60)
    print(classification_report(y_test, sym_preds, target_names=label_names))
    sym_acc = accuracy_score(y_test, sym_preds)
    results_summary["Symbolic"] = {"accuracy": sym_acc, "cv_mean": None, "cv_std": None}

    # ── (B) Connectionist Model (MLP) ─────────────────────────────────────────
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=16)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_cv_mean, mlp_cv_std = cv_accuracy(
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=16), X, y
    )

    print("=" * 60)
    print("  CONNECTIONIST MODEL (Multi-Layer Perceptron)")
    print("=" * 60)
    print(classification_report(y_test, mlp_preds, target_names=label_names))
    print(f"  5-Fold CV Accuracy: {mlp_cv_mean:.3f} ± {mlp_cv_std:.3f}\n")
    results_summary["MLP"] = {"accuracy": accuracy_score(y_test, mlp_preds),
                               "cv_mean": mlp_cv_mean, "cv_std": mlp_cv_std}

    # ── (C) Bayesian Model (Naive Bayes) ──────────────────────────────────────
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_cv_mean, nb_cv_std = cv_accuracy(MultinomialNB(), X, y)

    print("=" * 60)
    print("  BAYESIAN MODEL (Naive Bayes)")
    print("=" * 60)
    print(classification_report(y_test, nb_preds, target_names=label_names))
    print(f"  5-Fold CV Accuracy: {nb_cv_mean:.3f} ± {nb_cv_std:.3f}\n")
    results_summary["NaiveBayes"] = {"accuracy": accuracy_score(y_test, nb_preds),
                                      "cv_mean": nb_cv_mean, "cv_std": nb_cv_std}

    # ── (D) Hybrid Model ─────────────────────────────────────────────────────
    # Architecture: symbolic output is added as an informative feature (prior knowledge).
    # MLP and Naive Bayes are trained on this enriched feature space.
    # Prediction: soft majority vote — when both agree, use consensus;
    # when they disagree, defer to MLP (higher standalone CV accuracy).
    # This mirrors dual-process cognition: fast heuristic (symbolic) scaffolds
    # slow deliberate learning (connectionist + probabilistic).

    sym_all  = np.array([symbolic_fn(row) for row in X])
    X_hybrid = np.hstack([X, sym_all.reshape(-1, 1)])

    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X_hybrid, y, test_size=0.25, random_state=16, stratify=y
    )

    mlp_h = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=16)
    nb_h  = MultinomialNB()
    mlp_h.fit(Xh_train, yh_train)
    nb_h.fit(Xh_train, yh_train)

    mlp_h_preds = mlp_h.predict(Xh_test)
    nb_h_preds  = nb_h.predict(Xh_test)

    # Soft majority: MLP fallback on disagreement
    hybrid_preds = np.where(mlp_h_preds == nb_h_preds, mlp_h_preds, mlp_h_preds)

    hybrid_cv_mean, hybrid_cv_std = cv_accuracy(
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=16),
        X_hybrid, y
    )

    print("=" * 60)
    print("  HYBRID MODEL (Symbolic Prior + MLP + Naive Bayes)")
    print("=" * 60)
    print(classification_report(yh_test, hybrid_preds, target_names=label_names))
    print(f"  5-Fold CV Accuracy (MLP-Hybrid): {hybrid_cv_mean:.3f} ± {hybrid_cv_std:.3f}\n")
    results_summary["Hybrid"] = {"accuracy": accuracy_score(yh_test, hybrid_preds),
                                  "cv_mean": hybrid_cv_mean, "cv_std": hybrid_cv_std}

    # ── Summary Table ───────────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
    print("-" * 52)
    for name, r in results_summary.items():
        cv_m = f"{r['cv_mean']:.3f}" if r['cv_mean'] else "  N/A "
        cv_s = f"{r['cv_std']:.3f}"  if r['cv_std']  else "  N/A "
        print(f"{name:<20} {r['accuracy']:>10.3f} {cv_m:>10} {cv_s:>10}")
    print()

    # ── Confusion Matrices ──────────────────────────────────────
    print("Confusion Matrices (rows=True, cols=Predicted):")
    for name, preds, yt in [("Symbolic",   sym_preds,    y_test),
                             ("MLP",        mlp_preds,    y_test),
                             ("Naive Bayes",nb_preds,     y_test),
                             ("Hybrid",     hybrid_preds, yh_test)]:
        cm = confusion_matrix(yt, preds)
        print(f"\n  {name}")
        print(f"  {'':<12} Pred Living  Pred Object")
        print(f"  {'True Living':<12}   {cm[0,0]:>5}        {cm[0,1]:>5}")
        print(f"  {'True Object':<12}   {cm[1,0]:>5}        {cm[1,1]:>5}")


if __name__ == "__main__":
    main()
