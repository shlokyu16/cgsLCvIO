import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

csv_path = "lcvio.csv"
df = pd.read_csv(csv_path)

# Normalize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Encode all categorical columns
encoders = {}
for col in ['shape', 'skin/material_type', 'mobility', 'vocality', 'surface_hardness', 'label']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare features and labels
X = df[['shape_enc', 'skin/material_type_enc', 'mobility_enc', 'vocality_enc', 'surface_hardness_enc']].values
y = df['label_enc'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Symbolic Model (Rule-Based)
def symbolic_classifier(row):
    shape = encoders['shape'].inverse_transform([row[0]])[0]
    material = encoders['skin/material_type'].inverse_transform([row[1]])[0]
    mobility = encoders['mobility'].inverse_transform([row[2]])[0]
    vocality = encoders['vocality'].inverse_transform([row[3]])[0]
    hardness = encoders['surface_hardness'].inverse_transform([row[4]])[0]

    if material in ['skin', 'fur', 'grass']:
        return 1  # Living
    elif mobility == 'Yes' and vocality == 'Yes':
        return 1
    elif shape == 'Organic' and material == 'wood':
        return 1
    else:
        return 0  # Object

symbolic_preds = [symbolic_classifier(row) for row in X_test]

# Connectionist Model (Neural Net)
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1605, random_state=16)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)

# Bayesian Model (Naive Bayes)
nb = MultinomialNB()
nb.fit(X_train, y_train)
bayes_preds = nb.predict(X_test)
# Is basically P(H|E) = P(E|H) * P(H)

# Compare Performance
print("\n .... Symbolic Model .... ")
print(classification_report(y_test, symbolic_preds, target_names=encoders['label'].classes_))

print("\n .... Connectionist Model (Neural Network) .... ")
print(classification_report(y_test, mlp_preds, target_names=encoders['label'].classes_))

print("\n .... Bayesian Model .... ")
print(classification_report(y_test, bayes_preds, target_names=encoders['label'].classes_))

# Hybrid Model, Train on NN and Bayes, and using Symbolic Model as a learning curve.
df['symbolic_pred'] = df[['shape_enc', 'skin/material_type_enc', 'mobility_enc', 'vocality_enc', 'surface_hardness_enc']].apply(symbolic_classifier, axis=1)
X = df[['shape_enc', 'skin/material_type_enc', 'mobility_enc', 'vocality_enc', 'surface_hardness_enc', 'symbolic_pred']].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1605, random_state=16)
mlp.fit(X_train, y_train)
nb = MultinomialNB()
nb.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
bayes_preds = nb.predict(X_test)
hybrid_preds = np.where(mlp_preds == bayes_preds, bayes_preds, mlp_preds)

print("\n--- Hybrid Model (Majority + MLP Fallback) ---")
print(classification_report(y_test, hybrid_preds, target_names=encoders['label'].classes_))
