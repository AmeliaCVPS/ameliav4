"""
train_model.py — Script de Treinamento do Modelo A.M.E.L.I.A v3.0
==================================================================
Execute UMA VEZ:  python train_model.py
Saída: modelo_amelia.joblib + encoder_amelia.joblib

12 Features do Protocolo de Manchester ampliado:
  pain_level, fever, shortness_of_breath, chest_pain,
  altered_consciousness, bleeding, duration_hours, age,
  vomiting, severe_headache, allergic_reaction, trauma
"""

import os, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE, "dataset_treino.csv")
MODEL_PATH   = os.path.join(BASE, "modelo_amelia.joblib")
ENCODER_PATH = os.path.join(BASE, "encoder_amelia.joblib")

FEATURES = [
    "pain_level","fever","shortness_of_breath","chest_pain",
    "altered_consciousness","bleeding","duration_hours","age",
    "vomiting","severe_headache","allergic_reaction","trauma",
]


def _manchester(pain, fever, sob, chest, cons, bleed, dur, age,
                vomit, headache, allergy, trauma):
    """Regras clínicas do Protocolo de Manchester."""
    # EMERGÊNCIA
    if (cons
        or (sob and chest) or (sob and pain >= 8)
        or pain >= 9
        or (chest and pain >= 8)
        or (bleed and pain >= 7)
        or (allergy and (sob or cons))
        or (trauma and (cons or (bleed and pain >= 7)))
        or (age < 3 and fever and pain >= 6)
        or (age < 1 and (fever or sob))
        or (age > 80 and chest and pain >= 8)
        or (headache and (cons or pain >= 9))):
        return "red"
    # MUITO URGENTE
    if (pain >= 7 or sob or chest
        or (fever and (age >= 65 or age <= 5 or pain >= 6))
        or bleed or allergy
        or (trauma and pain >= 6)
        or (headache and (pain >= 7 or fever))
        or (vomit and (dur <= 4 or pain >= 6))
        or (pain >= 6 and dur <= 2)
        or (age < 2 and fever)
        or (age > 70 and pain >= 6)):
        return "orange"
    # URGENTE
    if (pain >= 4
        or (fever and (pain >= 2 or dur <= 24))
        or (vomit and pain >= 3)
        or (headache and pain >= 4)
        or (trauma and pain >= 3)
        or dur <= 8
        or (age < 5 and (fever or pain >= 3))
        or (age > 65 and pain >= 3)):
        return "yellow"
    return "green"


def _generate_synthetic(n_per_class=300, seed=42):
    np.random.seed(seed)
    rows, counts = [], {"red":0,"orange":0,"yellow":0,"green":0}
    pain_probs = [.04,.05,.07,.09,.10,.12,.12,.13,.11,.09,.08]
    dur_vals   = [.25,.5,1,2,3,4,6,8,12,24,48,72,120,168]
    dur_probs  = [.03,.05,.08,.08,.08,.07,.08,.07,.08,.10,.10,.07,.06,.05]
    max_try    = n_per_class * 80

    for _ in range(max_try):
        if min(counts.values()) >= n_per_class:
            break
        pain    = np.random.choice(range(11), p=pain_probs)
        fever   = np.random.randint(0,2)
        sob     = np.random.randint(0,2)
        chest   = int(np.random.random()<.15)
        cons    = int(np.random.random()<.08)
        bleed   = int(np.random.random()<.12)
        dur     = np.random.choice(dur_vals, p=dur_probs)
        age     = max(0, min(95, int(np.random.gamma(5,8))))
        vomit   = int(np.random.random()<.20)
        head    = int(np.random.random()<.18)
        allergy = int(np.random.random()<.10)
        trauma  = int(np.random.random()<.12)

        color = _manchester(pain,fever,sob,chest,cons,bleed,dur,age,
                            vomit,head,allergy,trauma)
        if counts[color] < n_per_class:
            counts[color] += 1
            rows.append([pain,fever,sob,chest,cons,bleed,dur,age,
                         vomit,head,allergy,trauma,color])

    df = pd.DataFrame(rows, columns=FEATURES+["risk_color"])
    print(f"  Sintéticos: {len(df)} | {counts}")
    return df


def train_and_save():
    print("="*62)
    print("  A.M.E.L.I.A v3.0 — Treinamento (12 features)")
    print("="*62)

    df_m = pd.read_csv(DATASET_PATH)
    print(f"\n📂 Manual: {len(df_m)} amostras")
    df_s = _generate_synthetic(300, 42)
    df   = pd.concat([df_m, df_s], ignore_index=True).sample(
               frac=1, random_state=42).reset_index(drop=True)
    print(f"📊 Total : {len(df)} amostras\n")

    emojis = {"red":"🔴","orange":"🟠","yellow":"🟡","green":"🟢"}
    for c,n in df["risk_color"].value_counts().items():
        print(f"  {emojis.get(c,'')} {c:<8} {'█'*(n//6)} {n}")

    X   = df[FEATURES].values.astype(float)
    enc = LabelEncoder()
    y   = enc.fit_transform(df["risk_color"].values)
    print(f"\nClasses: {list(enc.classes_)}")

    X_tr,X_te,y_tr,y_te = train_test_split(
        X,y,test_size=.20,random_state=42,stratify=y)
    print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}\n")

    model = RandomForestClassifier(
        n_estimators=250, max_depth=14,
        min_samples_split=3, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1)

    print("🌲 Treinando (250 árvores, 12 features)…")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    print("\n📊 Relatório:")
    print(classification_report(
        enc.inverse_transform(y_te),
        enc.inverse_transform(y_pred)))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"🎯 CV 10-fold: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    print("\n🔍 Feature Importance:")
    for feat,imp in sorted(zip(FEATURES,model.feature_importances_),
                           key=lambda x:x[1],reverse=True):
        print(f"  {feat:<28} {'█'*int(imp*70)} {imp:.4f}")

    joblib.dump(model, MODEL_PATH,   compress=3)
    joblib.dump(enc,   ENCODER_PATH, compress=3)
    print(f"\n💾 {MODEL_PATH}")
    print(f"💾 {ENCODER_PATH}")
    status = "✅ META ATINGIDA" if cv.mean()>=.96 else "⚠️  abaixo de 96%"
    print(f"\n{status} — {cv.mean()*100:.2f}%")
    print("="*62)
    return model, enc

if __name__ == "__main__":
    train_and_save()
