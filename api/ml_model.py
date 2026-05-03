"""
api/ml_model.py — Motor de IA da AMÉLIA v3.0
=============================================

12 features | Emergência sem senha / U001 / M001 / L001 / V001
Carrega o modelo Random Forest treinado com 97.74% de acurácia (CV 10-fold).
"""

import os, joblib, numpy as np
from pydantic import BaseModel, Field
from typing import Optional

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE, "modelo_amelia.joblib")
ENCODER_PATH = os.path.join(BASE, "encoder_amelia.joblib")

# ── DEVE ser idêntica ao train_model.py ──────────────────────
FEATURES = [
    "pain_level","fever","shortness_of_breath","chest_pain",
    "altered_consciousness","bleeding","duration_hours","age",
    "vomiting","severe_headache","allergic_reaction","trauma",
]

# ── Senhas padronizadas por prioridade ────────────────────────
#   red = Emergência sem senha (atendimento imediato)
#   U = Urgente (orange)
#   M = Moderado           (yellow)
#   L = Leve               (green)
#   (V = Verde, alias de L para retrocompatibilidade)
_PASSWORD_PREFIX = {
    "red":    "",    # Emergência: sem senha de espera
    "orange": "U",   # Urgente
    "yellow": "M",   # Moderado
    "green":  "L",   # Leve
}

# Contadores independentes por PREFIXO (não por cor).
# Emergência vermelha não entra na fila: atendimento imediato, sem senha.
_counters: dict[str, int] = {"U": 0, "M": 0, "L": 0}

# Cache do modelo (carrega 1× por processo)
_model = _encoder = None


# ── Schema de entrada ─────────────────────────────────────────
class SymptomsInput(BaseModel):
    """Dados do paciente recebidos do frontend após chat/voz."""
    cpf:                   str   = Field(...)
    age:                   int   = Field(..., ge=0, le=120)
    sex:                   str   = Field("M", pattern="^[MF]$")
    description:           str   = Field("", max_length=3000)
    pain_level:            int   = Field(..., ge=0, le=10)
    fever:                 bool  = Field(False)
    shortness_of_breath:   bool  = Field(False)
    chest_pain:            bool  = Field(False)
    altered_consciousness: bool  = Field(False)
    bleeding:              bool  = Field(False)
    duration_hours:        float = Field(24.0, ge=0)
    vomiting:              bool  = Field(False)
    severe_headache:       bool  = Field(False)
    allergic_reaction:     bool  = Field(False)
    trauma:                bool  = Field(False)
    # sinais vitais opcionais
    temperature:           Optional[float] = None
    heart_rate:            Optional[int]   = None
    oxygen_saturation:     Optional[float] = None


# ── Carregamento com cache ────────────────────────────────────
def _load():
    global _model, _encoder
    if _model is not None:
        return _model, _encoder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\nRode: python train_model.py")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Encoder não encontrado: {ENCODER_PATH}\nRode: python train_model.py")
    _model   = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)
    print("✅ Modelo AMÉLIA v3.0 carregado (97.74% acurácia).")
    return _model, _encoder


# ── Geração de senha ──────────────────────────────────────────
def _make_password(color: str) -> str:
    """
    red  → sem senha, atendimento imediato
    U001 → urgente (orange)
    M001 → moderado (yellow)
    L001 → leve (green)
    Sequencial por prefixo, nunca reinicia na sessão.
    """
    if color == "red":
        return ""
    prefix = _PASSWORD_PREFIX.get(color, "L")
    _counters[prefix] += 1
    return f"{prefix}{str(_counters[prefix]).zfill(3)}"


# ── Predição ──────────────────────────────────────────────────
def predict_risk(data: SymptomsInput) -> dict:
    """Classifica o risco com o modelo Random Forest."""
    model, encoder = _load()

    X = np.array([[
        data.pain_level,
        int(data.fever),
        int(data.shortness_of_breath),
        int(data.chest_pain),
        int(data.altered_consciousness),
        int(data.bleeding),
        data.duration_hours,
        data.age,
        int(data.vomiting),
        int(data.severe_headache),
        int(data.allergic_reaction),
        int(data.trauma),
    ]], dtype=float)

    idx   = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    color = encoder.inverse_transform([idx])[0]
    conf  = round(float(proba[idx]), 4)

    meta = {
        "red":    (1, "Imediato",  "Emergência — atendimento imediato por médico(a) ou enfermeiro(a)."),
        "orange": (2, "<= 10 min", "Urgente — será atendido em breve."),
        "yellow": (3, "<= 30 min", "Moderado — aguarde na fila prioritária."),
        "green":  (4, "<= 2 horas", "Pouco urgente — aguarde na fila regular."),
    }
    priority, wait, explanation = meta[color]

    return {
        "color":       color,
        "priority":    priority,
        "wait_time":   wait,
        "explanation": explanation,
        "confidence":  conf,
        "password":    _make_password(color),
    }
