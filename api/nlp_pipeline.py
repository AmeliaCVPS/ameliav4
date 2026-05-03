"""
API/nlp_pipeline.py - NLP clinico da AMELIA
===========================================

Pipeline hibrido:
  texto livre -> NLP/regex clinico -> features estruturadas -> Random Forest

Pipeline alternativo:
  texto livre -> BERT fine-tuned -> classificacao direta

O modulo nao exige transformers em runtime: BERT e carregado de forma
preguicosa apenas quando houver um modelo treinado em disco.
"""

from __future__ import annotations

import os
import re
import math
import unicodedata
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Any

from .ml_model import SymptomsInput, predict_risk, _make_password


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BERT_DIR = os.path.join(BASE_DIR, "models", "bert_triage")

COLOR_TO_PT = {
    "red": "vermelho",
    "orange": "laranja",
    "yellow": "amarelo",
    "green": "verde",
}
PT_TO_COLOR = {v: k for k, v in COLOR_TO_PT.items()}
ID_TO_COLOR = {0: "green", 1: "yellow", 2: "orange", 3: "red"}
COLOR_TO_ID = {v: k for k, v in ID_TO_COLOR.items()}
RISK_META = {
    "red": (1, "Imediato", "Emergência — atendimento imediato por médico(a) ou enfermeiro(a)."),
    "orange": (2, "<= 10 min", "Urgente — será atendido em breve."),
    "yellow": (3, "<= 30 min", "Moderado — aguarde na fila prioritária."),
    "green": (4, "<= 2 horas", "Pouco urgente — aguarde na fila regular."),
}


COMMON_TYPOS = {
    "to": "estou",
    "tô": "estou",
    "tou": "estou",
    "ta": "esta",
    "tá": "esta",
    "vc": "voce",
    "q": "que",
    "pq": "porque",
    "peit": "peito",
    "cabeca": "cabeca",
    "dor d cabeca": "dor de cabeca",
    "falta d ar": "falta de ar",
    "falta ar": "falta de ar",
    "vomitu": "vomito",
    "vomitandoo": "vomitando",
    "febriu": "febril",
    "coraçao": "coracao",
    "coração": "coracao",
    "consiencia": "consciencia",
}

SYMPTOM_LEXICON: dict[str, list[str]] = {
    "fever": [
        "febre", "febril", "temperatura alta", "corpo quente", "calafrio",
        "tremedeira de febre", "ardendo em febre", "febrao",
    ],
    "shortness_of_breath": [
        "falta de ar", "sem ar", "dificuldade para respirar", "nao consigo respirar",
        "respiracao dificil", "ofegante", "ofego", "sufocando", "dispneia",
        "chiado no peito", "crise de asma", "asma atacada",
    ],
    "chest_pain": [
        "dor no peito", "aperto no peito", "pressao no peito", "peso no peito",
        "dor toracica", "dor precordial", "angina", "infarto",
        "coracao doendo", "dor irradiando para o braco", "dor no braço esquerdo",
    ],
    "altered_consciousness": [
        "desmaio", "desmaiei", "desmaiou", "apagao", "apagou",
        "perdi a consciencia", "perdeu a consciencia", "confuso", "desorientado",
        "convulsao", "convulsionando", "nao responde", "sonolento demais",
        "perdeu o sentido", "sincope",
    ],
    "bleeding": [
        "sangue", "sangrando", "sangramento", "hemorragia", "corte profundo",
        "vomitei sangue", "tosse com sangue", "sangue nas fezes", "urina com sangue",
        "nariz sangrando", "ferida aberta",
    ],
    "vomiting": [
        "vomito", "vomitando", "vomitei", "nausea", "enjoo forte",
        "ansia", "nao para de vomitar", "vomitos repetidos",
    ],
    "severe_headache": [
        "dor de cabeca", "cefaleia", "enxaqueca", "cabeca latejando",
        "pior dor de cabeca", "cabeca explodindo", "pressao na cabeca",
        "dor intensa na cabeca",
    ],
    "allergic_reaction": [
        "alergia", "alergico", "alergica", "reacao alergica", "urticaria",
        "coceira no corpo", "inchaco na garganta", "rosto inchado", "anafilaxia",
        "choque alergico", "picada de inseto",
    ],
    "trauma": [
        "acidente", "queda", "cai", "caiu", "bati", "bateu", "colisao",
        "atropelado", "pancada", "machucado", "fratura", "torcao",
        "entorse", "queimadura", "ferimento", "trauma",
    ],
}

FEATURE_KEYS = [
    "fever",
    "shortness_of_breath",
    "chest_pain",
    "altered_consciousness",
    "bleeding",
    "vomiting",
    "severe_headache",
    "allergic_reaction",
    "trauma",
]


@dataclass
class ExtractedFeatures:
    text_normalized: str
    symptoms: list[str]
    pain_level: int
    duration_hours: float
    entities: list[dict[str, Any]] = field(default_factory=list)
    fever: bool = False
    shortness_of_breath: bool = False
    chest_pain: bool = False
    altered_consciousness: bool = False
    bleeding: bool = False
    vomiting: bool = False
    severe_headache: bool = False
    allergic_reaction: bool = False
    trauma: bool = False
    heart_rate: int | None = None
    temperature: float | None = None
    oxygen_saturation: float | None = None


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_text(text: str) -> str:
    """Normaliza texto brasileiro informal preservando numeros clinicos."""
    text = str(text or "").lower()
    text = re.sub("[\U00010000-\U0010ffff]", " ", text)
    text = text.replace("\ufe0f", " ").replace("\u200d", " ")
    text = strip_accents(text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z0-9.,/%+\-: ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for wrong, right in COMMON_TYPOS.items():
        wrong_n = strip_accents(wrong.lower())
        right_n = strip_accents(right.lower())
        text = re.sub(rf"\b{re.escape(wrong_n)}\b", right_n, text)
    return text


def _has_phrase(text: str, phrases: list[str]) -> bool:
    return any(strip_accents(p.lower()) in text for p in phrases)


def extract_symptoms(text_norm: str) -> tuple[dict[str, bool], list[str]]:
    flags: dict[str, bool] = {}
    found: list[str] = []
    for key, phrases in SYMPTOM_LEXICON.items():
        hit = _has_phrase(text_norm, phrases)
        flags[key] = hit
        if hit:
            found.append(key)
    return flags, found


def extract_symptom_entities(text_norm: str) -> list[dict[str, Any]]:
    """NER leve baseado em dicionario clinico, com spans no texto normalizado."""
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for key, phrases in SYMPTOM_LEXICON.items():
        for phrase in phrases:
            phrase_norm = strip_accents(phrase.lower())
            for match in re.finditer(re.escape(phrase_norm), text_norm):
                marker = (key, match.start(), match.end())
                if marker in seen:
                    continue
                seen.add(marker)
                entities.append({
                    "label": "SINTOMA",
                    "feature": key,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                })
    return sorted(entities, key=lambda item: (item["start"], item["end"]))


def extract_pain_level(text_norm: str) -> int:
    patterns = [
        r"\bdor\s*(?:nota|nivel|grau)?\s*(10|[0-9])\b",
        r"\b(10|[0-9])\s*/\s*10\b",
        r"\b(10|[0-9])\s+de\s+10\b",
        r"\bescala\s*(10|[0-9])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_norm)
        if match:
            return max(0, min(10, int(match.group(1))))

    if re.search(r"insuportavel|pior dor|horrivel|terrivel|agonia|excruciante", text_norm):
        return 10
    if re.search(r"muito forte|fortissima|intensa|severa|forte demais", text_norm):
        return 8
    if re.search(r"\bforte\b|bastante dor|consideravel", text_norm):
        return 7
    if re.search(r"moderad[ao]|media|razoavel|suportavel", text_norm):
        return 5
    if re.search(r"leve|fraquinha|fraca|pouca|levinha", text_norm):
        return 3
    if re.search(r"sem dor|nao doi|nenhuma dor|quase nada", text_norm):
        return 0
    return 5


def extract_duration_hours(text_norm: str) -> float:
    units = [
        (r"(\d+(?:[.,]\d+)?)\s*(?:min|mins|minuto|minutos)\b", 1 / 60),
        (r"(\d+(?:[.,]\d+)?)\s*(?:h|hr|hrs|hora|horas)\b", 1),
        (r"(\d+(?:[.,]\d+)?)\s*(?:dia|dias)\b", 24),
        (r"(\d+(?:[.,]\d+)?)\s*(?:semana|semanas)\b", 168),
    ]
    for pattern, factor in units:
        match = re.search(pattern, text_norm)
        if match:
            value = float(match.group(1).replace(",", "."))
            return max(0.25, value * factor)

    if re.search(r"agora|ha pouco|faz pouco|de repente|subito|minutos", text_norm):
        return 0.25
    if re.search(r"hoje cedo|esta manha|esta tarde", text_norm):
        return 4
    if re.search(r"ontem|noite passada", text_norm):
        return 24
    if re.search(r"faz tempo|cronico|semanas", text_norm):
        return 336
    return 24


def extract_vitals(text_norm: str) -> dict[str, Any]:
    vitals: dict[str, Any] = {
        "heart_rate": None,
        "temperature": None,
        "oxygen_saturation": None,
    }

    temp = re.search(r"\b(3[5-9](?:[.,]\d)?|4[0-2](?:[.,]\d)?)\s*(?:c|graus|ºc)?\b", text_norm)
    if temp:
        vitals["temperature"] = float(temp.group(1).replace(",", "."))

    spo2 = re.search(r"\b(?:spo2|saturacao|sat)\s*(?:de|=|:)?\s*(\d{2,3})\s*%?", text_norm)
    if spo2:
        value = int(spo2.group(1))
        if 50 <= value <= 100:
            vitals["oxygen_saturation"] = value

    hr = re.search(r"\b(?:fc|batimentos|frequencia cardiaca|bpm)\s*(?:de|=|:)?\s*(\d{2,3})\b", text_norm)
    if hr:
        value = int(hr.group(1))
        if 30 <= value <= 240:
            vitals["heart_rate"] = value

    return vitals


def extract_features_from_text(text: str) -> ExtractedFeatures:
    text_norm = normalize_text(text)
    flags, symptoms = extract_symptoms(text_norm)
    entities = extract_symptom_entities(text_norm)
    vitals = extract_vitals(text_norm)

    return ExtractedFeatures(
        text_normalized=text_norm,
        symptoms=symptoms,
        entities=entities,
        pain_level=extract_pain_level(text_norm),
        duration_hours=extract_duration_hours(text_norm),
        heart_rate=vitals["heart_rate"],
        temperature=vitals["temperature"],
        oxygen_saturation=vitals["oxygen_saturation"],
        **flags,
    )


def to_symptoms_input(
    text: str,
    *,
    cpf: str = "00000000000",
    age: int = 30,
    sex: str = "M",
    temperature: float | None = None,
) -> tuple[SymptomsInput, ExtractedFeatures]:
    extracted = extract_features_from_text(text)
    temp = temperature if temperature is not None else extracted.temperature
    if temp is not None and temp >= 37.8:
        extracted.fever = True
        if "fever" not in extracted.symptoms:
            extracted.symptoms.append("fever")

    if extracted.oxygen_saturation is not None and extracted.oxygen_saturation < 92:
        extracted.shortness_of_breath = True
        if "shortness_of_breath" not in extracted.symptoms:
            extracted.symptoms.append("shortness_of_breath")

    sex_norm = str(sex or "M").upper()
    data = SymptomsInput(
        cpf=cpf,
        age=max(0, min(120, int(age or 30))),
        sex=sex_norm if sex_norm in {"M", "F"} else "M",
        description=str(text or "")[:3000],
        pain_level=extracted.pain_level,
        fever=extracted.fever,
        shortness_of_breath=extracted.shortness_of_breath,
        chest_pain=extracted.chest_pain,
        altered_consciousness=extracted.altered_consciousness,
        bleeding=extracted.bleeding,
        duration_hours=extracted.duration_hours,
        vomiting=extracted.vomiting,
        severe_headache=extracted.severe_headache,
        allergic_reaction=extracted.allergic_reaction,
        trauma=extracted.trauma,
        temperature=temp,
        heart_rate=extracted.heart_rate,
        oxygen_saturation=extracted.oxygen_saturation,
    )
    return data, extracted


def classify_hybrid_text(
    text: str,
    *,
    cpf: str = "00000000000",
    age: int = 30,
    sex: str = "M",
    temperature: float | None = None,
) -> dict[str, Any]:
    """Texto -> features -> Random Forest existente."""
    symptoms_input, extracted = to_symptoms_input(
        text, cpf=cpf, age=age, sex=sex, temperature=temperature
    )
    cls = predict_risk(symptoms_input)
    return {
        "classificacao": COLOR_TO_PT.get(cls["color"], cls["color"]),
        "classification": cls,
        "password": cls.get("password", ""),
        "sintomas": extracted.symptoms,
        "features": symptoms_input.model_dump() if hasattr(symptoms_input, "model_dump") else symptoms_input.dict(),
        "nlp": asdict(extracted),
        "confianca": cls.get("confidence"),
        "metodo": "nlp_features_random_forest",
    }


@lru_cache(maxsize=2)
def _load_bert(model_dir: str = DEFAULT_BERT_DIR):
    if not os.path.isdir(model_dir):
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model, torch


def classify_bert_text(text: str, model_dir: str = DEFAULT_BERT_DIR) -> dict[str, Any] | None:
    loaded = _load_bert(model_dir)
    if loaded is None:
        return None
    tokenizer, model, torch = loaded
    clean = normalize_text(text)
    encoded = tokenizer(
        clean,
        truncation=True,
        max_length=192,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
    color = ID_TO_COLOR.get(label_id, "green")
    confidence = float(probs[label_id].item())
    return {
        "classificacao": COLOR_TO_PT[color],
        "classification": {
            "color": color,
            "priority": RISK_META[color][0],
            "wait_time": RISK_META[color][1],
            "explanation": RISK_META[color][2],
            "confidence": round(confidence, 4),
            "password": "",
        },
        "confianca": round(confidence, 4),
        "metodo": "bert_direct",
    }


def triage_text(
    text: str,
    *,
    cpf: str = "00000000000",
    age: int = 30,
    sex: str = "M",
    temperature: float | None = None,
    use_bert: bool = True,
    bert_model_dir: str = DEFAULT_BERT_DIR,
) -> dict[str, Any]:
    """
    Tenta BERT direto quando disponivel, mas sempre retorna features do
    pipeline hibrido para auditoria clinica e compatibilidade com o RF.
    """
    hybrid = classify_hybrid_text(
        text, cpf=cpf, age=age, sex=sex, temperature=temperature
    )
    bert = classify_bert_text(text, bert_model_dir) if use_bert else None
    if not bert:
        return hybrid

    # Se BERT estiver confiante, usa sua classe direta, preservando as features.
    if bert["confianca"] >= 0.70:
        color = PT_TO_COLOR[bert["classificacao"]]
        priority, wait_time, explanation = RISK_META[color]
        old_color = hybrid["classification"].get("color")
        password = hybrid.get("password", "") if color == old_color else ("" if color == "red" else _make_password(color))
        hybrid["bert"] = bert
        hybrid["classificacao"] = bert["classificacao"]
        hybrid["classification"]["color"] = color
        hybrid["classification"]["priority"] = priority
        hybrid["classification"]["wait_time"] = wait_time
        hybrid["classification"]["explanation"] = explanation
        hybrid["classification"]["confidence"] = bert["confianca"]
        hybrid["classification"]["password"] = password
        hybrid["password"] = password
        hybrid["confianca"] = bert["confianca"]
        hybrid["metodo"] = "bert_direct_with_feature_audit"
    else:
        hybrid["bert"] = bert
        hybrid["metodo"] = "hybrid_rf_low_bert_confidence"
    return hybrid


def symptoms_to_human(symptoms: list[str]) -> list[str]:
    labels = {
        "fever": "febre",
        "shortness_of_breath": "falta de ar",
        "chest_pain": "dor no peito",
        "altered_consciousness": "alteracao de consciencia",
        "bleeding": "sangramento",
        "vomiting": "vomito/nausea",
        "severe_headache": "cefaleia intensa",
        "allergic_reaction": "reacao alergica",
        "trauma": "trauma/acidente",
    }
    return [labels.get(s, s) for s in symptoms]
