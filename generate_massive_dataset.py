"""
generate_massive_dataset.py - Dataset massivo sintetico AMELIA
==============================================================

Gera duas saidas em streaming:
  1) CSV estruturado com features clinicas + colunas em portugues
  2) JSONL textual com frases realistas de pacientes brasileiros

Exemplos:
  python generate_massive_dataset.py --n 10000 --out-dir data/generated
  python generate_massive_dataset.py --n 2000000 --chunk-size 50000 --gzip

Para escala de GB, use --gzip e um --chunk-size alto. O script nao mantem o
dataset inteiro em memoria.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
from dataclasses import dataclass
from typing import TextIO


FEATURES = [
    "pain_level", "fever", "shortness_of_breath", "chest_pain",
    "altered_consciousness", "bleeding", "duration_hours", "age",
    "vomiting", "severe_headache", "allergic_reaction", "trauma",
]

PT_CLASS = {
    "red": "vermelho",
    "orange": "laranja",
    "yellow": "amarelo",
    "green": "verde",
}

SYMPTOMS = {
    "fever": {
        "pt": "febre",
        "phrases": ["febre", "corpo quente", "calafrios", "temperatura alta", "febrao"],
    },
    "shortness_of_breath": {
        "pt": "falta de ar",
        "phrases": ["falta de ar", "sem ar", "ofegante", "dificuldade pra respirar", "sufocando"],
    },
    "chest_pain": {
        "pt": "dor no peito",
        "phrases": ["dor no peito", "aperto no peito", "pressao no peito", "peso no peito", "dor no coracao"],
    },
    "altered_consciousness": {
        "pt": "alteracao de consciencia",
        "phrases": ["desmaiei", "apagou", "confuso", "convulsao", "perdeu a consciencia"],
    },
    "bleeding": {
        "pt": "sangramento",
        "phrases": ["sangrando", "sangramento", "muito sangue", "corte profundo", "hemorragia"],
    },
    "vomiting": {
        "pt": "vomito",
        "phrases": ["vomitando", "vomitei", "nausea forte", "enjoo forte", "ansia de vomito"],
    },
    "severe_headache": {
        "pt": "dor de cabeca intensa",
        "phrases": ["dor de cabeca", "enxaqueca", "cabeca latejando", "pressao na cabeca", "cefaleia"],
    },
    "allergic_reaction": {
        "pt": "reacao alergica",
        "phrases": ["alergia", "coceira no corpo", "rosto inchado", "inchaco na garganta", "urticaria"],
    },
    "trauma": {
        "pt": "trauma",
        "phrases": ["cai", "queda", "acidente", "bati a cabeca", "fratura", "machucado"],
    },
}

OPENERS = [
    "Estou com {symptoms}",
    "To com {symptoms}",
    "Eu estou sentindo {symptoms}",
    "Meu problema e {symptoms}",
    "Vim porque estou com {symptoms}",
    "Preciso de ajuda, estou com {symptoms}",
    "Doutor, estou com {symptoms}",
]

INTENSITY_TEXT = {
    0: ["sem dor"],
    1: ["quase sem dor", "bem leve"],
    2: ["leve", "fraquinha"],
    3: ["leve", "incomodando um pouco"],
    4: ["moderada", "chata"],
    5: ["moderada", "suportavel"],
    6: ["forte", "bastante dor"],
    7: ["forte", "muito incomoda"],
    8: ["muito forte", "intensa"],
    9: ["quase insuportavel", "muito intensa"],
    10: ["insuportavel", "a pior dor que ja senti"],
}

TYPO_MAP = {
    "estou": ["to", "tou", "tô"],
    "dor": ["dooor", "dr"],
    "peito": ["peit", "peitoo"],
    "cabeca": ["cabeça", "cabeca"],
    "respirar": ["respira", "respirar"],
    "vomitando": ["vomitandoo", "vomitano"],
    "muito": ["mt", "mto"],
    "ha": ["faz", "há"],
}


@dataclass
class SyntheticCase:
    text: str
    risk_color: str
    symptoms: list[str]
    pain_level: int
    fever: int
    shortness_of_breath: int
    chest_pain: int
    altered_consciousness: int
    bleeding: int
    duration_hours: float
    age: int
    vomiting: int
    severe_headache: int
    allergic_reaction: int
    trauma: int
    heart_rate: int
    temperature: float
    spo2: int
    comorbidity: str


def manchester_rule(case: dict) -> str:
    pain = case["pain_level"]
    fever = case["fever"]
    sob = case["shortness_of_breath"]
    chest = case["chest_pain"]
    cons = case["altered_consciousness"]
    bleed = case["bleeding"]
    dur = case["duration_hours"]
    age = case["age"]
    vomit = case["vomiting"]
    headache = case["severe_headache"]
    allergy = case["allergic_reaction"]
    trauma = case["trauma"]

    if (
        cons or (sob and chest) or (sob and pain >= 8) or pain >= 9
        or (chest and pain >= 8) or (bleed and pain >= 7)
        or (allergy and (sob or cons)) or (trauma and (cons or (bleed and pain >= 7)))
        or (age < 1 and (fever or sob)) or (age > 80 and chest and pain >= 8)
        or (headache and (cons or pain >= 9))
    ):
        return "red"
    if (
        pain >= 7 or sob or chest or bleed or allergy
        or (fever and (age >= 65 or age <= 5 or pain >= 6))
        or (trauma and pain >= 6) or (headache and (pain >= 7 or fever))
        or (vomit and (dur <= 4 or pain >= 6)) or (pain >= 6 and dur <= 2)
    ):
        return "orange"
    if (
        pain >= 4 or (fever and (pain >= 2 or dur <= 24))
        or (vomit and pain >= 3) or (headache and pain >= 4)
        or (trauma and pain >= 3) or dur <= 8
        or (age < 5 and (fever or pain >= 3)) or (age > 65 and pain >= 3)
    ):
        return "yellow"
    return "green"


def maybe_typo(text: str, rng: random.Random, rate: float) -> str:
    if rng.random() > rate:
        return text
    words = text.split()
    for i, word in enumerate(words):
        key = word.lower().strip(".,")
        if key in TYPO_MAP and rng.random() < 0.35:
            words[i] = rng.choice(TYPO_MAP[key])
    return " ".join(words)


def duration_phrase(hours: float, rng: random.Random) -> str:
    if hours < 1:
        minutes = max(5, int(hours * 60))
        return rng.choice([f"ha {minutes} minutos", f"faz {minutes} min", f"desde uns {minutes} minutos"])
    if hours < 24:
        h = int(hours)
        return rng.choice([f"ha {h} horas", f"faz umas {h}h", f"desde hoje, umas {h} horas"])
    days = max(1, int(hours / 24))
    return rng.choice([f"ha {days} dias", f"desde {days} dias", f"faz uns {days} dias"])


def sample_case(rng: random.Random, typo_rate: float) -> SyntheticCase:
    age = int(max(0, min(95, rng.gammavariate(5, 8))))
    pain_level = rng.choices(range(11), weights=[3, 4, 6, 8, 10, 12, 13, 14, 12, 10, 8])[0]
    duration_hours = rng.choice([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 48, 72, 120, 168])

    flags = {
        "fever": int(rng.random() < 0.22),
        "shortness_of_breath": int(rng.random() < 0.13),
        "chest_pain": int(rng.random() < 0.12),
        "altered_consciousness": int(rng.random() < 0.05),
        "bleeding": int(rng.random() < 0.10),
        "vomiting": int(rng.random() < 0.18),
        "severe_headache": int(rng.random() < 0.16),
        "allergic_reaction": int(rng.random() < 0.08),
        "trauma": int(rng.random() < 0.12),
    }

    # Injeta combinacoes clinicamente importantes.
    if rng.random() < 0.08:
        flags["chest_pain"] = 1
        flags["shortness_of_breath"] = 1
        pain_level = max(pain_level, rng.choice([7, 8, 9]))
        duration_hours = rng.choice([0.25, 0.5, 1, 2])
    if rng.random() < 0.05:
        flags["allergic_reaction"] = 1
        flags["shortness_of_breath"] = 1
    if rng.random() < 0.04:
        flags["trauma"] = 1
        flags["bleeding"] = 1
        pain_level = max(pain_level, rng.choice([6, 7, 8]))

    temperature = round(rng.normalvariate(36.8, 0.45), 1)
    if flags["fever"]:
        temperature = round(rng.uniform(37.8, 40.4), 1)
    spo2 = int(rng.normalvariate(97, 2))
    if flags["shortness_of_breath"]:
        spo2 = int(rng.uniform(86, 95))
    heart_rate = int(rng.normalvariate(82, 15))
    if pain_level >= 7 or flags["fever"] or flags["bleeding"]:
        heart_rate += rng.randint(8, 35)

    comorbidity = rng.choice([
        "nenhuma", "hipertensao", "diabetes", "asma", "cardiopatia",
        "gestante", "idoso fragil", "dpoc",
    ])

    symptom_keys = [key for key, value in flags.items() if value]
    if not symptom_keys:
        symptom_keys = [rng.choice(["severe_headache", "vomiting", "fever"])]
        flags[symptom_keys[0]] = 1
    base = {"pain_level": pain_level, "duration_hours": duration_hours, "age": age, **flags}
    risk_color = manchester_rule(base)

    symptom_words = [rng.choice(SYMPTOMS[key]["phrases"]) for key in symptom_keys]
    symptoms_text = ", ".join(symptom_words[:-1]) + (" e " if len(symptom_words) > 1 else "") + symptom_words[-1]
    text = rng.choice(OPENERS).format(symptoms=symptoms_text)
    text += f", dor {rng.choice(INTENSITY_TEXT[pain_level])}"
    text += f", {duration_phrase(duration_hours, rng)}"
    if rng.random() < 0.45:
        text += f", tenho {age} anos"
    if rng.random() < 0.25:
        text += f", saturacao {spo2}%"
    if rng.random() < 0.20:
        text += f", temperatura {temperature}"
    if comorbidity != "nenhuma" and rng.random() < 0.35:
        text += f", tenho {comorbidity}"
    text = maybe_typo(text, rng, typo_rate)

    return SyntheticCase(
        text=text,
        risk_color=risk_color,
        symptoms=[SYMPTOMS[k]["pt"] for k in symptom_keys],
        pain_level=pain_level,
        duration_hours=duration_hours,
        age=age,
        heart_rate=max(35, min(220, heart_rate)),
        temperature=temperature,
        spo2=max(50, min(100, spo2)),
        comorbidity=comorbidity,
        **flags,
    )


def open_text(path: str, gzip_enabled: bool) -> TextIO:
    if gzip_enabled:
        return gzip.open(path + ".gz", "wt", encoding="utf-8", newline="")
    return open(path, "w", encoding="utf-8", newline="")


def generate(n: int, out_dir: str, chunk_size: int, seed: int, gzip_enabled: bool, typo_rate: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "triagem_massiva_estruturado.csv")
    jsonl_path = os.path.join(out_dir, "triagem_massiva_texto.jsonl")
    rng = random.Random(seed)

    csv_fields = [
        "texto", "sintoma", "intensidade", "duracao_min", "idade",
        "frequencia_cardiaca", "temperatura", "spo2", "comorbidade",
        "classificacao", *FEATURES, "risk_color",
    ]

    with open_text(csv_path, gzip_enabled) as csv_f, open_text(jsonl_path, gzip_enabled) as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()

        for i in range(1, n + 1):
            case = sample_case(rng, typo_rate)
            row = {
                "texto": case.text,
                "sintoma": "; ".join(case.symptoms),
                "intensidade": case.pain_level,
                "duracao_min": int(case.duration_hours * 60),
                "idade": case.age,
                "frequencia_cardiaca": case.heart_rate,
                "temperatura": case.temperature,
                "spo2": case.spo2,
                "comorbidade": case.comorbidity,
                "classificacao": PT_CLASS[case.risk_color],
                **{feat: getattr(case, feat) for feat in FEATURES},
                "risk_color": case.risk_color,
            }
            writer.writerow(row)
            jsonl_f.write(json.dumps({
                "texto": case.text,
                "classificacao": PT_CLASS[case.risk_color],
                "risk_color": case.risk_color,
                "sintomas": case.symptoms,
                "features": {feat: getattr(case, feat) for feat in FEATURES},
                "sinais_vitais": {
                    "frequencia_cardiaca": case.heart_rate,
                    "temperatura": case.temperature,
                    "spo2": case.spo2,
                },
                "comorbidade": case.comorbidity,
            }, ensure_ascii=False) + "\n")

            if i % chunk_size == 0:
                csv_f.flush()
                jsonl_f.flush()
                print(f"Gerados {i:,}/{n:,} exemplos")

    suffix = ".gz" if gzip_enabled else ""
    print(f"CSV : {csv_path}{suffix}")
    print(f"JSONL: {jsonl_path}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="numero de exemplos")
    parser.add_argument("--out-dir", default="data/generated", help="diretorio de saida")
    parser.add_argument("--chunk-size", type=int, default=50000, help="flush/progresso a cada N linhas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gzip", action="store_true", help="salvar .gz")
    parser.add_argument("--typo-rate", type=float, default=0.18, help="probabilidade de erro/giria")
    args = parser.parse_args()
    generate(args.n, args.out_dir, args.chunk_size, args.seed, args.gzip, args.typo_rate)


if __name__ == "__main__":
    main()
