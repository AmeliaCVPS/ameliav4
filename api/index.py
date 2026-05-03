"""
api/index.py — API Principal da AMÉLIA v3.0
============================================
Rotas:
  GET  /api/health                → Status
  POST /api/classify              → Triagem via JSON estruturado
  POST /api/classify/from-text    → Triagem via texto livre (voz)
  GET  /api/prontuario/{id}       → Consulta prontuário
"""

import re, uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .database import init_db, save_prontuario, load_prontuario
from .ml_model  import predict_risk, SymptomsInput
from .nlp_pipeline import triage_text, symptoms_to_human

app = FastAPI(
    title="A.M.E.L.I.A — API de Triagem",
    description="Random Forest 97.74% · 12 features · Protocolo de Manchester",
    version="3.0.0",
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.on_event("startup")
async def _startup():
    init_db()
    print("🚀 A.M.E.L.I.A API v3.0 iniciada!")


# ── Schema texto livre ────────────────────────────────────────
class TextTriageInput(BaseModel):
    cpf:         str            = Field(...)
    age:         int            = Field(..., ge=0, le=120)
    sex:         str            = Field("M", pattern="^[MF]$")
    text:        str            = Field(..., max_length=4000)
    temperature: Optional[float] = None


class FreeTextTriageInput(BaseModel):
    texto:       str             = Field(..., min_length=2, max_length=4000)
    cpf:         str             = Field("00000000000")
    age:         int             = Field(30, ge=0, le=120)
    sex:         str             = Field("M", pattern="^[MF]$")
    temperature: Optional[float] = None
    use_bert:    bool            = True


# ── Extração de sintomas do texto ────────────────────────────
def _clean_patient_text(text: str) -> str:
    """Remove emojis e símbolos de variação antes de salvar/analisar o relato."""
    text = re.sub("[\U00010000-\U0010ffff]", "", text or "")
    text = text.replace("\ufe0f", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", text).strip()


def _extract(text: str, cpf: str, age: int, sex: str,
             temperature: Optional[float] = None) -> SymptomsInput:
    """
    Análise léxica completa do relato do paciente.
    Extrai todas as 12 features para o modelo ML.
    """
    clean_text = _clean_patient_text(text)
    t = clean_text.lower()

    # ── Nível de dor (0-10) ──────────────────────────────────
    pain = 5
    nums = re.findall(r'\b(10|[0-9])\b', t)
    if nums:
        pain = int(nums[0])
    elif any(w in t for w in ["insuportável","horrível","terrível","agonia",
                               "pior dor","absurda","excruciante"]):
        pain = 10
    elif any(w in t for w in ["muito forte","intensa","severa","aguda","forte demais"]):
        pain = 8
    elif any(w in t for w in ["forte","considerável","bastante"]):
        pain = 7
    elif any(w in t for w in ["moderada","média","razoável","suportável"]):
        pain = 5
    elif any(w in t for w in ["leve","fraca","pequena","pouca","levinha"]):
        pain = 3
    elif any(w in t for w in ["mínima","quase nada","quase não","imperceptível"]):
        pain = 1
    elif any(w in t for w in ["sem dor","não doi","não tenho dor","nenhuma dor"]):
        pain = 0

    # ── Febre ────────────────────────────────────────────────
    fever = any(w in t for w in [
        "febre","febril","temperatura alta","quente demais",
        "38","39","40","37.5","37,5","temperatura elevada",
        "com febre","está febril","mediu febre",
    ])

    # ── Falta de ar ──────────────────────────────────────────
    sob = any(w in t for w in [
        "falta de ar","sem ar","dificuldade para respirar",
        "não consigo respirar","ofego","sufocando","sufocação",
        "respiração difícil","cansaço ao respirar","dispneia",
        "fôlego","não está conseguindo respirar","ofegante",
        "pneumonia","bronquite","crise asmática","asma",
    ])

    # ── Dor no peito ─────────────────────────────────────────
    chest = any(w in t for w in [
        "dor no peito","aperto no peito","pressão no peito",
        "dor torácica","coração doendo","infarto","angina",
        "irradiação para o braço","dor irradiando","aperto",
        "peso no peito","sensação de aperto","dor precordial",
    ])

    # ── Consciência alterada ─────────────────────────────────
    cons = any(w in t for w in [
        "desmaiei","perdi a consciência","apagou","convulsão",
        "convulsionando","desorientado","confuso","não responde",
        "tontura intensa","não consigo ficar de pé","síncope",
        "perdeu o sentido","desmaiou","tonteira muito forte",
        "ataque epiléptico","epilepsia","tremedeira","espasmo",
    ])

    # ── Sangramento ──────────────────────────────────────────
    bleed = any(w in t for w in [
        "sangue","sangrando","sangramento","hemorragia",
        "vomitei sangue","fezes negras","melena","hematêmese",
        "urina com sangue","epistaxe","nariz sangrando",
        "sangue na urina","hemoptise","sangue nas fezes",
        "ferida com sangue","cortado","corte profundo",
    ])

    # ── Vômito ───────────────────────────────────────────────
    vomit = any(w in t for w in [
        "vômito","vomitando","vomitei","náusea intensa",
        "enjoo forte","ânsia","vomitou muito","não para de vomitar",
        "vômitos repetidos","enjôo","jogou tudo para fora",
    ])

    # ── Dor de cabeça intensa ────────────────────────────────
    headache = any(w in t for w in [
        "dor de cabeça","cefaleia","enxaqueca","cabeça latejando",
        "dor na cabeça muito forte","pior dor de cabeça",
        "cabeça doendo muito","dor intensa na cabeça",
        "pressão na cabeça","migranea","migrânea","cabeça explodindo",
    ])

    # ── Reação alérgica ──────────────────────────────────────
    allergy = any(w in t for w in [
        "alergia","alérgico","alérgica","reação alérgica",
        "urticária","coceira no corpo todo","inchaço na garganta",
        "anafilaxia","erupção cutânea","vermelhidão no corpo",
        "picada de inseto","abelha","vespa","ferroada",
        "comeu algo","ingestão de alimento","choque alérgico",
    ])

    # ── Trauma / acidente ────────────────────────────────────
    trauma = any(w in t for w in [
        "acidente","caí","queda","bateu a cabeça","bati","trauma",
        "colisão","batida de carro","atropelado","caiu",
        "pancada","bateu","machucou","machucado","lesão",
        "fratura","osso","tornozel","tornozelo torcido","entorse",
        "queimadura","escaldou","queimou","ferimento","ferida",
    ])

    # ── Duração ──────────────────────────────────────────────
    dur: float = 24.0
    if m := re.search(r'(\d+(?:[.,]\d+)?)\s*semana', t):
        dur = float(m.group(1).replace(',','.')) * 168
    elif m := re.search(r'(\d+(?:[.,]\d+)?)\s*dia', t):
        dur = float(m.group(1).replace(',','.')) * 24
    elif m := re.search(r'(\d+(?:[.,]\d+)?)\s*hora', t):
        dur = max(0.5, float(m.group(1).replace(',','.')))
    elif m := re.search(r'(\d+(?:[.,]\d+)?)\s*minuto', t):
        dur = max(0.25, float(m.group(1).replace(',','.')) / 60)
    elif any(w in t for w in ["agora mesmo","agora pouco","acabou de","há pouco","minutos"]):
        dur = 0.25
    elif any(w in t for w in ["hoje cedo","esta manhã","esta tarde","hoje à tarde"]):
        dur = 4.0
    elif any(w in t for w in ["ontem","desde ontem","noite passada"]):
        dur = 24.0
    elif any(w in t for w in ["há dias","vários dias","alguns dias","poucos dias"]):
        dur = 72.0
    elif any(w in t for w in ["semanas","faz tempo","há muito","crônico","crónico"]):
        dur = 336.0

    return SymptomsInput(
        cpf=cpf, age=age, sex=sex, description=clean_text,
        pain_level=pain,
        fever=fever,
        shortness_of_breath=sob,
        chest_pain=chest,
        altered_consciousness=cons,
        bleeding=bleed,
        duration_hours=dur,
        vomiting=vomit,
        severe_headache=headache,
        allergic_reaction=allergy,
        trauma=trauma,
        temperature=temperature,
    )


# ── Helper ────────────────────────────────────────────────────
def _build_prontuario(data: SymptomsInput, cls: dict) -> dict:
    return {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient":   {"age": data.age, "sex": data.sex},
        "symptoms": {
            "description":           data.description,
            "pain_level":            data.pain_level,
            "fever":                 data.fever,
            "shortness_of_breath":   data.shortness_of_breath,
            "chest_pain":            data.chest_pain,
            "altered_consciousness": data.altered_consciousness,
            "bleeding":              data.bleeding,
            "duration_hours":        data.duration_hours,
            "vomiting":              data.vomiting,
            "severe_headache":       data.severe_headache,
            "allergic_reaction":     data.allergic_reaction,
            "trauma":                data.trauma,
        },
        "vital_signs": {
            "temperature":       data.temperature,
            "heart_rate":        data.heart_rate,
            "oxygen_saturation": data.oxygen_saturation,
        },
        "classification": cls,
    }


# ── Rotas ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "online", "system": "A.M.E.L.I.A", "version": "3.0.0"}


@app.post("/api/classify")
async def classify(data: SymptomsInput):
    """Triagem via dados estruturados (botões do frontend)."""
    try:
        cls  = predict_risk(data)
        pron = _build_prontuario(data, cls)
        save_prontuario(pron, data.cpf)
        message = (
            f"{cls['explanation']} Sem senha de espera: atendimento imediato."
            if cls["color"] == "red"
            else f"{cls['explanation']} Senha: {cls['password']}."
        )
        return {
            "prontuario_id": pron["id"],
            "password":      cls["password"],
            "classification": {
                "color":       cls["color"],
                "priority":    cls["priority"],
                "wait_time":   cls["wait_time"],
                "explanation": cls["explanation"],
                "confidence":  cls["confidence"],
            },
            "message": message,
        }
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/classify/from-text")
async def classify_from_text(data: TextTriageInput):
    """Triagem via texto livre (relato por voz ou digitação livre)."""
    try:
        symp = _extract(data.text, data.cpf, data.age, data.sex, data.temperature)
        return await classify(symp)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/triagem")
@app.post("/api/triagem")
async def triagem_nlp(data: FreeTextTriageInput):
    """
    Endpoint NLP principal:
      texto -> BERT (se treinado) + features NLP -> Random Forest
    """
    try:
        result = triage_text(
            data.texto,
            cpf=data.cpf,
            age=data.age,
            sex=data.sex,
            temperature=data.temperature,
            use_bert=data.use_bert,
        )
        cls = result["classification"]
        password = result.get("password") or cls.get("password", "")
        message = (
            f"{cls['explanation']} Sem senha de espera: atendimento imediato."
            if cls["color"] == "red"
            else f"{cls['explanation']} Senha: {password}."
        )
        return {
            "classificacao": result["classificacao"],
            "sintomas": symptoms_to_human(result["sintomas"]),
            "entidades": result["nlp"].get("entities", []),
            "confianca": result["confianca"],
            "metodo": result["metodo"],
            "password": password,
            "classification": {
                "color": cls["color"],
                "priority": cls.get("priority"),
                "wait_time": cls.get("wait_time"),
                "explanation": cls.get("explanation"),
                "confidence": cls.get("confidence"),
            },
            "features": result["features"],
            "nlp": result["nlp"],
            "bert": result.get("bert"),
            "message": message,
        }
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/prontuario/{prontuario_id}")
async def get_prontuario(prontuario_id: str):
    p = load_prontuario(prontuario_id)
    if not p:
        raise HTTPException(404, detail="Prontuário não encontrado.")
    return p
