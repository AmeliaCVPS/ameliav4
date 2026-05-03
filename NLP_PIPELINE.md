# Pipeline NLP Clinico AMELIA

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para treinar BERT em GPU, instale uma versao do PyTorch compativel com sua CUDA antes de rodar o treino.

## 2. Gerar dataset massivo

Teste pequeno:

```bash
python generate_massive_dataset.py --n 10000 --out-dir data/generated
```

Escala maior, com gzip:

```bash
python generate_massive_dataset.py --n 2000000 --chunk-size 50000 --gzip --out-dir data/generated
```

Saidas:

- `data/generated/triagem_massiva_estruturado.csv`
- `data/generated/triagem_massiva_texto.jsonl`

O CSV inclui colunas em portugues e as 12 features usadas pelo Random Forest.

## 3. Treinar BERT

```bash
python train_bert_triage.py ^
  --data data/generated/triagem_massiva_texto.jsonl ^
  --output-dir models/bert_triage ^
  --epochs 3 ^
  --batch-size 8
```

Modelo base padrao:

```text
neuralmind/bert-base-portuguese-cased
```

Para teste rapido:

```bash
python train_bert_triage.py --data data/generated/triagem_massiva_texto.jsonl --sample 2000 --epochs 1
```

## 4. Rodar API

```bash
uvicorn API.index:app --reload
```

Endpoint NLP principal:

```http
POST /api/triagem
Content-Type: application/json

{
  "texto": "Estou com aperto no peito muito forte ha 10 minutos e falta de ar",
  "age": 62,
  "sex": "M",
  "cpf": "00000000000",
  "use_bert": true
}
```

Resposta esperada:

```json
{
  "classificacao": "vermelho",
  "sintomas": ["falta de ar", "dor no peito"],
  "confianca": 0.92,
  "metodo": "nlp_features_random_forest",
  "password": "",
  "classification": {
    "color": "red",
    "priority": 1,
    "wait_time": "Imediato",
    "explanation": "Emergência — atendimento imediato por médico(a) ou enfermeiro(a).",
    "confidence": 0.92
  }
}
```

Se `models/bert_triage` existir, o endpoint tenta BERT direto. Se nao existir ou a confianca for baixa, usa o pipeline hibrido `texto -> features -> Random Forest`.

## 5. Frontend

O frontend ja tenta `/api/triagem` na finalizacao da triagem. Se esse endpoint nao estiver disponivel, ele volta para `/api/classify`.

