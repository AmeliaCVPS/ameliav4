"""
api/database.py — Camada de Persistência da AMÉLIA
====================================================

Detecta automaticamente o ambiente:
  • Sem DATABASE_URL → SQLite local (desenvolvimento/feira)
  • Com DATABASE_URL  → PostgreSQL (Vercel + Supabase/Neon)

Conformidade LGPD:
  • CPF/SUS nunca armazenados — apenas hash SHA-256 irreversível
  • Dados clínicos cifrados com Fernet (AES-128) antes do banco
  • Campos em texto claro: cor, senha, timestamp (para relatórios)
"""

import os
import json
import hashlib
import sqlite3
from cryptography.fernet import Fernet

# ─────────────────────────────────────────────────────────────
# CHAVE DE CRIPTOGRAFIA
# ─────────────────────────────────────────────────────────────

def _init_cipher() -> Fernet:
    """
    Carrega ou gera a chave Fernet.

    Desenvolvimento: lê/cria 'secret.key' na raiz do projeto.
    Produção (Vercel): lê a variável de ambiente FERNET_KEY.

    ⚠️  Nunca comite secret.key. Adicione ao .gitignore!
    """
    raw = os.getenv("FERNET_KEY")
    if raw:
        return Fernet(raw.encode() if isinstance(raw, str) else raw)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key_path = os.path.join(root, "secret.key")

    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return Fernet(f.read())

    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    print(f"⚠️  Chave Fernet gerada → {key_path}  (adicione ao .gitignore!)")
    return Fernet(key)


CIPHER = _init_cipher()

# ─────────────────────────────────────────────────────────────
# DETECÇÃO DE BANCO
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")  # definido no Vercel
USE_POSTGRES = bool(DATABASE_URL)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(_ROOT, "amelia_local.db")


def _get_conn():
    """Retorna (conexão, tipo) para o banco correto."""
    if USE_POSTGRES:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        conn.cursor_factory = RealDictCursor
        return conn, "pg"
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


# ─────────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────────

def init_db():
    """Cria tabelas se não existirem. Chamada no startup da API."""
    conn, kind = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prontuarios (
            id             TEXT PRIMARY KEY,
            patient_hash   TEXT NOT NULL,
            timestamp      TEXT NOT NULL,
            encrypted_data TEXT NOT NULL,
            color          TEXT NOT NULL,
            password       TEXT NOT NULL
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ DB [{kind.upper()}] pronto.")


# ─────────────────────────────────────────────────────────────
# CRIPTOGRAFIA E HASH
# ─────────────────────────────────────────────────────────────

def encrypt(text: str) -> str:
    return CIPHER.encrypt(text.encode()).decode()

def decrypt(token: str) -> str:
    return CIPHER.decrypt(token.encode()).decode()

def hash_cpf(cpf: str) -> str:
    """Hash SHA-256 do CPF (somente dígitos). Irreversível."""
    return hashlib.sha256("".join(c for c in cpf if c.isdigit()).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# OPERAÇÕES
# ─────────────────────────────────────────────────────────────

def save_prontuario(prontuario: dict, cpf: str):
    """Criptografa e salva prontuário. CPF → hash."""
    conn, kind = _get_conn()
    cur = conn.cursor()

    blob = encrypt(json.dumps(prontuario, ensure_ascii=False))

    if kind == "pg":
        cur.execute("""
            INSERT INTO prontuarios (id, patient_hash, timestamp, encrypted_data, color, password)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET encrypted_data = EXCLUDED.encrypted_data
        """, (
            prontuario["id"],
            hash_cpf(cpf),
            prontuario["timestamp"],
            blob,
            prontuario["classification"]["color"],
            prontuario["classification"]["password"],
        ))
    else:
        cur.execute("""
            INSERT OR REPLACE INTO prontuarios
            (id, patient_hash, timestamp, encrypted_data, color, password)
            VALUES (?,?,?,?,?,?)
        """, (
            prontuario["id"],
            hash_cpf(cpf),
            prontuario["timestamp"],
            blob,
            prontuario["classification"]["color"],
            prontuario["classification"]["password"],
        ))

    conn.commit()
    cur.close()
    conn.close()


def load_prontuario(prontuario_id: str) -> dict | None:
    """Carrega e descriptografa prontuário pelo UUID."""
    conn, kind = _get_conn()
    cur = conn.cursor()

    q = "SELECT encrypted_data FROM prontuarios WHERE id = ?"
    if kind == "pg":
        q = q.replace("?", "%s")

    cur.execute(q, (prontuario_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return None

    raw = row["encrypted_data"] if kind == "pg" else row[0]
    return json.loads(decrypt(raw))
