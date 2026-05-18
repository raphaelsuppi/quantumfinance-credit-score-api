"""
QuantumFinance Credit Score API — Versão de Deploy
====================================================
Arquivo único autocontido para deploy em Render / Railway / Fly.io

Inclui:
  - Pré-processamento inline (sem import de módulos locais)
  - Modelo treinado na inicialização com dados sintéticos
  - FastAPI com JWT, X-API-Key e rate limiting
  - Endpoints: /api/v1/token, /api/v1/score, /api/v1/score/batch, /api/v1/health
"""

import os, re, json, pickle, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
SECRET_KEY       = os.getenv("SECRET_KEY", "qf-quantumfinance-2026-secret")
ALGORITHM        = "HS256"
TOKEN_EXPIRE_MIN = 60
RATE_LIMIT_RPM   = int(os.getenv("RATE_LIMIT", "60"))
PORT             = int(os.getenv("PORT", "8000"))
API_V            = "v1"

VALID_API_KEYS = {
    "qf-partner-001": {"client_id": "parceiro_alpha", "name": "Parceiro Alpha"},
    "qf-partner-002": {"client_id": "parceiro_beta",  "name": "Parceiro Beta"},
    "qf-dev-key-000": {"client_id": "dev_sandbox",    "name": "Dev Sandbox"},
}

OCCUPATION_LIST = [
    "Scientist","Teacher","Engineer","Entrepreneur","Developer","Lawyer",
    "Media_Manager","Doctor","Journalist","Manager","Accountant",
    "Musician","Mechanic","Writer","Architect","Unknown"
]
PAYMENT_BEHAVIOURS = [
    "High_spent_Large_value_payments","High_spent_Medium_value_payments",
    "High_spent_Small_value_payments","Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments","Low_spent_Small_value_payments",
]
CREDIT_MIX_MAP      = {"Bad": 0, "Standard": 1, "Good": 2}
PAYMENT_MIN_MAP     = {"No": 0, "NM": 0, "Yes": 1}
TARGET_LABELS       = ["Poor", "Standard", "Good"]

SCORE_CONFIG = {
    0: (  0, 300, "Poor",     False),
    1: (300, 400, "Standard", True),
    2: (700, 300, "Good",     True),
}

# ──────────────────────────────────────────────
# TREINAMENTO NA INICIALIZAÇÃO
# ──────────────────────────────────────────────
_model = None
_encoders = {}
_feature_names = []
_model_version = "1"
_start_time = time.time()

def _generate_data(n=8000):
    np.random.seed(42)
    scores = np.random.choice([0,1,2], n, p=[0.20,0.53,0.27])
    occ_enc = LabelEncoder().fit(OCCUPATION_LIST)
    pb_enc  = LabelEncoder().fit(PAYMENT_BEHAVIOURS)

    X = pd.DataFrame({
        "Age":                     np.clip(np.random.normal(40,12,n).astype(int), 18, 80),
        "Occupation_enc":          np.random.randint(0, len(OCCUPATION_LIST), n),
        "Annual_Income":           np.clip(np.random.lognormal(11,0.6,n), 5000, 300000),
        "Monthly_Inhand_Salary":   np.clip(np.random.lognormal(8.3,0.5,n), 500, 20000),
        "Num_Bank_Accounts":       np.random.randint(1,10,n),
        "Num_Credit_Card":         np.random.randint(1,10,n),
        "Interest_Rate":           np.random.randint(1,35,n),
        "Num_of_Loan":             np.random.randint(0,8,n),
        "Delay_from_due_date":     np.abs(np.random.normal(8,10,n)).astype(int),
        "Num_of_Delayed_Payment":  np.abs(np.random.normal(4,5,n)).astype(int),
        "Changed_Credit_Limit":    np.random.uniform(-5,30,n),
        "Num_Credit_Inquiries":    np.random.randint(0,10,n),
        "Credit_Mix":              np.random.choice([0,1,2], n),
        "Outstanding_Debt":        np.clip(np.random.exponential(1500,n), 0, 20000),
        "Credit_Utilization_Ratio":np.random.uniform(10,80,n),
        "Credit_History_Months":   np.random.randint(12,360,n),
        "Payment_of_Min_Amount":   np.random.choice([0,1], n),
        "Total_EMI_per_month":     np.clip(np.random.exponential(200,n), 0, 3000),
        "Amount_invested_monthly": np.clip(np.random.exponential(300,n), 0, 5000),
        "Payment_Behaviour_enc":   np.random.randint(0,len(PAYMENT_BEHAVIOURS),n),
        "Monthly_Balance":         np.clip(np.random.lognormal(6,0.8,n), 0, 8000),
    })

    # Correlacionar features com o target para o modelo aprender
    for i in range(n):
        s = scores[i]
        if s == 2:   # Good
            X.loc[i,"Delay_from_due_date"]     = max(0, X.loc[i,"Delay_from_due_date"] - 8)
            X.loc[i,"Num_of_Delayed_Payment"]  = max(0, X.loc[i,"Num_of_Delayed_Payment"] - 4)
            X.loc[i,"Credit_Mix"]              = 2
            X.loc[i,"Outstanding_Debt"]        *= 0.4
            X.loc[i,"Credit_History_Months"]   += 60
        elif s == 0: # Poor
            X.loc[i,"Delay_from_due_date"]    += 20
            X.loc[i,"Num_of_Delayed_Payment"] += 8
            X.loc[i,"Credit_Mix"]              = 0
            X.loc[i,"Outstanding_Debt"]       *= 2.5
            X.loc[i,"Credit_History_Months"]   = max(1, X.loc[i,"Credit_History_Months"] - 40)

    # Features derivadas
    X["Debt_to_Income"] = X["Outstanding_Debt"] / (X["Annual_Income"] + 1)
    X["EMI_to_Salary"]  = X["Total_EMI_per_month"] / (X["Monthly_Inhand_Salary"] + 1)
    X["Has_Delay"]      = (X["Delay_from_due_date"] > 0).astype(int)

    return X, pd.Series(scores)

def _train_model():
    global _model, _encoders, _feature_names

    print("🔧 Treinando modelo de demonstração...")
    X, y = _generate_data(8000)
    _feature_names = X.columns.tolist()
    _encoders["features"] = _feature_names

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    _model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    _model.fit(X_train, y_train)
    print(f"✅ Modelo treinado — features: {len(_feature_names)}")

def _parse_credit_history_age(val):
    if not isinstance(val, str):
        return 60
    years  = re.search(r"(\d+)\s*Year",  val)
    months = re.search(r"(\d+)\s*Month", val)
    y = int(years.group(1))  if years  else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m

def _preprocess_input(data: dict) -> pd.DataFrame:
    occ = data.get("Occupation", "Unknown")
    if occ not in OCCUPATION_LIST:
        occ = "Unknown"
    occ_idx = OCCUPATION_LIST.index(occ)

    pb = data.get("Payment_Behaviour", PAYMENT_BEHAVIOURS[4])
    if pb not in PAYMENT_BEHAVIOURS:
        pb = PAYMENT_BEHAVIOURS[4]
    pb_idx = PAYMENT_BEHAVIOURS.index(pb)

    outstanding = float(data.get("Outstanding_Debt", 0))
    annual      = float(data.get("Annual_Income", 1))
    emi         = float(data.get("Total_EMI_per_month", 0))
    salary      = float(data.get("Monthly_Inhand_Salary", 1))
    delay       = int(data.get("Delay_from_due_date", 0))

    row = {
        "Age":                     int(data.get("Age", 35)),
        "Occupation_enc":          occ_idx,
        "Annual_Income":           annual,
        "Monthly_Inhand_Salary":   salary,
        "Num_Bank_Accounts":       int(data.get("Num_Bank_Accounts", 3)),
        "Num_Credit_Card":         int(data.get("Num_Credit_Card", 3)),
        "Interest_Rate":           int(data.get("Interest_Rate", 12)),
        "Num_of_Loan":             int(data.get("Num_of_Loan", 2)),
        "Delay_from_due_date":     delay,
        "Num_of_Delayed_Payment":  int(data.get("Num_of_Delayed_Payment", 2)),
        "Changed_Credit_Limit":    float(data.get("Changed_Credit_Limit", 5.0)),
        "Num_Credit_Inquiries":    int(data.get("Num_Credit_Inquiries", 3)),
        "Credit_Mix":              CREDIT_MIX_MAP.get(data.get("Credit_Mix","Standard"), 1),
        "Outstanding_Debt":        outstanding,
        "Credit_Utilization_Ratio":float(data.get("Credit_Utilization_Ratio", 30.0)),
        "Credit_History_Months":   _parse_credit_history_age(data.get("Credit_History_Age","5 Years and 0 Months")),
        "Payment_of_Min_Amount":   PAYMENT_MIN_MAP.get(data.get("Payment_of_Min_Amount","No"), 0),
        "Total_EMI_per_month":     emi,
        "Amount_invested_monthly": float(data.get("Amount_invested_monthly", 300.0)),
        "Payment_Behaviour_enc":   pb_idx,
        "Monthly_Balance":         float(data.get("Monthly_Balance", 500.0)),
        "Debt_to_Income":          outstanding / (annual + 1),
        "EMI_to_Salary":           emi / (salary + 1),
        "Has_Delay":               1 if delay > 0 else 0,
    }
    return pd.DataFrame([row])[_feature_names]

def _probas_to_result(probas) -> dict:
    pred = int(np.argmax(probas))
    base, amp, label, approved = SCORE_CONFIG[pred]
    score = int(base + amp * probas[pred])
    if   score >= 950: rating = "AAA"
    elif score >= 850: rating = "AA"
    elif score >= 700: rating = "A"
    elif score >= 580: rating = "BBB"
    elif score >= 450: rating = "BB"
    elif score >= 300: rating = "B"
    else:              rating = "C"
    return {
        "score":         score,
        "rating":        rating,
        "credit_class":  label,
        "approved":      approved,
        "prob_poor":     round(float(probas[0]), 4),
        "prob_standard": round(float(probas[1]), 4),
        "prob_good":     round(float(probas[2]), 4),
        "version":       _model_version,
    }

# ──────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────
app = FastAPI(
    title="QuantumFinance Credit Score API",
    description=(
        "API de score de crédito — MBA QuantumFinance.\n\n"
        "**Dataset:** Kaggle Credit Score Classification (parisrohan)\n\n"
        "**Target:** Poor / Standard / Good\n\n"
        "**Autenticação:** Bearer JWT ou X-API-Key\n\n"
        "**API Keys disponíveis para teste:**\n"
        "- `qf-partner-001` (Parceiro Alpha)\n"
        "- `qf-partner-002` (Parceiro Beta)\n"
        "- `qf-dev-key-000` (Sandbox)\n\n"
        "**Rate limit:** 60 req/min por cliente"
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

bearer_scheme  = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_rate_store: dict[str, list[float]] = defaultdict(list)


# ── Schemas ──────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    api_key: str = Field(..., example="qf-partner-001")

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int

class ScoreRequest(BaseModel):
    Age:                      int   = Field(..., ge=18, le=100,  example=35)
    Occupation:               str   = Field(...,                  example="Engineer")
    Annual_Income:            float = Field(..., gt=0,            example=65000.0)
    Monthly_Inhand_Salary:    float = Field(..., ge=0,            example=4500.0)
    Num_Bank_Accounts:        int   = Field(..., ge=0, le=20,     example=3)
    Num_Credit_Card:          int   = Field(..., ge=0, le=15,     example=4)
    Interest_Rate:            int   = Field(..., ge=1, le=50,     example=12)
    Num_of_Loan:              int   = Field(..., ge=0, le=15,     example=2)
    Delay_from_due_date:      int   = Field(..., ge=0, le=100,    example=5)
    Num_of_Delayed_Payment:   int   = Field(..., ge=0, le=30,     example=2)
    Changed_Credit_Limit:     float = Field(...,                  example=5.5)
    Num_Credit_Inquiries:     int   = Field(..., ge=0, le=20,     example=3)
    Credit_Mix:               str   = Field(...,                  example="Standard")
    Outstanding_Debt:         float = Field(..., ge=0,            example=1200.0)
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100,    example=32.5)
    Credit_History_Age:       str   = Field(...,                  example="8 Years and 3 Months")
    Payment_of_Min_Amount:    str   = Field(...,                  example="No")
    Total_EMI_per_month:      float = Field(..., ge=0,            example=200.0)
    Amount_invested_monthly:  float = Field(..., ge=0,            example=500.0)
    Payment_Behaviour:        str   = Field(...,                  example="Low_spent_Medium_value_payments")
    Monthly_Balance:          float = Field(..., ge=0,            example=800.0)

class ScoreResponse(BaseModel):
    score:         int
    rating:        str
    credit_class:  str
    approved:      bool
    prob_poor:     float
    prob_standard: float
    prob_good:     float
    version:       str
    timestamp:     str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _create_jwt(client_id: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    return jwt.encode({"sub": client_id, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def _decode_jwt(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        cid = payload.get("sub")
        if not cid:
            raise HTTPException(401, "Token inválido")
        return cid
    except JWTError:
        raise HTTPException(401, "Token expirado ou inválido")

def _check_rate(client_id: str):
    now  = time.time()
    hits = [t for t in _rate_store[client_id] if now - t < 60.0]
    if len(hits) >= RATE_LIMIT_RPM:
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_RPM} req/min",
                            headers={"Retry-After": "60"})
    hits.append(now)
    _rate_store[client_id] = hits

def _get_client(
    bearer:  Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    if bearer and bearer.credentials:
        return _decode_jwt(bearer.credentials)
    if api_key:
        info = VALID_API_KEYS.get(api_key)
        if info:
            return info["client_id"]
    raise HTTPException(401, "Autenticação necessária: Bearer token ou X-API-Key",
                        headers={"WWW-Authenticate": "Bearer"})


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    _train_model()

@app.post(f"/api/{API_V}/token", response_model=TokenResponse, tags=["Autenticação"],
          summary="Obter Token JWT")
def get_token(body: TokenRequest):
    """
    Gera um Bearer JWT a partir de uma API Key válida.

    **API Keys para teste:**
    - `qf-partner-001`
    - `qf-partner-002`
    - `qf-dev-key-000`
    """
    info = VALID_API_KEYS.get(body.api_key)
    if not info:
        raise HTTPException(401, "API Key inválida")
    return TokenResponse(access_token=_create_jwt(info["client_id"]),
                         expires_in=TOKEN_EXPIRE_MIN * 60)

@app.post(f"/api/{API_V}/score", response_model=ScoreResponse, tags=["Score"],
          summary="Calcular Score de Crédito")
def get_score(body: ScoreRequest, client_id: str = Depends(_get_client)):
    """
    Retorna o score de crédito de um cliente.

    **Score:** 0–1000 (maior = melhor pagador)

    **Ratings:** AAA > AA > A > BBB > BB > B > C

    **Classes:** Good (aprovado) / Standard (aprovado) / Poor (reprovado)
    """
    _check_rate(client_id)
    X      = _preprocess_input(body.model_dump())
    probas = _model.predict_proba(X)[0]
    result = _probas_to_result(probas)
    return ScoreResponse(**result, timestamp=datetime.utcnow().isoformat() + "Z")

@app.post(f"/api/{API_V}/score/batch", tags=["Score"],
          summary="Score em Lote (até 100 clientes)")
def get_score_batch(
    body: dict,
    client_id: str = Depends(_get_client)
):
    """
    Predição em lote para até 100 clientes.

    **Payload:** `{"clients": [<ScoreRequest>, ...]}`
    """
    _check_rate(client_id)
    clients = body.get("clients", [])
    if not clients or len(clients) > 100:
        raise HTTPException(422, "Entre 1 e 100 clientes por requisição")
    results = []
    for i, c in enumerate(clients):
        try:
            X = _preprocess_input(c)
            p = _model.predict_proba(X)[0]
            r = _probas_to_result(p)
            r["client_index"] = i
        except Exception as e:
            r = {"client_index": i, "error": str(e)}
        results.append(r)
    return {"results": results, "total": len(results),
            "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get(f"/api/{API_V}/health", tags=["Operacional"],
         summary="Health Check")
def health():
    """Verifica disponibilidade da API. Não requer autenticação."""
    return {"status": "healthy", "model_version": _model_version,
            "uptime_s": round(time.time() - _start_time, 1),
            "model_trained": _model is not None}

@app.get("/", include_in_schema=False)
def root():
    return {"message": "QuantumFinance Credit Score API",
            "docs": "/docs", "health": f"/api/{API_V}/health"}

@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    return JSONResponse(exc.status_code,
                        {"error": {"code": exc.status_code, "message": exc.detail,
                                   "timestamp": datetime.utcnow().isoformat() + "Z"}})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
