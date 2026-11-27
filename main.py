"""FastAPI service for Fake Job Posting Detection using Bernoulli Naive Bayes."""
# https://real-fake-job.onrender.com/predict

from contextlib import asynccontextmanager
import os
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import kagglehub
import pandas as pd
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "model_artifacts.joblib")

DATASET_SLUG = "shivamb/real-or-fake-fake-jobposting-prediction"
DATASET_FILENAME = "fake_job_postings.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
ALPHA = 1.0
MIN_RECALL = 0.85

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions", "has_salary"]


def _optimize_threshold(y_val, y_val_prob) -> float:
    """Find the threshold meeting MIN_RECALL with the fewest false positives."""
    fpr_values, tpr_values, thresholds = roc_curve(
        y_val,
        y_val_prob,
        drop_intermediate=False,
    )
    total_real = int((y_val == 0).sum())

    best_fp: Optional[int] = None
    best_recall = -1.0
    best_threshold: Optional[float] = None

    for i, threshold in enumerate(thresholds):
        recall = float(tpr_values[i])
        if recall < MIN_RECALL:
            continue

        fp = int(round(fpr_values[i] * total_real))
        if best_fp is None or fp < best_fp or (fp == best_fp and recall > best_recall):
            best_fp = fp
            best_recall = recall
            best_threshold = float(threshold)

    if best_threshold is None:
        raise ValueError("No threshold satisfies the minimum recall.")

    return best_threshold


def train_and_optimize_model() -> dict:
    """Download dataset, train BernoulliNB + CountVectorizer, and optimize threshold."""
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    df = pd.read_csv(os.path.join(dataset_path, DATASET_FILENAME))

    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df["full_text"] = (
        df["title"]
        + " "
        + df["company_profile"]
        + " "
        + df["description"]
        + " "
        + df["requirements"]
        + " "
        + df["benefits"]
    )
    df["has_salary"] = (
        df["salary_range"].fillna("").astype(str).str.strip().ne("").astype(int)
    )

    x_text = df["full_text"]
    x_meta = df[BINARY_COLS].values
    y = df["fraudulent"].astype(int)

    x_text_tv, _, x_meta_tv, _, y_tv, _ = train_test_split(
        x_text,
        x_meta,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_text_train, x_text_val, x_meta_train, x_meta_val, y_train, y_val = train_test_split(
        x_text_tv,
        x_meta_tv,
        y_tv,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_tv,
    )

    vectorizer = CountVectorizer(
        binary=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
    )
    x_train_final = hstack(
        [vectorizer.fit_transform(x_text_train), csr_matrix(x_meta_train)],
        format="csr",
    )
    x_val_final = hstack(
        [vectorizer.transform(x_text_val), csr_matrix(x_meta_val)],
        format="csr",
    )

    model = BernoulliNB(alpha=ALPHA, binarize=0.0)
    model.fit(x_train_final, y_train)

    y_val_prob = model.predict_proba(x_val_final)[:, 1]
    best_threshold = _optimize_threshold(y_val, y_val_prob)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "best_threshold": best_threshold,
    }


def load_model_into_memory() -> dict:
    """Load pre-trained artifacts into memory, or train and cache them if missing."""
    if os.path.exists(MODEL_ARTIFACT_PATH):
        return joblib.load(MODEL_ARTIFACT_PATH)

    artifacts = train_and_optimize_model()
    joblib.dump(artifacts, MODEL_ARTIFACT_PATH, compress=3)
    return artifacts


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model, vectorizer, and threshold into RAM once at startup."""
    application.state.artifacts = load_model_into_memory()
    yield
    application.state.artifacts = None


app = FastAPI(
    title="Fake Job Posting Detection API",
    description="Detects fraudulent job postings using an in-memory Bernoulli Naive Bayes model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobPostingRequest(BaseModel):
    """Input payload representing a job posting to evaluate."""

    title: str = Field(..., min_length=1, description="Job posting title")
    description: str = Field(..., min_length=1, description="Full job description")
    company_profile: str = Field(default="", description="Company overview/profile")
    requirements: str = Field(default="", description="Job requirements")
    benefits: str = Field(default="", description="Job benefits")
    telecommuting: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if remote/telecommuting, 0 otherwise",
    )
    has_company_logo: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if posting includes a company logo, 0 otherwise",
    )
    has_questions: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if posting includes screening questions, 0 otherwise",
    )
    has_salary: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if salary range is specified, 0 otherwise",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom decision threshold between 0 and 1",
    )


class PredictionResponse(BaseModel):
    """Prediction result returned by the fraud detection model."""

    title: str
    fraud_score: float
    fraud_probability_percent: float
    predicted_class: Literal[0, 1]
    verdict: str
    threshold: float


class HealthResponse(BaseModel):
    """Service health status."""

    status: str
    model_loaded: bool
    optimized_threshold: Optional[float]


@app.get("/", tags=["General"])
def root():
    """Root endpoint with service overview."""
    return {
        "service": "Fake Job Posting Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check(request: Request):
    """Health check endpoint verifying the model is loaded in memory."""
    artifacts = getattr(request.app.state, "artifacts", None)
    return HealthResponse(
        status="ok" if artifacts else "initializing",
        model_loaded=artifacts is not None,
        optimized_threshold=artifacts["best_threshold"] if artifacts else None,
    )


def _run_prediction(payload: JobPostingRequest, request: Request) -> PredictionResponse:
    """Run fraud prediction for a validated job posting payload using the in-memory model."""
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded in memory.")

    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    active_threshold = (
        payload.threshold if payload.threshold is not None else artifacts["best_threshold"]
    )

    combined_text = (
        f"{payload.title} "
        f"{payload.company_profile} "
        f"{payload.description} "
        f"{payload.requirements} "
        f"{payload.benefits}"
    )

    text_features = vectorizer.transform([combined_text])
    meta_features = csr_matrix(
        [
            [
                payload.telecommuting,
                payload.has_company_logo,
                payload.has_questions,
                payload.has_salary,
            ]
        ]
    )
    combined_features = hstack([text_features, meta_features], format="csr")

    fake_prob = float(model.predict_proba(combined_features)[0, 1])
    predicted_class = 1 if fake_prob >= active_threshold else 0
    verdict = "FAKE / SCAM" if predicted_class == 1 else "PREDICTED REAL"

    return PredictionResponse(
        title=payload.title,
        fraud_score=fake_prob,
        fraud_probability_percent=round(fake_prob * 100, 2),
        predicted_class=predicted_class,
        verdict=verdict,
        threshold=active_threshold,
    )


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_job_posting_get(
    payload: Annotated[JobPostingRequest, Query()],
    request: Request,
):
    """Predict whether a job posting is real (0) or fake/scam (1) via GET query parameters."""
    return _run_prediction(payload, request)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_job_posting_post(payload: JobPostingRequest, request: Request):
    """Predict whether a job posting is real (0) or fake/scam (1) via POST JSON body."""
    return _run_prediction(payload, request)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
