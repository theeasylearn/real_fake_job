# Fake Job Posting Detection API Documentation

FastAPI service powered by a Bernoulli Naive Bayes classifier and binary bag-of-ngrams `CountVectorizer`, loaded into memory on startup for low-latency inference.

---

## Base URL

- **Local:** `http://127.0.0.1:8000`
- **Render.com:** `https://<your-service-name>.onrender.com`
- **Interactive Swagger UI:** `<BASE_URL>/docs`
- **ReDoc UI:** `<BASE_URL>/redoc`

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first startup, if `model_artifacts.joblib` is not present in the project directory, the service automatically downloads the dataset, trains the model, optimizes the threshold, saves `model_artifacts.joblib`, and keeps the model in memory (`app.state.artifacts`). On subsequent restarts (including on Render), it loads `model_artifacts.joblib` directly into memory in milliseconds.

---

## Deploying on Render.com

1. Push `main.py`, `requirements.txt`, and `model_artifacts.joblib` to your Git repository.
2. Create a new **Web Service** on [Render](https://render.com) connected to your repository.
3. Configure the service settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/health`

---

## Endpoints

### 1. `GET /` — Service Info

Returns basic service metadata and available endpoints.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/"
```

**Example Response:**
```json
{
  "service": "Fake Job Posting Detection API",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

---

### 2. `GET /health` — Health Check

Verifies the API is running and the model is loaded into memory.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Example Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "optimized_threshold": 0.9999999596609549
}
```

---

### 3. `POST /predict` — Predict Job Posting Fraud

Evaluates a job posting and returns its fraud probability, predicted class (`0` = Real, `1` = Fake), and verdict.

#### Request Body Schema (`application/json`)

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `title` | `string` | Yes | — | Job posting title (non-empty) |
| `description` | `string` | Yes | — | Full job description (non-empty) |
| `company_profile` | `string` | No | `""` | Company overview/profile |
| `requirements` | `string` | No | `""` | Job requirements |
| `benefits` | `string` | No | `""` | Job benefits |
| `telecommuting` | `0 \| 1` | No | `0` | `1` if remote/telecommuting, else `0` |
| `has_company_logo` | `0 \| 1` | No | `0` | `1` if company logo is present, else `0` |
| `has_questions` | `0 \| 1` | No | `0` | `1` if screening questions exist, else `0` |
| `has_salary` | `0 \| 1` | No | `0` | `1` if salary range is specified, else `0` |
| `threshold` | `float \| null` | No | `null` | Custom decision threshold (`0.0` to `1.0`). Uses the optimized threshold when omitted or `null`. |

#### Response Body Schema

| Field | Type | Description |
| --- | --- | --- |
| `title` | `string` | Job title from the request |
| `fraud_score` | `float` | Predicted probability of fraud (`0.0` to `1.0`) |
| `fraud_probability_percent` | `float` | `fraud_score` expressed as a percentage (`0.0` to `100.0`) |
| `predicted_class` | `0 \| 1` | `1` if `fraud_score >= threshold` (Fake), `0` otherwise (Real) |
| `verdict` | `string` | `"FAKE / SCAM"` or `"PREDICTED REAL"` |
| `threshold` | `float` | Decision threshold used for classification |

---

## Code Examples

### cURL — Fake Job Example

```bash
curl -X POST "https://<your-service-name>.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Urgent Work From Home Data Entry Clerk - Earn $3500/Week",
    "description": "Immediate start! No experience required. Earn daily cash payouts from home processing simple forms. Send bank details for direct wire transfer setup.",
    "requirements": "Must have internet access and basic typing skills.",
    "benefits": "Daily cash bonuses, flexible hours.",
    "telecommuting": 1,
    "has_company_logo": 0,
    "has_questions": 0,
    "has_salary": 1
  }'
```

**Example Response:**
```json
{
  "title": "Urgent Work From Home Data Entry Clerk - Earn $3500/Week",
  "fraud_score": 1.0,
  "fraud_probability_percent": 100.0,
  "predicted_class": 1,
  "verdict": "FAKE / SCAM",
  "threshold": 0.9999999596609549
}
```

---

### cURL — Second Example (with Company Profile & Logo)

```bash
curl -X POST "https://<your-service-name>.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Python Backend Engineer",
    "description": "We are looking for an experienced backend engineer to design scalable REST APIs, optimize PostgreSQL queries, and maintain CI/CD pipelines.",
    "company_profile": "Founded in 2016, CloudScale builds enterprise infrastructure monitoring tools.",
    "requirements": "4+ years of experience with Python, Django or FastAPI, Docker, and AWS.",
    "benefits": "Health insurance, 401(k) matching, annual learning stipend.",
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1,
    "has_salary": 1
  }'
```

**Example Response:**
```json
{
  "title": "Senior Python Backend Engineer",
  "fraud_score": 0.9999999995581561,
  "fraud_probability_percent": 100.0,
  "predicted_class": 1,
  "verdict": "FAKE / SCAM",
  "threshold": 0.9999999596609549
}
```

---

### Python (`requests`)

```python
import requests

API_URL = "https://<your-service-name>.onrender.com/predict"

payload = {
    "title": "Senior Python Backend Engineer",
    "description": (
        "We are looking for an experienced backend engineer to design "
        "scalable REST APIs, optimize PostgreSQL queries, and maintain CI/CD pipelines."
    ),
    "company_profile": "Founded in 2016, CloudScale builds enterprise infrastructure monitoring tools.",
    "requirements": "4+ years of experience with Python, Django or FastAPI, Docker, and AWS.",
    "benefits": "Health insurance, 401(k) matching, annual learning stipend.",
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1,
    "has_salary": 1,
}

response = requests.post(API_URL, json=payload, timeout=30)
response.raise_for_status()
print(response.json())
```

---

### JavaScript (`fetch`)

```javascript
const response = await fetch("https://<your-service-name>.onrender.com/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    title: "Urgent Work From Home Data Entry Clerk - Earn $3500/Week",
    description:
      "Immediate start! No experience required. Earn daily cash payouts from home processing simple forms. Send bank details for direct wire transfer setup.",
    requirements: "Must have internet access and basic typing skills.",
    benefits: "Daily cash bonuses, flexible hours.",
    telecommuting: 1,
    has_company_logo: 0,
    has_questions: 0,
    has_salary: 1,
  }),
});

if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${await response.text()}`);
}

const result = await response.json();
console.log(result);
```
