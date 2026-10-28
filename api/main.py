"""FastAPI Model Serving Application with Prometheus metrics and structured telemetry."""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from mlops_platform.inference.predict import ModelInferenceService
from mlops_platform.inference.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger

# ----------------------------------------------------
# Prometheus Instrumentation
# ----------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status_code"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to generate ML predictions",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICTION_CHURN_TOTAL = Counter(
    "prediction_churn_total",
    "Total customer churn predictions by outcome class",
    ["churn_class"],
)
MODEL_LOADED_GAUGE = Gauge(
    "model_loaded_status",
    "Whether the champion ML model is loaded and ready (1 = yes, 0 = no)",
)


# ----------------------------------------------------
# Lifespan Application Manager
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context: Load ML model on startup, clean up on shutdown."""
    logger.info("Initializing FastAPI inference service...")
    try:
        service = ModelInferenceService.get_instance()
        if service.model is not None:
            MODEL_LOADED_GAUGE.set(1)
            logger.info("Champion model successfully loaded on startup.")
        else:
            MODEL_LOADED_GAUGE.set(0)
            logger.warning(
                "Model could not be loaded on startup. Run training pipeline to generate champion model."
            )
    except Exception as e:
        MODEL_LOADED_GAUGE.set(0)
        logger.error("Error during model startup loading: %s", e)

    yield

    logger.info("Shutting down FastAPI inference service.")


# ----------------------------------------------------
# FastAPI Application Configuration
# ----------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-grade MLOps model serving service with automated lifecycle tracking and drift monitoring.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
config = load_config()
allowed_origins = config.get("security", {}).get("allowed_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if isinstance(allowed_origins, list) else [allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Request Tracing & Telemetry Middleware
# ----------------------------------------------------
@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """Inject Request-ID, measure request latency, and record Prometheus request metrics."""
    req_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    request.state.request_id = req_id

    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        # Metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()

        return response
    except Exception as exc:
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="500",
        ).inc()
        logger.error("Unhandled request error [ID: %s]: %s", req_id, exc)
        raise exc


# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health & Status"],
    summary="Liveness probe",
)
def health_check() -> HealthResponse:
    """Return API service liveness status."""
    return HealthResponse(
        status="healthy",
        timestamp_utc=datetime.now(UTC).isoformat(),
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["Health & Status"],
    summary="Readiness probe",
)
def readiness_check() -> ReadyResponse:
    """Return whether model is loaded in memory and successfully serves test predictions."""
    service = ModelInferenceService.get_instance()
    is_ready = service.is_ready()

    if not is_ready:
        return ReadyResponse(
            status="not_ready",
            model_loaded=False,
            model_version=service.model_version if service.model is not None else None,
            message="Model is not loaded or failed test inference. Retrain or register champion model.",
        )

    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_version=service.model_version,
        message="Model is ready to serve predictions.",
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model Info"],
    summary="Get current champion model metadata",
)
def get_model_info() -> ModelInfoResponse:
    """Return details regarding currently loaded champion model."""
    service = ModelInferenceService.get_instance()
    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training pipeline has been executed.",
        )

    return ModelInfoResponse(
        model_name=service.model_name,
        model_version=service.model_version,
        run_id=service.run_id,
        alias=service.alias,
        expected_features=service.feature_columns,
        status="loaded",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict churn for a single customer",
)
def predict_single(
    features: CustomerFeatures,
    request: Request,
) -> PredictionResponse:
    """Predict churn probability and risk tier for a single customer record."""
    service = ModelInferenceService.get_instance()
    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training pipeline has been executed.",
        )

    req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:8]}")

    start_time = time.perf_counter()
    with PREDICTION_LATENCY.labels(endpoint="/predict").time():
        results = service.predict_features([features], request_id=req_id)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    result = results[0]
    PREDICTION_CHURN_TOTAL.labels(churn_class=str(result.churn_prediction)).inc()

    return PredictionResponse(
        success=True,
        result=result,
        model_name=service.model_name,
        model_version=service.model_version,
        request_id=req_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Predict churn for a batch of customers",
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Predict churn probability and risk tier for multiple customers in batch."""
    service = ModelInferenceService.get_instance()
    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training pipeline has been executed.",
        )

    req_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:8]}")

    start_time = time.perf_counter()
    with PREDICTION_LATENCY.labels(endpoint="/predict/batch").time():
        results = service.predict_features(payload.records, request_id=req_id)
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    for res in results:
        PREDICTION_CHURN_TOTAL.labels(churn_class=str(res.churn_prediction)).inc()

    return BatchPredictionResponse(
        success=True,
        results=results,
        record_count=len(results),
        model_name=service.model_name,
        model_version=service.model_version,
        request_id=req_id,
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Scrape Prometheus application metrics",
)
def metrics() -> Response:
    """Expose Prometheus application metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
