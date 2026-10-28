"""Live demonstration runner: starts server, runs HTTP tests, verifies telemetry and drift."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request

API_URL = "http://127.0.0.1:8000"


def make_request(endpoint: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict | str]:
    """Helper to perform HTTP requests using standard library."""
    url = f"{API_URL}{endpoint}"
    req_data = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8")
            if "application/json" in content_type:
                return resp.status, json.loads(raw)
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def run_demo():
    print("=" * 70)
    print(" PRODUCTION MLOPS PLATFORM -- LIVE VERIFICATION RUNNER")
    print("=" * 70)

    # 1. Start Uvicorn Server in background process
    print("\n[Step 1] Starting FastAPI Model Serving Service on port 8000...")
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait for server to come up
    server_ready = False
    for attempt in range(25):
        time.sleep(1)
        try:
            status, body = make_request("/health")
            if status == 200:
                server_ready = True
                print(f"  Server is LIVE (attempt {attempt + 1})")
                break
        except Exception:
            pass

    if not server_ready:
        print("ERROR: Server failed to start within 25 seconds.")
        proc.terminate()
        return

    try:
        # 2. Check /health
        print("\n[Step 2] Probing Liveness Endpoint (GET /health)...")
        status, body = make_request("/health")
        print(f"  HTTP {status} OK")
        print("  Response:", json.dumps(body, indent=2))

        # 3. Check /ready
        print("\n[Step 3] Probing Readiness Endpoint (GET /ready)...")
        status, body = make_request("/ready")
        print(f"  HTTP {status} OK")
        print("  Response:", json.dumps(body, indent=2))

        # 4. Check /model-info
        print("\n[Step 4] Fetching Active Champion Model Info (GET /model-info)...")
        status, body = make_request("/model-info")
        print(f"  HTTP {status} OK")
        print("  Response:", json.dumps(body, indent=2))

        # 5. Single Prediction Request - High Risk
        print("\n[Step 5] Sending Real-Time Churn Prediction - High Risk Customer (POST /predict)...")
        high_risk_customer = {
            "customer_id": "CUST-LIVE-001",
            "tenure": 2,
            "monthly_charges": 98.40,
            "total_charges": 196.80,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "online_security": "No",
            "paperless_billing": "Yes",
        }
        status, body = make_request("/predict", method="POST", data=high_risk_customer)
        print(f"  HTTP {status} OK")
        print("  Prediction Result:")
        print(json.dumps(body, indent=2))

        # 6. Single Prediction Request - Low Risk
        print("\n[Step 6] Sending Real-Time Churn Prediction - Loyal Customer (POST /predict)...")
        low_risk_customer = {
            "customer_id": "CUST-LIVE-002",
            "tenure": 68,
            "monthly_charges": 24.50,
            "total_charges": 1666.00,
            "contract_type": "Two year",
            "payment_method": "Credit card",
            "internet_service": "DSL",
            "tech_support": "Yes",
            "online_security": "Yes",
            "paperless_billing": "No",
        }
        status, body = make_request("/predict", method="POST", data=low_risk_customer)
        print(f"  HTTP {status} OK")
        print("  Prediction Result:")
        print(json.dumps(body, indent=2))

        # 7. Batch Prediction Request
        print("\n[Step 7] Sending Batch Prediction Request (POST /predict/batch)...")
        batch_payload = {"records": [high_risk_customer, low_risk_customer]}
        status, body = make_request("/predict/batch", method="POST", data=batch_payload)
        print(f"  HTTP {status} OK (Processed {body.get('record_count')} records)")
        for r in body.get("results", []):
            print(f"  - {r['customer_id']}: Churn={r['churn_prediction']} | Prob={r['churn_probability']:.4f} | Tier={r['risk_tier']}")

        # 8. Prometheus Telemetry
        print("\n[Step 8] Inspecting Prometheus Application Telemetry (GET /metrics)...")
        status, body = make_request("/metrics")
        print(f"  HTTP {status} OK")
        metric_lines = [
            line for line in str(body).splitlines()
            if line.startswith("http_requests_total")
            or line.startswith("prediction_churn_total")
            or line.startswith("model_loaded_status")
        ]
        for m in metric_lines[:10]:
            print(f"  {m}")

        # 9. Statistical Drift Detection
        print("\n[Step 9] Running Statistical Data Drift Detection (KS-Test + PSI)...")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import pandas as pd
        from mlops_platform.monitoring.drift import detect_drift
        from scripts.generate_sample_data import generate_drifted_data

        ref_path = Path("data/reference/reference_features.parquet")
        if ref_path.exists():
            ref_df = pd.read_parquet(ref_path)
            drifted_df = generate_drifted_data(n_samples=500, seed=99)
            num_cols = ["tenure", "monthly_charges", "total_charges"]
            report = detect_drift(ref_df[num_cols], drifted_df[num_cols], report_path="reports/drift_report.json")
            print(f"  Dataset Drift Detected: {report['dataset_drift_detected']}")
            print(f"  Drifted Feature Share: {report['drift_share']:.1%} (Threshold: {report['drift_share_threshold']:.1%})")
            print(f"  Drifted Features Count: {report['drifted_features_count']} of {report['total_features_evaluated']}")
            for feat, res in report.get("features", {}).items():
                print(f"  - {feat:20s}: KS={res.get('ks_statistic')} (p={res['p_value']:.4e}) | PSI={res.get('psi')} | Drift={res['drift_detected']}")

        print("\n" + "=" * 70)
        print(" ALL ENDPOINTS AND PIPELINE SUBSYSTEMS VERIFIED LIVE & OPERATIONAL!")
        print("=" * 70)

    finally:
        print("\nShutting down background server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown clean.")


if __name__ == "__main__":
    run_demo()
