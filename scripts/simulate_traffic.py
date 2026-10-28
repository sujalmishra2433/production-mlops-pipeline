"""Script to simulate inference API traffic and distribution drift."""

import argparse
import time

import httpx

from mlops_platform.inference.predict import ModelInferenceService
from mlops_platform.inference.schemas import CustomerFeatures
from scripts.generate_sample_data import generate_drifted_data, generate_synthetic_churn_data


def simulate_direct_traffic(n_samples: int = 50, drift: bool = False) -> None:
    """Simulate inference directly using ModelInferenceService."""
    print(f"Simulating {n_samples} direct inference requests (drift={drift})...")
    service = ModelInferenceService.get_instance()

    df = generate_drifted_data(n_samples=n_samples) if drift else generate_synthetic_churn_data(n_samples=n_samples)
    records = df.drop(columns=["churn"]).to_dict(orient="records")
    features_list = [CustomerFeatures(**rec) for rec in records]

    results = service.predict_features(features_list, request_id="traffic-sim")
    print(f"Successfully generated {len(results)} predictions logged to telemetry file.")


def simulate_http_traffic(url: str, n_samples: int = 50, drift: bool = False) -> None:
    """Simulate inference over HTTP against live FastAPI service."""
    print(f"Sending {n_samples} HTTP prediction requests to {url} (drift={drift})...")
    df = generate_drifted_data(n_samples=n_samples) if drift else generate_synthetic_churn_data(n_samples=n_samples)
    records = df.drop(columns=["churn"]).to_dict(orient="records")

    with httpx.Client(timeout=10.0) as client:
        # Check health first
        resp = client.get(f"{url}/health")
        if resp.status_code != 200:
            print(f"API is not healthy: {resp.text}")
            return

        successes = 0
        for i, rec in enumerate(records):
            r = client.post(f"{url}/predict", json=rec)
            if r.status_code == 200:
                successes += 1
            if (i + 1) % 10 == 0:
                print(f"  Sent {i + 1}/{n_samples} requests (Success: {successes})")
            time.sleep(0.02)  # Tiny spacing

    print(f"Finished HTTP simulation: {successes}/{n_samples} requests succeeded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate API inference traffic")
    parser.add_argument("--samples", type=int, default=50, help="Number of requests")
    parser.add_argument("--drift", action="store_true", help="Simulate drifted traffic")
    parser.add_argument("--api-url", type=str, default=None, help="FastAPI base URL (e.g. http://127.0.0.1:8000)")
    args = parser.parse_args()

    if args.api_url:
        simulate_http_traffic(url=args.api_url, n_samples=args.samples, drift=args.drift)
    else:
        simulate_direct_traffic(n_samples=args.samples, drift=args.drift)


if __name__ == "__main__":
    main()
