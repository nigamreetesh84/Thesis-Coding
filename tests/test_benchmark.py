from __future__ import annotations

import json
import math
import time

from fastapi.testclient import TestClient

from config.settings import settings
from src.api import app


def test_first_request_and_throughput_meet_targets(monkeypatch) -> None:
    monkeypatch.setattr(settings, "use_stub_llm", True)
    payload = {
        "annual_income": 800_000,
        "age": 27,
        "cibil": 760,
        "employment": "salaried",
        "existing_relationship": False,
        "banks": ["SBI", "HDFC"],
    }
    endpoint_latencies: list[float] = []
    request_durations: list[float] = []

    with TestClient(app) as client:
        for request_number in range(21):
            started = time.perf_counter()
            response = client.post(
                "/recommend",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            request_durations.append(time.perf_counter() - started)
            assert response.status_code == 200
            result = response.json()
            endpoint_latencies.append(result["latency_ms"])
            assert result["decisions"]
            if request_number == 0:
                first_request_latency = result["latency_ms"]

    percentile_index = math.ceil(0.95 * len(endpoint_latencies)) - 1
    p95_ms = sorted(endpoint_latencies)[percentile_index]
    throughput = len(request_durations) / sum(request_durations)
    print(
        f"first_request_ms={first_request_latency:.3f} "
        f"p95_ms={p95_ms:.3f} throughput_req_s={throughput:.2f}"
    )

    assert first_request_latency <= 750
    assert p95_ms <= 750
    assert throughput > 5