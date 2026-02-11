import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_query_trace_small_talk():
    response = client.post(
        "/query", json={"query": "Hello", "config": {"temperature": 0.0}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "trace" in data
    assert len(data["trace"]) >= 2  # Input guardrail + Routing + Execution
    events = [step["event"] for step in data["trace"]]
    assert "INPUT_GUARDRAIL" in events
    assert "ROUTING" in events
    assert any("EXECUTION" in e for e in events)


def test_query_trace_structured_data():
    # This query should trigger the rule-based router for STRUCTURED_DATA
    response = client.post(
        "/query", json={"query": "What is the visa requirement for India to Singapore?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "trace" in data
    events = [step["event"] for step in data["trace"]]
    assert "ROUTING" in events
    assert "TOOL_EXECUTION" in events
    # Verify tool data is included in the trace step
    tool_step = next(s for s in data["trace"] if s["event"] == "TOOL_EXECUTION")
    assert tool_step["status"] == "success"
    assert "requires_visa" in tool_step["data"]


def test_query_trace_rag():
    # Requires the index to be built.
    # Since it's a unit test, we might not have it built, but retriever.load() should handle it in pipeline.
    response = client.post(
        "/query", json={"query": "What is the policy for business travel?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "trace" in data
    events = [step["event"] for step in data["trace"]]
    assert "ROUTING" in events
    if data["route"] == "FACT_FROM_DOCS":
        assert any("RAG_PIPELINE" in e for e in events)
        rag_step = next(s for s in data["trace"] if s["event"] == "RAG_PIPELINE")
        assert "grounded" in rag_step["data"]


def test_query_config_override():
    # Test temperature override (just checking it doesn't crash, we can't easily verify internal temp here)
    response = client.post(
        "/query",
        json={
            "query": "How are you?",
            "config": {"temperature": 0.5, "model": "llama3.1:8b"},
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
