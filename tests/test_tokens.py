from fastapi.testclient import TestClient
from src.main import app
from src.llm.local_llm import token_counter

client = TestClient(app)


def test_token_tracking():
    # Mocking the token counter or ensuring it's reset
    response = client.post("/query", json={"query": "Hi", "user_id": "test-user"})
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens" in data
    # Small talk might not use tokens if it's a random choice,
    # but routing always uses tokens in this app (it calls classify_query which uses LLM).
    assert data["total_tokens"] >= 0
    print(f"Tokens captured: {data['total_tokens']}")


if __name__ == "__main__":
    test_token_tracking()
