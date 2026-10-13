from tests.conftest import SOLAR, make_pdf


def upload(client, name, data, tags=""):
    return client.post("/documents", files={"file": (name, data)}, data={"tags": tags})


def test_health_and_config_hide_secrets(client):
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/config").json()
    assert cfg["openai_api_key"] is False  # boolean, never the key itself


def test_upload_list_search_delete_flow(client):
    r = upload(client, "solar.txt", SOLAR.encode(), "energy, solar")
    assert r.status_code == 201
    doc = r.json()
    assert doc["num_chunks"] >= 1 and doc["tags"] == ["energy", "solar"]

    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc["doc_id"]]

    s = client.post("/search", json={"query": "rooftop solar payback period", "k": 2}).json()
    assert not s["insufficient_evidence"]
    assert s["citations"][0]["filename"] == "solar.txt" and s["citations"][0]["label"] == "S1"

    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 204
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 404
    assert client.post("/search", json={"query": "solar"}).json()["insufficient_evidence"] is True


def test_pdf_upload(client):
    r = upload(client, "r.pdf", make_pdf(["Hydrogen storage is expensive."]))
    assert r.status_code == 201 and r.json()["num_pages"] == 1


def test_invalid_uploads_return_422(client):
    assert upload(client, "x.exe", b"abc").status_code == 422
    assert upload(client, "x.pdf", b"fake").status_code == 422
    assert upload(client, "x.txt", b"").status_code == 422


def test_search_validation(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422
    assert client.post("/search", json={"query": "   "}).status_code == 422
