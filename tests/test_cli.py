import subprocess
import uuid
import json

def run_cli(args):
    # Use subprocess so that sys.argv is handled properly
    result = subprocess.run(
        ["uv", "run", "s3_vectors_at_home.cli"] + args,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def unique_id():
    return f"cli-{uuid.uuid4()}"

def test_cli_upload_and_search(tmp_path):
    doc_id = unique_id()
    text = "CLI integration test"
    out = run_cli(["upload", "--doc-id", doc_id, "--text", text])
    assert f"Uploaded {doc_id}" in out

    out = run_cli(["search", "--query", text, "--top-k", "1"])
    assert doc_id in out

    out = run_cli(["get", "--doc-id", doc_id])
    assert doc_id in out and text in out

    run_cli(["delete", "--doc-id", doc_id])

def test_cli_bulk_upload(tmp_path):
    docs = [
        {"doc_id": unique_id(), "text": "bulk doc 1"},
        {"doc_id": unique_id(), "text": "bulk doc 2"}
    ]
    json_path = tmp_path / "bulk.json"
    json_path.write_text(json.dumps(docs))
    out = run_cli(["bulk-upload", "--file", str(json_path)])
    assert "Bulk uploaded 2 documents." in out
    # Clean up
    for d in docs:
        run_cli(["delete", "--doc-id", d["doc_id"]])

def test_cli_list_and_status():
    # Just ensure these don't error out
    out = run_cli(["list"])
    assert "doc_id" in out or out == ""
    out = run_cli(["status"])
    assert "Documents:" in out
    assert "Embedding model:" in out
