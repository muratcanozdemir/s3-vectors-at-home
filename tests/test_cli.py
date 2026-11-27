import json
import subprocess
import uuid


def run_cli(args):
    # Use subprocess so that sys.argv is handled properly
    result = subprocess.run(
        ["uv", "run", "s3-vectors-cli"] + args,
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

    out = run_cli(["search", "--query", text, "--top-k", "1", "--scores"])
    score, listed_id = out.splitlines()[-1].split()
    assert listed_id == doc_id and float(score) > 0.99

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
    # The old assertion ("doc_id" in out or out == "") only held for an
    # empty bucket: list prints "<id> - <preview>", never the literal
    # "doc_id". Assert on a doc we own instead.
    doc_id = unique_id()
    run_cli(["upload", "--doc-id", doc_id, "--text", "listed doc"])
    try:
        out = run_cli(["list", "--limit", "1000"])
        assert f"{doc_id} - listed doc" in out
    finally:
        run_cli(["delete", "--doc-id", doc_id])
    out = run_cli(["status"])
    assert "Documents:" in out
    assert "Embedding model:" in out


def test_cli_named_index_metadata_and_filter():
    name = f"cli-{uuid.uuid4().hex[:8]}"
    assert f"Created index {name}" in run_cli(["indexes", "create", name])
    try:
        assert name in run_cli(["indexes", "list"]).splitlines()
        run_cli(["upload", "--index", name, "--doc-id", "a", "--text", "cats and dogs", "--metadata", '{"kind": "pets"}'])
        run_cli(["upload", "--index", name, "--doc-id", "b", "--text", "cats and lions", "--metadata", '{"kind": "wild"}'])
        out = run_cli(["search", "--index", name, "--query", "cats", "--filter", '{"kind": "wild"}'])
        assert out == "Matches: ['b']"
        assert '"kind": "pets"' in run_cli(["get", "--index", name, "--doc-id", "a"])
        assert "Documents: 2" in run_cli(["status", "--index", name])
    finally:
        run_cli(["indexes", "delete", name])
    result = subprocess.run(
        ["uv", "run", "s3-vectors-cli", "list", "--index", name], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1 and "not found" in result.stderr
