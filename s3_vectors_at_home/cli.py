import argparse
import json
import sys

from vectorstore.core import get_store


def cmd_upload(args):
    try:
        get_store().add_document(args.doc_id, args.text)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Uploaded {args.doc_id}")


def cmd_bulk_upload(args):
    with open(args.file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    assert isinstance(docs, list), "Bulk file should be a JSON list of {doc_id, text}"
    store = get_store()
    for doc in docs:
        try:
            store.add_document(doc["doc_id"], doc["text"])
        except ValueError as e:
            print(f"Error uploading {doc.get('doc_id')!r}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Uploaded {doc['doc_id']}")
    print(f"Bulk uploaded {len(docs)} documents.")


def cmd_search(args):
    try:
        results = get_store().search(args.query, args.top_k)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Matches:", results)


def cmd_get(args):
    doc = get_store().get_document(args.doc_id)
    if doc:
        print(json.dumps(doc, indent=2))
    else:
        print("Not found.", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    docs = get_store().list_documents(skip=args.skip, limit=args.limit)
    for d in docs:
        print(d["doc_id"], "-", d["text_preview"].replace("\n", " ")[:60])


def cmd_delete(args):
    ok = get_store().delete_document(args.doc_id)
    if ok:
        print(f"Deleted {args.doc_id}")
    else:
        print("Not found or error.", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    store = get_store()
    print(f"Documents: {store.count_documents()}")
    print(f"Embedding model: {store.embedding_model_name()}")


def main():
    ap = argparse.ArgumentParser(description="S3 Vectors at Home CLI")
    subp = ap.add_subparsers(dest="cmd", required=True)

    up = subp.add_parser("upload", help="Upload a single doc")
    up.add_argument("--doc-id", required=True)
    up.add_argument("--text", required=True)
    up.set_defaults(func=cmd_upload)

    bup = subp.add_parser("bulk-upload", help="Bulk upload from JSON file")
    bup.add_argument("--file", required=True, help="JSON list of {doc_id, text}")
    bup.set_defaults(func=cmd_bulk_upload)

    s = subp.add_parser("search", help="Search for docs by text query")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    g = subp.add_parser("get", help="Get a document by id")
    g.add_argument("--doc-id", required=True)
    g.set_defaults(func=cmd_get)

    ls = subp.add_parser("list", help="List documents")
    ls.add_argument("--skip", type=int, default=0)
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    d = subp.add_parser("delete", help="Delete a document by id")
    d.add_argument("--doc-id", required=True)
    d.set_defaults(func=cmd_delete)

    st = subp.add_parser("status", help="Show status and stats")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
