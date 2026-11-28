import argparse
import json
import os
import sys

from minio.error import S3Error

from vectorstore import backup
from vectorstore.backup import BackupError
from vectorstore.core import (
    BUCKET,
    DEFAULT_INDEX,
    IndexAlreadyExistsError,
    IndexCorruptError,
    IndexNotFoundError,
    IndexWriteConflictError,
    client_from_env,
    get_store,
)


def _json_arg(value: str, what: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"--{what} is not valid JSON: {e}") from None


def cmd_upload(args):
    metadata = _json_arg(args.metadata, "metadata") if args.metadata else None
    get_store().add_document(args.doc_id, args.text, metadata, index=args.index)
    print(f"Uploaded {args.doc_id}")


def cmd_bulk_upload(args):
    with open(args.file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not all(
        isinstance(d, dict) and "doc_id" in d and "text" in d for d in docs
    ):
        raise ValueError("bulk file must be a JSON list of {doc_id, text[, metadata]} objects")
    try:
        count = get_store().add_documents(((d["doc_id"], d["text"], d.get("metadata")) for d in docs), index=args.index)
    except ValueError as e:
        raise ValueError(f"{e} (nothing was uploaded)") from None
    print(f"Bulk uploaded {count} documents.")


def cmd_search(args):
    flt = _json_arg(args.filter, "filter") if args.filter else None
    store = get_store()
    if args.passages:
        for r in store.search_with_scores(args.query, args.top_k, index=args.index, filter=flt, passages=True, exact=args.exact):
            print(f"{r['score']:.4f}  {r['doc_id']}  [{r['span'][0]}:{r['span'][1]}]")
            print("    " + " ".join(r["passage"].split()))
    elif args.scores:
        for r in store.search_with_scores(args.query, args.top_k, index=args.index, filter=flt, exact=args.exact):
            print(f"{r['score']:.4f}  {r['doc_id']}")
    else:
        print("Matches:", store.search(args.query, args.top_k, index=args.index, filter=flt, exact=args.exact))


def cmd_get(args):
    doc = get_store().get_document(args.doc_id, index=args.index)
    if doc:
        print(json.dumps(doc, indent=2))
    else:
        print("Not found.", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    docs = get_store().list_documents(skip=args.skip, limit=args.limit, index=args.index)
    for d in docs:
        print(d["doc_id"], "-", d["text_preview"].replace("\n", " ")[:60])


def cmd_delete(args):
    if get_store().delete_document(args.doc_id, index=args.index):
        print(f"Deleted {args.doc_id}")
    else:
        print("Not found.", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    store = get_store()
    print(f"Index: {args.index}")
    stats = store.index_stats(index=args.index)
    print(f"Documents: {stats['documents']} ({stats['vectors']} vectors)")
    if stats["search"] == "approximate":
        print(f"Search: approximate (graph {'ready' if stats['graph_ready'] else 'building; searching exactly meanwhile'})")
    print(f"Segments: {stats['segments']} ({stats['bytes']:,} bytes)")
    print(f"Embedding model: {store.embedding_model_name()}")


def cmd_indexes_list(_args):
    for name in get_store().list_indexes():
        print(name)


def cmd_indexes_create(args):
    search = None
    if args.approximate:
        search = {"mode": "approximate", "m": args.m, "ef_construction": args.ef_construction, "ef_search": args.ef_search}
    elif any(v is not None for v in (args.m, args.ef_construction, args.ef_search)):
        raise ValueError("--m, --ef-construction and --ef-search need --approximate")
    get_store().create_index(args.name, search=search)
    print(f"Created index {args.name}")


def cmd_indexes_compact(args):
    stats = get_store().compact(index=args.name)
    print(f"Compacted index {args.name}: {stats['documents']} documents in {stats['segments']} segment(s), {stats['bytes']:,} bytes")


def cmd_indexes_delete(args):
    get_store().delete_index(args.name)
    print(f"Deleted index {args.name}")


def cmd_backup(args):
    catalog = backup.backup(client_from_env(), BUCKET, args.dir)
    total = sum(f["bytes"] for f in catalog["files"].values())
    print(f"Backed up {len(catalog['indexes'])} index(es), {total:,} bytes, to {args.dir}")


def cmd_restore(args):
    names = backup.restore(
        client_from_env(), BUCKET, args.dir, only=args.index, replace=args.replace, dry_run=args.dry_run
    )
    print(f"{'Would restore' if args.dry_run else 'Restored'} {len(names)} index(es)")


def main():
    # A CLI run is over in seconds: building an approximate index's graph
    # would only make it wait at exit. Its searches are exact instead.
    os.environ.setdefault("S3VAH_GRAPH_BUILD", "off")
    ap = argparse.ArgumentParser(description="S3 Vectors at Home CLI")
    # Every document command takes --index, after the subcommand name.
    index_opt = argparse.ArgumentParser(add_help=False)
    index_opt.add_argument("--index", default=DEFAULT_INDEX, help=f"index to use (default: {DEFAULT_INDEX})")
    subp = ap.add_subparsers(dest="cmd", required=True)

    up = subp.add_parser("upload", parents=[index_opt], help="Upload a single doc")
    up.add_argument("--doc-id", required=True)
    up.add_argument("--text", required=True)
    up.add_argument("--metadata", help='JSON object, e.g. \'{"lang": "en", "year": 2024}\'')
    up.set_defaults(func=cmd_upload)

    bup = subp.add_parser("bulk-upload", parents=[index_opt], help="Bulk upload from JSON file")
    bup.add_argument("--file", required=True, help="JSON list of {doc_id, text[, metadata]}")
    bup.set_defaults(func=cmd_bulk_upload)

    s = subp.add_parser("search", parents=[index_opt], help="Search for docs by text query")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--filter", help='metadata filter as JSON, e.g. \'{"lang": "en"}\'')
    s.add_argument("--scores", action="store_true", help="Print one 'score  doc_id' line per match")
    s.add_argument("--passages", action="store_true", help="Also print each match's best-matching passage")
    s.add_argument("--exact", action="store_true", help="On an approximate index, compare against every vector")
    s.set_defaults(func=cmd_search)

    g = subp.add_parser("get", parents=[index_opt], help="Get a document by id")
    g.add_argument("--doc-id", required=True)
    g.set_defaults(func=cmd_get)

    ls = subp.add_parser("list", parents=[index_opt], help="List documents")
    ls.add_argument("--skip", type=int, default=0)
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    d = subp.add_parser("delete", parents=[index_opt], help="Delete a document by id")
    d.add_argument("--doc-id", required=True)
    d.set_defaults(func=cmd_delete)

    st = subp.add_parser("status", parents=[index_opt], help="Show status and stats")
    st.set_defaults(func=cmd_status)

    ix = subp.add_parser("indexes", help="List, create, delete or compact indexes")
    ixs = ix.add_subparsers(dest="indexes_cmd", required=True)
    ixs.add_parser("list", help="List indexes").set_defaults(func=cmd_indexes_list)
    c = ixs.add_parser("create", help="Create an index")
    c.add_argument("name")
    c.add_argument("--approximate", action="store_true", help="Search with an HNSW graph: much faster on large indexes, not always exact")
    c.add_argument("--m", type=int, help="HNSW links per vector (4-128, default 32)")
    c.add_argument("--ef-construction", type=int, help="HNSW build-time candidate list (8-1024, default 64)")
    c.add_argument("--ef-search", type=int, help="HNSW search-time candidate list (8-4096, default 64)")
    c.set_defaults(func=cmd_indexes_create)
    dl = ixs.add_parser("delete", help="Delete an index and all its documents")
    dl.add_argument("name")
    dl.set_defaults(func=cmd_indexes_delete)
    cp = ixs.add_parser("compact", help="Merge an index's segments and reclaim space from overwritten/deleted docs")
    cp.add_argument("name", nargs="?", default=DEFAULT_INDEX)
    cp.set_defaults(func=cmd_indexes_compact)

    bk = subp.add_parser("backup", help="Back up every index (text API and S3 Vectors) to a new local directory")
    bk.add_argument("dir", help="directory to create (or an empty one)")
    bk.set_defaults(func=cmd_backup)
    rs = subp.add_parser("restore", help="Restore indexes from a backup directory")
    rs.add_argument("dir", help="a directory made by 'backup'")
    rs.add_argument("--index", action="append", metavar="NAME",
                    help="restore only this index (repeatable); S3 Vectors indexes are BUCKET/INDEX")
    rs.add_argument("--replace", action="store_true", help="overwrite indexes that exist already")
    rs.add_argument("--dry-run", action="store_true", help="check the backup and report what would be restored")
    rs.set_defaults(func=cmd_restore)

    args = ap.parse_args()
    try:
        args.func(args)
    except (
        ValueError, IndexNotFoundError, IndexAlreadyExistsError, IndexWriteConflictError, IndexCorruptError, BackupError
    ) as e:
        msg = {
            IndexNotFoundError: f"index {e.args[0]!r} not found",
            IndexAlreadyExistsError: f"index {e.args[0]!r} already exists",
        }.get(type(e), str(e))
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:  # a missing file, or the object store unreachable
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except S3Error as e:
        print(f"Error: object store refused the request: {e.code}: {e.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
