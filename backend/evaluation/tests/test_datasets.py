import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    with (ROOT/"datasets"/name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
def test_every_dataset_has_at_least_100_cases():
    names=[
        "ingestion.jsonl","chunking.jsonl","retrieval.jsonl","generation.jsonl",
        "adaptive_retrieval.jsonl","end_to_end.jsonl","reranking.jsonl",
        "query_expansion.jsonl","citation_validation.jsonl","presidio_pii.jsonl",
        "prompt_injection.jsonl","access_control.jsonl"
    ]
    for name in names:
        assert len(load(name)) >= 100, name
def test_each_qa_stage_has_100_unique_ids():
    for name in ["retrieval.jsonl","generation.jsonl","adaptive_retrieval.jsonl","end_to_end.jsonl"]:
        rows=load(name)
        assert len(rows)==100
        assert len({r["id"] for r in rows})==100
def test_references_resolve_to_corpus():
    corpus=json.loads((ROOT/"corpus"/"synthetic_corpus.json").read_text(encoding="utf-8"))
    sources={d["source_id"] for d in corpus}
    passages={p["passage_id"] for d in corpus for p in d["sections"]}
    for name in ["retrieval.jsonl","generation.jsonl","adaptive_retrieval.jsonl","end_to_end.jsonl"]:
        for r in load(name):
            assert set(r.get("relevant_source_ids",[])) <= sources
            assert set(r.get("relevant_passage_ids",[])) <= passages
