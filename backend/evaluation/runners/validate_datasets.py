import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
def main():
    expected = {
        "ingestion.jsonl":100, "chunking.jsonl":100, "retrieval.jsonl":100,
        "generation.jsonl":100, "adaptive_retrieval.jsonl":100, "end_to_end.jsonl":100,
        "reranking.jsonl":100, "query_expansion.jsonl":100, "citation_validation.jsonl":100,
        "presidio_pii.jsonl":100, "prompt_injection.jsonl":100, "access_control.jsonl":100
    }
    for name, minimum in expected.items():
        data=rows(ROOT/"datasets"/name)
        assert len(data) >= minimum, f"{name}: expected at least {minimum}, got {len(data)}"
        ids=[r["id"] for r in data]
        assert len(ids)==len(set(ids)), f"{name}: duplicate IDs"
        print(f"PASS {name}: {len(data)} cases")
    corpus=json.loads((ROOT/"corpus"/"synthetic_corpus.json").read_text(encoding="utf-8"))
    sources={d["source_id"] for d in corpus}
    passages={p["passage_id"] for d in corpus for p in d["sections"]}
    for name in ["retrieval.jsonl","generation.jsonl","adaptive_retrieval.jsonl","end_to_end.jsonl"]:
        for r in rows(ROOT/"datasets"/name):
            assert set(r.get("relevant_source_ids",[])) <= sources, r["id"]
            assert set(r.get("relevant_passage_ids",[])) <= passages, r["id"]
    print("PASS: all QA evidence references resolve to the synthetic corpus.")
    print("NOTE: structural validation does not measure pipeline accuracy or security.")
if __name__=="__main__": main()
