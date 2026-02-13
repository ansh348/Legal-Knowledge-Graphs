import sys
sys.path.insert(0, '.')
from eval_retrieval_v2 import *
from pathlib import Path
import numpy as np

print('Loading graphs...')
graphs = iter_graphs(Path('iltur_graphs'))
case_ids = [cid for cid, _ in graphs]
graph_list = [g for _, g in graphs]

print('Loading queries...')
queries, query_texts, _, _ = build_annotation_qrels(graph_list, case_ids, 50)

print('Loading raw texts...')
raw_texts = load_raw_iltur_texts(case_ids)
doc_texts = [raw_texts.get(cid, 'empty') for cid in case_ids]

print('Building qrels...')
qrels_b, qrels_g, diag = build_regex_qrels(queries, raw_texts, case_ids)
valid = [i for i in range(len(queries)) if 3 <= len(qrels_b[i]) <= int(len(case_ids)*0.4)]
queries = [queries[i] for i in valid]
query_texts = [query_texts[i] for i in valid]
qrels_b = [qrels_b[i] for i in valid]
qrels_g = [qrels_g[i] for i in valid]

print(f'{len(queries)} queries ready. Running ColBERTv2...')
scores = run_colbert(doc_texts, query_texts)
pq, mm = evaluate_method(scores, qrels_b, qrels_g)
print(f'ColBERTv2: nDCG@10={mm["nDCG@10"]:.3f}  MAP={mm["MAP"]:.3f}  P@10={mm["P@10"]:.3f}')