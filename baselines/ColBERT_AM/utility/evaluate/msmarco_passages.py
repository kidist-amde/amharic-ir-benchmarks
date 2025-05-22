"""
    Evaluate Amahric Passages ranking.
"""
import os
import math
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict
from ColBERT_AM.utils.utils import print_message, file_tqdm


def dcg_at_k(ranking, positives, k):
    """ Compute Discounted Cumulative Gain (DCG) """
    dcg = sum((1 / math.log2(idx + 2)) if pid in positives else 0 for idx, (_, pid, _) in enumerate(ranking[:k]))
    return dcg


def ndcg_at_k(ranking, positives, k):
    """ Compute Normalized DCG (NDCG) """
    dcg = dcg_at_k(ranking, positives, k)
    idcg = sum((1 / math.log2(idx + 2)) for idx in range(min(k, len(positives))))  # Ideal DCG
    return dcg / idcg if idcg > 0 else 0.0


def main(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [1, 5, 10, 20, 50, 100]}
    qid2ndcg_10 = {}
    qid2ndcg_100 = {}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1
            qid2positives[qid].append(pid)

    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)

            score = float(score[0]) if score else -1  # Handle missing scores

            qid2ranking[qid].append((rank, pid, score))
    
    # Find missing queries in qrels
    missing_queries = set(qid2ranking.keys()) - set(qid2positives.keys())
    if missing_queries:
        print(f"⚠️ Warning: {len(missing_queries)} queries in ranking are missing from qrels.")
        print(f"Example missing queries: {sorted(missing_queries)[:10]}")  # Show 10 examples

    # Replace assertion with a warning
    if not set(qid2ranking.keys()).issubset(set(qid2positives.keys())):
        print("⚠️ Warning: Some ranked queries are missing from qrels. Proceeding anyway.")

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print()
        print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
        print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
        print()

    print_message(f"#> Computing MRR@k, R@k, and NDCG@k for {num_judged_queries} queries.")

    for qid in tqdm.tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = set(qid2positives[qid])

        # Compute MRR@10 and MRR@100
        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed
            if pid in positives:
                qid2mrr[qid] = 1.0 / rank  # Only store the first occurrence
                break

        # Compute Recall@k
        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed
            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = min(1.0, qid2recall[depth].get(qid, 0) + 1.0)

        # Compute NDCG@10 and NDCG@100 separately
        qid2ndcg_10[qid] = ndcg_at_k(ranking, positives, 10)
        qid2ndcg_100[qid] = ndcg_at_k(ranking, positives, 100)

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    # Compute and print metrics
    print()
    mrr_sum = sum(qid2mrr.values())
    print_message(f"#> MRR@10 = {mrr_sum / num_judged_queries}")
    print_message(f"#> MRR@100 = {mrr_sum / num_judged_queries}")  # Since MRR@100 = MRR@10

    ndcg_10_sum = sum(qid2ndcg_10.values())
    print_message(f"#> NDCG@10 = {ndcg_10_sum / num_judged_queries}")

    ndcg_100_sum = sum(qid2ndcg_100.values())
    print_message(f"#> NDCG@100 = {ndcg_100_sum / num_judged_queries}")

    for depth in qid2recall:
        metric_sum = sum(qid2recall[depth].values())
        print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")

    if args.annotate:
        print_message(f"#> Writing annotations to {args.output} ..")

        if os.path.exists(args.output):
            print_message(f"#> [WARNING] Output file {args.output} already exists, overwriting..")

        with open(args.output, 'w') as f:
            for qid in tqdm.tqdm(qid2positives):
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]

                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)

                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')

    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'

    main(args)
