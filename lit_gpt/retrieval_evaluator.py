# Yoinked with <3 from MTEB project and reworked slightly.
# https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/RetrievalEvaluator.py.

from typing import Dict, List, Set
import numpy as np


class RetrievalEvaluator:
    """
    This class evaluates a series of predictions for an Information Retrieval (IR) problem.
    It accepts a set of labels called qrels (query relevance judgements) and a set of predictions.
    It measures Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k and MAP@k for different k values.
    """

    def __init__(
        self,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        using_distances: bool = True,
    ):
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.using_distances = using_distances

    def compute_metrics(
        self, qrels: Dict[str, Set], predictions: Dict[str, List]
    ):
        """
        Compute metrics for a list of queries

        Args:
            qrels (`Dict[str, Set]`): Dictionary with query indices as keys and values the set of relevant document indices
            predictions (`Dict[str, List]`): Dictionary with query indices as keys and values the list of retrieved documents as dictionaries with keys "idx" and "score"

        Returns:
            `Dict[str, Dict[str, float]]`: Dictionary with keys "mrr@k", "ndcg@k", "accuracy@k", "precision_recall@k", "map@k"
                which values are dictionaries with scores for different k values
        """
        # Init score computation values
        num_hits_at_k = {"accuracy_at_" + str(k): 0 for k in self.accuracy_at_k}
        precisions_at_k = {
            "precision_at_" + str(k): [] for k in self.precision_recall_at_k
        }
        recall_at_k = {
            "recall_at_" + str(k): [] for k in self.precision_recall_at_k
        }
        MRR = {"mrr_at_" + str(k): 0 for k in self.mrr_at_k}
        ndcg = {"ndcg_at_" + str(k): [] for k in self.ndcg_at_k}
        AveP_at_k = {"map_at_" + str(k): [] for k in self.map_at_k}

        # Compute scores on results
        for query_idx in predictions.keys():
            # Sort scores
            top_hits = sorted(
                predictions[query_idx],
                key=lambda x: x["score"],
                reverse=(False if self.using_distances else True),
            )
            query_relevant_docs = qrels[query_idx]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["idx"] in query_relevant_docs:
                        num_hits_at_k["accuracy_at_" + str(k_val)] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["idx"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k["precision_at_" + str(k_val)].append(
                    num_correct / k_val
                )
                recall_at_k["recall_at_" + str(k_val)].append(
                    num_correct / len(query_relevant_docs)
                )

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["idx"] in query_relevant_docs:
                        MRR["mrr_at_" + str(k_val)] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["idx"] in query_relevant_docs else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(
                    predicted_relevance, k_val
                ) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg["ndcg_at_" + str(k_val)].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["idx"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(
                    k_val, len(query_relevant_docs)
                )
                AveP_at_k["map_at_" + str(k_val)].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(list(predictions.keys()))

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(list(predictions.keys()))

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            **num_hits_at_k,
            **precisions_at_k,
            **recall_at_k,
            **MRR,
            **ndcg,
            **AveP_at_k,
        }

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg