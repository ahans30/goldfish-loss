import torch
from torch import nn, Tensor
from typing import Iterable, Dict
# from ..SentenceTransformer import SentenceTransformer
from lit_gpt.model import GPT


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(nn.Module):
    """
    This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
    where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

    This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
    as it will sample in each batch n-1 negative docs randomly.

    The performance usually increases with increasing batch sizes.

    For more information, see: https://arxiv.org/pdf/1705.00652.pdf
    (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

    You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
    (a_1, p_1, n_1), (a_2, p_2, n_2)

    Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

    Example::

        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('distilbert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
            InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(self, model: GPT, scale: float = 20.0, similarity_fct=cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: list[Tensor, Tensor], loss_type: str):
        embeddings_a_bsz_T_d = sentence_features[0]
        embeddings_b_bsz_T_d = sentence_features[1]
        if loss_type == "batch_negative":
            embeddings_a_bsz_d = embeddings_a_bsz_T_d.reshape(-1, embeddings_a_bsz_T_d.size(-1))
            embeddings_b_bsz_d = embeddings_b_bsz_T_d.reshape(-1, embeddings_b_bsz_T_d.size(-1))
            scores = self.similarity_fct(embeddings_a_bsz_d, embeddings_b_bsz_d) * self.scale # [b, b]
            labels = torch.tensor(
                    range(len(scores)), dtype=torch.long, device=scores.device
                    )  # Example a[i] should match with b[i]    [0, 1, 2, 3, ...]
            accuracy = (torch.argmax(scores, dim=1) == labels).float().mean()   # we want to check the retrieval accuracy
            return self.cross_entropy_loss(scores, labels), accuracy.clone().detach()
        else:
            # loss_type == "single_negative"
            loss_and_accuracies = [self.forward([embeddings_a_bsz_T_d[i], embeddings_b_bsz_T_d[i]], "batch_negative") for i in range(embeddings_a_bsz_T_d.size(0))]
            losses = [x[0] for x in loss_and_accuracies]
            accuracies = [x[1] for x in loss_and_accuracies]
            return torch.stack(losses).mean(), torch.stack(accuracies).mean().clone().detach()

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}