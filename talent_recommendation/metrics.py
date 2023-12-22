from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch

def compute_auc(pos_scores, neg_scores):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from positive and negative scores.

    :param pos_scores: Tensor of scores for the positive class.
    :type pos_scores: torch.Tensor
    :param neg_scores: Tensor of scores for the negative class.
    :type neg_scores: torch.Tensor
    :return: The ROC AUC score.
    :rtype: float

    Example:
        >>> pos_scores = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_scores = torch.tensor([0.3, 0.2, 0.4])
        >>> auc = compute_auc(pos_scores, neg_scores)
        >>> print(f"ROC AUC Score: {auc}")
    """
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    scores = torch.cat([pos_scores, neg_scores])
    return roc_auc_score(labels.detach().cpu(), scores.detach().cpu())


def compute_precision_recall_f1(pos_scores, neg_scores, threshold=0.5):
    """
    Compute precision, recall, and F1-score based on positive and negative scores, using a specified threshold.

    :param pos_scores: Tensor of scores for the positive class.
    :type pos_scores: torch.Tensor
    :param neg_scores: Tensor of scores for the negative class.
    :type neg_scores: torch.Tensor
    :param threshold: The threshold for classifying scores into positive or negative, defaults to 0.5.
    :type threshold: float, optional
    :return: The precision, recall, and F1-score.
    :rtype: (float, float, float)

    Example:
        >>> pos_scores = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_scores = torch.tensor([0.3, 0.2, 0.4])
        >>> precision, recall, f1 = compute_precision_recall_f1(pos_scores, neg_scores)
        >>> print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    """
    scores = torch.cat([pos_scores, neg_scores])
    predictions = (scores > threshold).float()
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    
    precision = precision_score(labels.detach().cpu(), predictions.detach().cpu())
    recall = recall_score(labels.detach().cpu(), predictions.detach().cpu())
    f1 = f1_score(labels.detach().cpu(), predictions.detach().cpu())

    return precision, recall, f1


def hit_at_k(pos_scores, neg_scores, k=10):
    """
    Compute the hit rate at k, which is the proportion of positive scores in the top-k combined scores.

    :param pos_scores: Tensor of scores for the positive class.
    :type pos_scores: torch.Tensor
    :param neg_scores: Tensor of scores for the negative class.
    :type neg_scores: torch.Tensor
    :param k: The number of top scores to consider for calculating the hit rate, defaults to 10.
    :type k: int, optional
    :return: The hit rate at k.
    :rtype: float

    Example:
        >>> pos_scores = torch.tensor([0.8, 0.9, 0.7])
        >>> neg_scores = torch.tensor([0.3, 0.2, 0.4, 0.5, 0.6])
        >>> hit_rate = hit_at_k(pos_scores, neg_scores, k=3)
        >>> print(f"Hit Rate at 3: {hit_rate}")
    """
    # Combine scores and sort them
    combined_scores = torch.cat([pos_scores, neg_scores])
    _, indices = combined_scores.topk(k)

    # Calculate hits
    hits = (indices < pos_scores.size(0)).float().sum().item()
    return hits / k