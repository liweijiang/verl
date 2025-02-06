import sacrebleu
import multiprocessing
from functools import partial
import numpy as np


def compute_bleu_for_one(candidate_idx, texts):
    """
    Computes BLEU score for one sentence against all others in its group.
    Args:
        candidate_idx: Index of the candidate sentence within its group.
        texts: List of text strings in the group.
    Returns:
        BLEU score for the candidate sentence.
    """
    candidate = texts[candidate_idx]
    references = texts[:candidate_idx] + texts[candidate_idx+1:]

    bleu_score = sacrebleu.sentence_bleu(
        candidate,
        references,
        smooth_method="exp",
        use_effective_order=True
    ).score
    return bleu_score


def compute_group_self_bleu(group_texts):
    """
    Computes Self-BLEU scores for a single group of texts.
    Args:
        group_texts: List of text strings.
    Returns:
        (scores, average): Tuple of individual scores and their average
    """
    group_size = len(group_texts)
    scores = [compute_bleu_for_one(i, group_texts)
              for i in range(group_size)]
    return scores, np.mean(scores)


def compute_all_groups_self_bleu_parallel(all_texts, group_size=5, num_workers=4):
    """
    Computes Self-BLEU scores for all groups in parallel.
    Args:
        all_texts: List of strings.
        group_size: Number of texts in each group.
        num_workers: Number of parallel processes.
    Returns:
        group_scores: List of (scores, average) tuples for each group.
    """
    # Validate input
    if len(all_texts) % group_size != 0:
        raise ValueError(
            f"Length of all_texts ({len(all_texts)}) must be divisible by group_size ({group_size})")

    # Split texts into groups
    groups = [all_texts[i:i+group_size]
              for i in range(0, len(all_texts), group_size)]

    # Create pool and compute scores
    with multiprocessing.Pool(num_workers) as pool:
        group_scores = pool.map(
            partial(compute_group_self_bleu),
            groups
        )
    return group_scores


def compute_score_all_types(all_texts, group_size, num_workers=4, is_average_across_prompts=True, score_types=['self_bleu_score']):
    # , 'pairwise_sentence_embedding_similarity'
    all_prompt_diversity_scores = {score_type: []
                                   for score_type in score_types}

    if 'self_bleu_score' in score_types:
        group_scores = compute_all_groups_self_bleu_parallel(
            all_texts,
            group_size=group_size,
            num_workers=num_workers
        )           
        for group_idx, (scores, avg) in enumerate(group_scores):
            all_prompt_diversity_scores['self_bleu_score'].extend(scores)

    if is_average_across_prompts:
        for score_type in score_types:
            all_prompt_diversity_scores[score_type] = np.mean(
                all_prompt_diversity_scores[score_type])
        return all_prompt_diversity_scores
    else:
        return all_prompt_diversity_scores


def compute_score(all_texts, group_size, score_type, num_workers=4, is_average_across_prompts=True):
    all_scores = []
    if score_type == 'self_bleu_score':
        group_scores = compute_all_groups_self_bleu_parallel(
            all_texts,
            group_size=group_size,
            num_workers=num_workers
        )           
        for group_idx, (scores, avg) in enumerate(group_scores):
            all_scores.extend(scores)

    if is_average_across_prompts:
        return np.mean(all_scores)
    else:
        return all_scores


# Example usage:
if __name__ == '__main__':
    # Example list of texts (multiple groups)
    all_texts = [
        # Group 1
        "The quick brown fox jumps over the lazy dog.",
        "A fast, brown fox leaps across a sleepy canine.",
        "The rapid fox swiftly jumps over the sluggish dog.",
        "An energetic fox hops over a dozing dog.",
        "A brown fox jumped over a lazy sleeping dog.",
        # Group 2
        "The cat sits on the windowsill watching birds.",
        "A feline perches near the window observing avians.",
        "The kitty is stationed at the window eyeing birds.",
        "A cat lounges by the windowsill looking at birds.",
        "The feline rests near the window watching flying creatures.",
    ] * (512)

    group_scores = compute_all_groups_self_bleu_parallel(
        all_texts,
        group_size=5,  # Now configurable
        num_workers=4
    )

    # # Print results
    # for group_idx, (scores, avg) in enumerate(group_scores):
    #     print(f"\nGroup {group_idx + 1}:")
    #     for i, score in enumerate(scores):
    #         print(f"Self-BLEU for sentence {i+1}: {score:.2f}")
    #     print(f"Group average Self-BLEU: {avg:.2f}")

    print(compute_score(all_texts, group_size=5, num_workers=4, is_average_across_prompts=False))
    