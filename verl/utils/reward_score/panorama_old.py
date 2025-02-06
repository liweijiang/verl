from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from functools import partial
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize_responses(responses):
    """
    Tokenize all responses once to avoid repeated tokenization.

    Args:
        responses (list): List of response strings

    Returns:
        list: List of tokenized responses
    """
    return [word_tokenize(response.lower().strip()) for response in responses]


def calculate_batch_bleu(args):
    """
    Calculate BLEU scores for a batch of responses.

    Args:
        args (tuple): Contains (start_idx, end_idx, tokenized_responses)

    Returns:
        list: BLEU scores for the batch
    """
    start_idx, end_idx, tokenized_responses = args
    weights = (0.25, 0.25, 0.25, 0.25)
    chencherry = SmoothingFunction()
    scores = []
    
    for idx in range(start_idx, end_idx):
        candidate = tokenized_responses[idx]
        references = [resp for i, resp in enumerate(tokenized_responses) if i != idx]
        try:
            score = sentence_bleu(references, candidate, weights=weights, 
                                smoothing_function=chencherry.method1)
            scores.append(score)
        except Exception as e:
            print(f"Error calculating BLEU score for response {idx}: {e}")
            scores.append(0.0)
            
    return scores


def get_self_bleu_score(responses, num_workers=None):
    """
    Calculate the self-BLEU score of a list of responses using parallel processing.

    Args:
        responses (list): List of response strings
        num_workers (int, optional): Number of worker processes. Defaults to CPU count.

    Returns:
        float: Average self-BLEU score
    """
    if len(responses) <= 1:
        return 0.0

    # Remove empty responses and strip whitespace
    responses = [r.strip() for r in responses if r.strip()]
    if not responses:
        return 0.0

    # Tokenize all responses once
    tokenized_responses = tokenize_responses(responses)
    total_responses = len(tokenized_responses)

    # Set up multiprocessing
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Ensure num_workers doesn't exceed number of responses
    num_workers = min(num_workers, total_responses)

    try:
        # Split work into batches
        batch_size = max(1, total_responses // num_workers)
        batches = []
        for i in range(0, total_responses, batch_size):
            end_idx = min(i + batch_size, total_responses)
            batches.append((i, end_idx, tokenized_responses))

        # Process batches in parallel
        with mp.Pool(num_workers) as pool:
            batch_results = pool.map(calculate_batch_bleu, batches)

        # Flatten results
        bleu_scores = [score for batch in batch_results for score in batch]
        return np.mean(bleu_scores)

    except Exception as e:
        print(f"Error in parallel processing: {e}")
        # Fallback to sequential processing with batching
        bleu_scores = calculate_batch_bleu((0, total_responses, tokenized_responses))
        return np.mean(bleu_scores)


def get_pairwise_sentence_embedding_similarity(responses, is_return_matrix=False):
    """
    Calculate the pairwise sentence embedding similarity of a list of responses using a single GPU.

    Args:
        responses (list): List of text responses
        is_return_matrix (bool): Whether to return the full similarity matrix

    Returns:
        float or numpy.ndarray: Average similarity or full similarity matrix
    """
    if len(responses) <= 1:
        return 0.0 if not is_return_matrix else np.zeros((1, 1))

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        device = torch.device("cuda:0")  # Use first GPU

        # Initialize model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model = model.to(device)

        # Get embeddings
        embeddings = model.encode(
            responses,
            convert_to_tensor=True,
            device=device
        )

        # Move to CPU for numpy calculations
        embeddings = embeddings.cpu().numpy()

        # Calculate pairwise cosine similarity
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = np.dot(
            embeddings, embeddings.T) / np.outer(norms, norms)

        # Handle numerical instabilities
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

        if is_return_matrix:
            return similarity_matrix
        else:
            # Exclude self-similarity from mean calculation
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            return np.mean(similarity_matrix[mask])

    except Exception as e:
        print(f"Error in GPU similarity calculation: {str(e)}")
        # Fallback to CPU
        try:
            print("Falling back to CPU processing...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(responses, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()

            norms = np.linalg.norm(embeddings, axis=1)
            similarity_matrix = np.dot(
                embeddings, embeddings.T) / np.outer(norms, norms)
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

            if is_return_matrix:
                return similarity_matrix
            else:
                mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                return np.mean(similarity_matrix[mask])

        except Exception as e:
            print(f"CPU fallback also failed: {str(e)}")
            return 0.0 if not is_return_matrix else np.zeros((len(responses), len(responses)))


def get_diversity_scores_per_prompt(responses, score_types=['self_bleu_score', 'pairwise_sentence_embedding_similarity']):
    """
    Calculate the diversity score of a list of responses.
    """
    scores = {}
    if 'self_bleu_score' in score_types:
        scores['self_bleu_score'] = get_self_bleu_score(responses)
    if 'pairwise_sentence_embedding_similarity' in score_types:
        scores['pairwise_sentence_embedding_similarity'] = get_pairwise_sentence_embedding_similarity(
            responses)
    return scores


def compute_score(grouped_records, is_average_across_prompts=True, score_types=['self_bleu_score', 'pairwise_sentence_embedding_similarity'], is_tqdm=True):
    all_prompt_diversity_scores = {score_type: []
                                   for score_type in score_types}
    for prompt in tqdm(grouped_records) if is_tqdm else grouped_records:
        responses = grouped_records[prompt]
        diversity_scores = get_diversity_scores_per_prompt(
            responses, score_types)
        for score_type in score_types:
            all_prompt_diversity_scores[score_type].extend(
                [diversity_scores[score_type]] * len(responses))
    if is_average_across_prompts:
        for score_type in score_types:
            all_prompt_diversity_scores[score_type] = np.mean(
                all_prompt_diversity_scores[score_type])
        return all_prompt_diversity_scores
    else:
        return all_prompt_diversity_scores


if __name__ == '__main__':
    validation_data = {
        "Name three animals that symbolize resilience and briefly explain why.": [
            "Here are three animals that symbolize resilience and the reasons why: 1. **Tortoise**: The tortoise is known for its ability to withstand harsh environments and survive for long periods without food or water. It also has a remarkable capacity to recover from injuries, making it a symbol of resilience. Its slow and steady approach to life is often seen as a model for perseverance and determination. 2. **Phoenix**: The phoenix is a mythical bird that is said to be born from its own ashes. When it dies, it is reborn, making it a symbol of transformation and resilience. The phoenix's ability to rise from the dead is seen as a powerful metaphor for overcoming adversity and finding strength in the face of challenge. 3. **Wolverine**: The wolverine is a small, fierce animal that is known for its tenacity and ability to adapt to harsh environments. It is also highly resilient, with the ability to survive in extreme cold and snow. Its determination and persistence in the face of adversity make it a symbol of resilience and a testament to the power of the human spirit. These animals embody the qualities of resilience, determination, and transformation, making them powerful symbols of hope and perseverance.",
            "Here are three animals that symbolize resilience: 1. **Tortoise**: The tortoise is known for its slow and steady approach to life, but it's also incredibly resilient in the face of adversity. Its shell provides protection from predators, and it can survive for long periods without food or water. The tortoise's ability to persevere in challenging conditions has made it a symbol of endurance and resilience. 2. **Falcon**: The falcon is a bird of prey that is renowned for its remarkable adaptability and resilience. Despite facing many challenges, such as harsh weather conditions, competition for food, and predators, the falcon remains a powerful and efficient hunter. Its ability to thrive in a wide range of environments has earned it the nickname \"king of the skies.\" 3. **Hedgehog**: The hedgehog is often overlooked, but it's a remarkable example of resilience. Despite being small and vulnerable, hedgehogs can survive in a variety of environments, from forests to urban areas. Their ability to curl up into a ball for defense and their spiky quills make them difficult to harm, allowing them to thrive in the face of adversity.",
            "Here are three animals that symbolize resilience: 1. **Cheetah**: Despite being one of the fastest land animals, cheetahs are also one of the most vulnerable species due to habitat loss, poaching, and human-wildlife conflict. However, they continue to thrive and adapt in the wild, demonstrating remarkable resilience in the face of adversity. 2. **Salmon**: Salmon are known for their remarkable migratory journey from the ocean to their birthplace in the river, a journey that can be up to 1,000 miles long. This incredible feat requires immense physical and mental resilience, as they face obstacles such as treacherous waters, predators, and habitat destruction along the way. 3. **Tortoise**: Tortoises are notorious for their slow and steady approach to life, but this comes with a remarkable level of resilience. Despite being small and vulnerable, they have survived for millions of years on our planet, facing threats such as habitat destruction, predators, and climate change. Their ability to adapt and thrive in a changing world makes them an icon of resilience."
        ]
    }

    for i in range(1024 * 5):
        validation_data["Name three animals that symbolize resilience and briefly explain why." + str(
            i)] = validation_data["Name three animals that symbolize resilience and briefly explain why."]

    print(compute_score(validation_data, is_average_across_prompts=False,
                        score_types=['self_bleu_score', 'pairwise_sentence_embedding_similarity']))
