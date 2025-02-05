import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from functools import partial



def tokenize_responses(responses):
    """
    Tokenize all responses once to avoid repeated tokenization.
    
    Args:
        responses (list): List of response strings
        
    Returns:
        list: List of tokenized responses
    """
    return [word_tokenize(response.lower().strip()) for response in responses]

def calculate_single_bleu(idx, tokenized_responses):
    """
    Calculate BLEU score for a single response against all others.
    
    Args:
        idx (int): Index of the candidate response
        tokenized_responses (list): List of tokenized responses
        
    Returns:
        float: BLEU score for the candidate
    """
    candidate = tokenized_responses[idx]
    references = [resp for i, resp in enumerate(tokenized_responses) if i != idx]
    
    # Use weights for different n-grams (1-gram to 4-gram)
    weights = (0.25, 0.25, 0.25, 0.25)
    
    try:
        return sentence_bleu(references, candidate, weights=weights)
    except Exception as e:
        print(f"Error calculating BLEU score for response {idx}: {e}")
        return 0.0

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
    
    # Set up multiprocessing
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Ensure num_workers doesn't exceed number of responses
    num_workers = min(num_workers, len(responses))
    
    try:
        # Create process pool and calculate BLEU scores in parallel
        with mp.Pool(num_workers) as pool:
            # Use partial to fix tokenized_responses argument
            calc_bleu = partial(calculate_single_bleu, tokenized_responses=tokenized_responses)
            # Calculate BLEU scores for all responses in parallel
            bleu_scores = pool.map(calc_bleu, range(len(responses)))
            
        # Calculate and return average BLEU score
        return np.mean(bleu_scores)
    
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        # Fallback to sequential processing
        bleu_scores = [calculate_single_bleu(i, tokenized_responses) 
                      for i in range(len(responses))]
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
        similarity_matrix = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)

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
            similarity_matrix = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

            if is_return_matrix:
                return similarity_matrix
            else:
                mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                return np.mean(similarity_matrix[mask])

        except Exception as e:
            print(f"CPU fallback also failed: {str(e)}")
            return 0.0 if not is_return_matrix else np.zeros((len(responses), len(responses)))


def get_diversity_scores_per_prompt(responses):
    """
    Calculate the diversity score of a list of responses.
    """
    self_bleu_score = get_self_bleu_score(responses)
    pairwise_sentence_embedding_similarity = get_pairwise_sentence_embedding_similarity(
        responses)
    return {
        'self_bleu_score': self_bleu_score,
        'pairwise_sentence_embedding_similarity': pairwise_sentence_embedding_similarity
    }


def compute_score(validation_data, is_average_across_prompts=True):
    all_prompt_diversity_scores = []
    for prompt in validation_data:
        responses = validation_data[prompt]
        diversity_scores = get_diversity_scores_per_prompt(responses)
        all_prompt_diversity_scores.append(diversity_scores)
    if is_average_across_prompts:
        for prompt_scores in all_prompt_diversity_scores:
            for key in prompt_scores:
                print(key, prompt_scores[key])
        
    # else:
    #     return all_prompt_diversity_scores


if __name__ == '__main__':
    validation_data = {
        "Name three animals that symbolize resilience and briefly explain why.": [
            "Here are three animals that symbolize resilience and the reasons why: 1. **Tortoise**: The tortoise is known for its ability to withstand harsh environments and survive for long periods without food or water. It also has a remarkable capacity to recover from injuries, making it a symbol of resilience. Its slow and steady approach to life is often seen as a model for perseverance and determination. 2. **Phoenix**: The phoenix is a mythical bird that is said to be born from its own ashes. When it dies, it is reborn, making it a symbol of transformation and resilience. The phoenix's ability to rise from the dead is seen as a powerful metaphor for overcoming adversity and finding strength in the face of challenge. 3. **Wolverine**: The wolverine is a small, fierce animal that is known for its tenacity and ability to adapt to harsh environments. It is also highly resilient, with the ability to survive in extreme cold and snow. Its determination and persistence in the face of adversity make it a symbol of resilience and a testament to the power of the human spirit. These animals embody the qualities of resilience, determination, and transformation, making them powerful symbols of hope and perseverance.",
            "Here are three animals that symbolize resilience: 1. **Tortoise**: The tortoise is known for its slow and steady approach to life, but it's also incredibly resilient in the face of adversity. Its shell provides protection from predators, and it can survive for long periods without food or water. The tortoise's ability to persevere in challenging conditions has made it a symbol of endurance and resilience. 2. **Falcon**: The falcon is a bird of prey that is renowned for its remarkable adaptability and resilience. Despite facing many challenges, such as harsh weather conditions, competition for food, and predators, the falcon remains a powerful and efficient hunter. Its ability to thrive in a wide range of environments has earned it the nickname \"king of the skies.\" 3. **Hedgehog**: The hedgehog is often overlooked, but it's a remarkable example of resilience. Despite being small and vulnerable, hedgehogs can survive in a variety of environments, from forests to urban areas. Their ability to curl up into a ball for defense and their spiky quills make them difficult to harm, allowing them to thrive in the face of adversity.",
            "Here are three animals that symbolize resilience: 1. **Cheetah**: Despite being one of the fastest land animals, cheetahs are also one of the most vulnerable species due to habitat loss, poaching, and human-wildlife conflict. However, they continue to thrive and adapt in the wild, demonstrating remarkable resilience in the face of adversity. 2. **Salmon**: Salmon are known for their remarkable migratory journey from the ocean to their birthplace in the river, a journey that can be up to 1,000 miles long. This incredible feat requires immense physical and mental resilience, as they face obstacles such as treacherous waters, predators, and habitat destruction along the way. 3. **Tortoise**: Tortoises are notorious for their slow and steady approach to life, but this comes with a remarkable level of resilience. Despite being small and vulnerable, they have survived for millions of years on our planet, facing threats such as habitat destruction, predators, and climate change. Their ability to adapt and thrive in a changing world makes them an icon of resilience."
        ]
    }

    print(compute_score(validation_data, is_average_across_prompts=True))




