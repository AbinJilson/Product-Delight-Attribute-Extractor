import json
import os
import json
import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm
from collections import Counter
import csv
import concurrent.futures
import logging

load_dotenv() # Load environment variables from .env file

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

JUNK_RESPONSES = [
    "i'm sorry",
    "no review text",
    "please provide the customer review"
]

PREFERRED_ATTRIBUTES = [
    'Fragrance', 'Longevity', 'Effectiveness', 'Texture', 'Packaging','Quality','Climate Suitability',
    'Ingredients', 'Value for Money', 'Skin Compatibility', 'Freshness', 'Odor Control',
    'Moisturizing', 'Non-allergenic'
]

def load_reviews(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads reviews from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A list of reviews, where each review is a dictionary.
        Returns an empty list if the file is not found or is invalid.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        print(f"Successfully loaded {len(reviews)} reviews from {filepath}")
        return reviews
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def extract_attributes_with_openai(review_body: str) -> List[str]:
    """
    Uses a sophisticated single-prompt strategy to extract product attributes.
    - Identifies specific attributes, preferring a canonical list.
    - Discovers and creates new attributes if necessary.
    - Falls back to 'General Satisfaction' for generic praise.
    - Ignores non-product aspects (e.g., customer service, shipping).
    """
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in the .env file.")
        return []

    try:
        prompt = (
            f"Analyze the customer review below to identify praised product attributes. Follow these steps:\n"
            f"1. **Focus strictly on the product itself.** Ignore comments about shipping, customer service, or the company.\n"
            f"2. **Identify specific positive qualities.** For example, 'smells great' or 'lasts all day'.\n"
            f"3. **Map to Preferred Attributes.** If possible, match the quality to one of these: {', '.join(PREFERRED_ATTRIBUTES)}.\n"
            f"4. **Create New Attributes if Needed.** If a specific quality isn't on the list, create a new, concise attribute name (e.g., 'Non-staining', 'Climate Suitability').\n"
            f"5. **Handle Generic Praise.** If and ONLY IF no specific attributes are mentioned, but the sentiment is positive (e.g., 'good product', 'love it'), return the single attribute 'General Satisfaction'.\n"
            f"6. **Return a comma-separated list.** If no positive attributes are found at all, return 'NONE'.\n\n"
            f"Review: \"{review_body}\"\n"
            f"Praised Attributes:"
        )

        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing customer reviews for product attributes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=60
        )

        attributes_text = response.choices[0].message.content.strip()

        # Robustly filter out junk responses and 'NONE'
        lower_attributes_text = attributes_text.lower()
        if lower_attributes_text == 'none' or any(junk in lower_attributes_text for junk in JUNK_RESPONSES):
            return []

        return [attr.strip() for attr in attributes_text.split(',') if attr.strip()]
    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return []

def get_embeddings(texts: List[str], model="text-embedding-3-large") -> np.ndarray:
    """
    Generates embeddings for a list of texts using OpenAI's API.
    """
    if not texts:
        return np.array([])
    try:
        response = openai.embeddings.create(input=texts, model=model)
        return np.array([item.embedding for item in response.data])
    except Exception as e:
        print(f"An error occurred while getting embeddings: {e}")
        return np.array([])

def cluster_attributes(
    unique_attributes: List[str], 
    attribute_frequencies: Counter, 
    embeddings: np.ndarray, 
    threshold=0.85
) -> Dict[str, str]:
    """
    Clusters attributes and selects the most frequent term as the canonical name for each cluster.
    """
    if embeddings.size == 0:
        return {}

    similarity_matrix = cosine_similarity(embeddings)
    n_attributes = len(unique_attributes)
    visited = [False] * n_attributes
    canonical_map = {}

    for i in range(n_attributes):
        if not visited[i]:
            # Start a new cluster
            cluster_indices = [i]
            # Use a queue for breadth-first search to find all connected components (clusters)
            queue = [i]
            visited[i] = True
            
            head = 0
            while head < len(queue):
                current_index = queue[head]
                head += 1
                
                for j in range(n_attributes):
                    if not visited[j] and similarity_matrix[current_index, j] > threshold:
                        visited[j] = True
                        cluster_indices.append(j)
                        queue.append(j)

            # Determine the canonical name for the found cluster
            cluster_attributes = [unique_attributes[k] for k in cluster_indices]
            
            # Find the most frequent attribute in the cluster to be the canonical name
            canonical_name = max(cluster_attributes, key=lambda attr: attribute_frequencies[attr])
            
            # Map all attributes in the cluster to the canonical name
            for attr in cluster_attributes:
                canonical_map[attr] = canonical_name
                
    return canonical_map

def write_json_output(reviews_with_attributes: List[Dict[str, Any]], filepath: str):
    """
    Writes the list of reviews, each with its extracted delight attributes, to a JSON file.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(reviews_with_attributes, f, indent=4)
        print(f"Enhanced review data saved to {filepath}")
    except Exception as e:
        print(f"Error writing JSON output to {filepath}: {e}")

def write_csv_output(ranked_attributes: List[tuple[str, int]], filepath: str):
    """
    Writes the ranked list of delight attributes and their frequencies to a CSV file.
    """
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Delight Attribute", "Frequency"])
            writer.writerows(ranked_attributes)
        print(f"Ranked attributes saved to {filepath}")
    except Exception as e:
        print(f"Error writing CSV output to {filepath}: {e}")

def load_evaluation_data(filepath: str) -> List[Dict[str, Any]]:
    """Loads evaluation data from a CSV file."""
    eval_data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_text = row.get('body')
                attributes_str = row.get('delight_attribute')
                if review_text and attributes_str:
                    expected_attributes = [attr.strip() for attr in attributes_str.split(',') if attr.strip()]
                    if expected_attributes:
                        eval_data.append({
                            'review_text': review_text,
                            'expected_attributes': expected_attributes
                        })
        logging.info(f"Successfully loaded {len(eval_data)} samples from evaluation file {filepath}")
        return eval_data
    except FileNotFoundError:
        logging.error(f"Evaluation file not found: {filepath}")
        return []
    except Exception as e:
        logging.error(f"Error reading evaluation file {filepath}: {e}")
        return []

def run_evaluation(evaluation_filepath: str):
    """Runs the evaluation process against a ground-truth dataset."""
    logging.info(f"--- Starting Evaluation --- ")
    eval_samples = load_evaluation_data(evaluation_filepath)
    if not eval_samples:
        return

    logging.info(f"Extracting attributes for {len(eval_samples)} evaluation samples...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sample = {executor.submit(extract_attributes_with_openai, sample['review_text']): sample for sample in eval_samples}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(eval_samples), desc="Evaluating"):
            sample = future_to_sample[future]
            try:
                sample['extracted_attributes'] = future.result()
            except Exception as exc:
                logging.error(f"An exception occurred while processing review: {sample['review_text'][:50]}... - {exc}")
                sample['extracted_attributes'] = []

    logging.info("Building canonical map for evaluation...")
    all_human_attributes = [attr for sample in eval_samples for attr in sample['expected_attributes']]
    all_machine_attributes = [attr for sample in eval_samples for attr in sample.get('extracted_attributes', [])]
    
    all_attributes_for_clustering = list(set(all_human_attributes + all_machine_attributes))
    attribute_frequencies = Counter(all_human_attributes + all_machine_attributes)
    
    embeddings = get_embeddings(all_attributes_for_clustering)
    raw_to_canonical_map = cluster_attributes(all_attributes_for_clustering, attribute_frequencies, embeddings)

    total_tp, total_fp, total_fn = 0, 0, 0
    success_examples, fp_examples, fn_examples = [], [], []

    for sample in eval_samples:
        expected_canonical = {raw_to_canonical_map.get(attr, attr) for attr in sample['expected_attributes']}
        extracted_canonical = {raw_to_canonical_map.get(attr, attr) for attr in sample.get('extracted_attributes', [])}
        
        tp_set = expected_canonical.intersection(extracted_canonical)
        fp_set = extracted_canonical - expected_canonical
        fn_set = expected_canonical - extracted_canonical

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)

        # Store examples for qualitative analysis
        if tp_set and len(success_examples) < 3:
            success_examples.append({
                "review": sample['review_text'], "expected": list(expected_canonical), 
                "extracted": list(extracted_canonical), "match": list(tp_set)
            })
        if fp_set and len(fp_examples) < 3:
            fp_examples.append({
                "review": sample['review_text'], "expected": list(expected_canonical),
                "extracted": list(extracted_canonical), "false_positives": list(fp_set)
            })
        if fn_set and len(fn_examples) < 3:
            fn_examples.append({
                "review": sample['review_text'], "expected": list(expected_canonical),
                "extracted": list(extracted_canonical), "missed": list(fn_set)
            })

    logging.info("--- Evaluation Results ---")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logging.info(f"Total True Positives:  {total_tp}")
    logging.info(f"Total False Positives: {total_fp}")
    logging.info(f"Total False Negatives: {total_fn}")
    logging.info(f"Precision: {precision:.2%}")
    logging.info(f"Recall:    {recall:.2%}")
    logging.info(f"F1-Score:  {f1_score:.2%}")

    logging.info("\n--- Analysis of Successes and Failures ---")
    if success_examples:
        logging.info("\n[SUCCESS EXAMPLES (Correct Matches)]")
        for ex in success_examples:
            logging.info(f"  Review:    '{ex['review'][:80]}...'")
            logging.info(f"  Expected:  {ex['expected']}")
            logging.info(f"  Extracted: {ex['extracted']}")
            logging.info(f"  MATCH:     {ex['match']}\n")

    if fp_examples:
        logging.info("\n[FAILURE EXAMPLES (Incorrect Extractions)]")
        for ex in fp_examples:
            logging.info(f"  Review:    '{ex['review'][:80]}...'")
            logging.info(f"  Expected:  {ex['expected']}")
            logging.info(f"  Extracted: {ex['extracted']}")
            logging.info(f"  EXTRACTED BUT NOT EXPECTED: {ex['false_positives']}\n")

    if fn_examples:
        logging.info("\n[FAILURE EXAMPLES (Missed Attributes)]")
        for ex in fn_examples:
            logging.info(f"  Review:    '{ex['review'][:80]}...'")
            logging.info(f"  Expected:  {ex['expected']}")
            logging.info(f"  Extracted: {ex['extracted']}")
            logging.info(f"  MISSED (Expected but not extracted): {ex['missed']}\n")

def main(reviews_filepath: str, top_n: int, json_output_path: str, csv_output_path: str):
    reviews = load_reviews(reviews_filepath)
    if not reviews:
        return

    print(f"Extracting attributes from {len(reviews)} reviews (this may take a while)...")
    reviews_with_raw_attributes = []
    failed_reviews = []

    # Use a ThreadPoolExecutor to process reviews in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future for each review body that is not empty
        future_to_review = {executor.submit(extract_attributes_with_openai, review.get('body')):
                            review for review in reviews if review.get('body') and isinstance(review.get('body'), str)}

        for future in tqdm(concurrent.futures.as_completed(future_to_review), total=len(future_to_review), desc="Extracting Attributes"):
            review = future_to_review[future]
            try:
                attributes = future.result()
                if attributes:
                    reviews_with_raw_attributes.append({'review_obj': review, 'raw_attributes': attributes})
                else:
                    failed_reviews.append(review.get('body'))
            except Exception as exc:
                print(f'{review.get("body")} generated an exception: {exc}')
                failed_reviews.append(review.get('body'))

    all_raw_attributes = [attr for item in reviews_with_raw_attributes for attr in item['raw_attributes']]

    if not all_raw_attributes:
        print("No attributes could be extracted. Exiting.")
        return

    print(f"\nFound {len(set(all_raw_attributes))} unique raw attributes.")
    
    attribute_frequencies = Counter(all_raw_attributes)
    unique_attributes = list(attribute_frequencies.keys())

    print("Generating embeddings for unique attributes...")
    embeddings = get_embeddings(unique_attributes)

    print("Clustering attributes to deduplicate...")
    raw_to_canonical_map = cluster_attributes(unique_attributes, attribute_frequencies, embeddings)

    reviews_with_final_attributes = []
    all_final_attributes = []
    for item in reviews_with_raw_attributes:
        review_obj = item['review_obj']
        canonical_attributes = sorted(list(set([raw_to_canonical_map.get(attr, attr) for attr in item['raw_attributes']])))
        review_obj['delight_attributes'] = canonical_attributes
        reviews_with_final_attributes.append(review_obj)
        all_final_attributes.extend(canonical_attributes)

    final_ranking = Counter(all_final_attributes)

    print("\n--- Generating Final Output Files ---")

    write_json_output(reviews_with_final_attributes, "reviews_with_attributes.json")
    write_csv_output(final_ranking.most_common(top_n), "delight_attributes_ranking.csv")

    failed_reviews_log = "failed_reviews.txt"
    with open(failed_reviews_log, "w", encoding="utf-8") as f:
        f.write("--- Reviews with No Extracted Attributes ---\n")
        for review_text in failed_reviews:
            f.write(f"{review_text}\n---\n")
    print(f"Log of reviews with no extracted attributes saved to {failed_reviews_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and rank product delight attributes from customer reviews.")
    parser.add_argument("reviews_file", nargs='?', default="reviews.json", help="Path to the input JSON file with reviews. Not required for evaluation.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top attributes to display.")
    parser.add_argument("--output-json", type=str, default="reviews_with_attributes.json", help="Path for the output JSON file.")
    parser.add_argument("--output-csv", type=str, default="delight_attributes_ranking.csv", help="Path for the output CSV file.")
    parser.add_argument("--evaluate", type=str, metavar="EVAL_FILE", help="Run in evaluation mode using the provided CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.evaluate:
        run_evaluation(args.evaluate)
    else:
        main(args.reviews_file, args.top_n, args.output_json, args.output_csv)
