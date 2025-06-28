# ✨ Product Delight Attribute Extractor ✨

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

This CLI tool analyzes customer reviews to identify and rank positively mentioned product attributes, which we call "delight attributes." It uses OpenAI's GPT-4 model for intelligent attribute extraction and semantic clustering to group similar concepts, providing actionable insights into what customers love about a product.

---

## 🚀 Features

- **Intelligent Attribute Extraction**: Leverages the power of OpenAI's GPT-4 to understand and extract specific product praises from unstructured review text.
- **Semantic Clustering**: Automatically groups semantically similar attributes (e.g., "smell" and "fragrance") using vector embeddings and clustering algorithms.
- **Ranked Outputs**: Produces a clean, ranked list of the most frequently mentioned delight attributes.
- **Structured Data Export**: Saves detailed results in both JSON and CSV formats for easy integration with other tools and analysis pipelines.
- **Parallel Processing**: Utilizes concurrent API calls for efficient processing of large datasets.
- **Robust Evaluation Mode**: Includes a built-in evaluation framework to measure extraction performance against a human-labeled dataset.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- An OpenAI API key

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbinJilson/Product-Delight-Attribute-Extractor.git
    cd Product-Delight-Attribute-Extractor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate it
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    - Create a file named `.env` in the project's root directory.
    - Add your OpenAI API key to this file:
      ```
      OPENAI_API_KEY="your_secret_api_key_here"
      ```

## ⚙️ Usage

The tool has two main modes: **extraction** and **evaluation**.

### 1. Extracting Attributes from Reviews

This is the default mode. It processes a JSON file of reviews and generates ranked attributes.

```bash
python main.py <path_to_reviews.json> --output-json <output.json> --output-csv <output.csv>
```

**Arguments:**
- `reviews_file` (required): Path to the input JSON file containing reviews.
- `--output-json` (optional): Path for the output JSON file. Defaults to `reviews_with_attributes.json`.
- `--output-csv` (optional): Path for the output CSV file. Defaults to `delight_attributes_ranking.csv`.

### 2. Evaluating Model Performance

This mode measures the tool's accuracy against a human-labeled CSV file and generates a detailed report.

```bash
python main.py --evaluate <path_to_evaluation.csv> --evaluation-report <report_path.txt>
```

This command produces a file (e.g., `evaluation_report.txt`) containing:
-   Overall metrics (Precision, Recall, F1-Score).
-   A qualitative analysis with examples of successful matches, incorrect extractions (false positives), and missed attributes (false negatives).

**Arguments:**
- `--evaluate` (required): Path to the evaluation CSV file. The file must contain `body` and `delight_attribute` columns.
- `--evaluation-report` (optional): Path for the output evaluation report file. Defaults to `evaluation_report.txt`.

## 🔬 Methodology

### Attribute Extraction

The core of the extraction process relies on **OpenAI's GPT-4.1 model**. We use a sophisticated, zero-shot prompt that instructs the model to:
1.  Identify and list all positively mentioned product-specific attributes.
2.  Invent a concise name for each attribute.
3.  Return "General Satisfaction" if the sentiment is positive but no specific attribute is mentioned.
4.  Explicitly ignore non-product feedback (e.g., shipping, customer service).
5.  Return "NONE" if no positive feedback is found.

> This single-prompt approach has proven more effective and efficient than older methods like separate classification and extraction steps.

### Semantic Clustering

To handle synonyms and varied phrasing (e.g., "smell" vs. "fragrance"), we perform semantic clustering:
1.  **Embedding**: All unique extracted attributes are converted into high-dimensional vectors using OpenAI's `text-embedding-3-large` model.
2.  **Similarity Calculation**: We compute the cosine similarity between all pairs of attribute embeddings.
3.  **Clustering**: We use **Agglomerative Clustering** to group attributes that are semantically close (i.e., have a high cosine similarity).
4.  **Canonicalization**: For each cluster, we select the most frequently occurring attribute in the original dataset as the canonical name for the entire group.

> This ensures that the final ranked list is clean, deduplicated, and accurately reflects the underlying customer sentiment.

## 📊 Evaluation Results

*(as of 2025-06-28, based on `delight-evaluation.csv`)*

- **Precision**: **57.89%**
- **Recall**: **64.71%**
- **F1-Score**: **61.11%**

### Analysis
- The model is effective at identifying and matching core, explicitly stated attributes like `Fragrance`, `Longevity`, and `Non-allergenic`.
- **False Positives** often occur when the model extracts a related but not identical concept (e.g., extracting `Freshness` alongside `Fragrance`) or defaults to `General Satisfaction` when a more specific attribute was expected.
- **False Negatives** (missed attributes) typically happen with more abstract or subtly implied concepts like `Compatibility` or `Product Quality`, where the review text is positive but doesn't use explicit keywords.

## 📝 Assumptions and Limitations

- **Language**: The tool assumes all reviews are in English.
- **Context**: The model is prompted to ignore non-product feedback, but its ability to do so depends entirely on the LLM's interpretation.
- **Ambiguity**: Very short or ambiguous reviews (e.g., "it's good") will likely be classified as `General Satisfaction`.
- **Clustering Threshold**: The effectiveness of semantic clustering is sensitive to the similarity threshold (currently `0.85`). This may need tuning for different datasets.

## 💡 Known Issues and Future Improvements

- **Over-eager `General Satisfaction`**: The model sometimes returns `General Satisfaction` even when a more specific attribute could be inferred. This could be addressed by further refining the system prompt.
- **Abstract Concepts**: The model struggles to capture abstract concepts that aren't tied to specific keywords. This is a known limitation of zero-shot LLM extraction.
- **Compound Attributes**: The tool handles compound attributes (e.g., "eco-friendly packaging") as single items, but their semantic clustering can be inconsistent.
- **Failed Extractions**: Reviews that are neutral, negative, or contain praise too subtle for the model may not yield any attributes. These are logged to `failed_reviews.txt` for manual inspection.

Potential improvements include fine-tuning the prompt, adjusting the clustering threshold, expanding the evaluation dataset, or experimenting with different embedding models.

---

## 🔮 Future Roadmap: A Fully Open-Source Vision

While the current implementation leverages the power of OpenAI's GPT models for rapid and accurate extraction, the long-term vision for this project is to become a **completely free, open-source, and self-hosted solution**. This can be achieved by replacing the reliance on the OpenAI API with a custom-trained Named Entity Recognition (NER) model.

### The Path Forward: Using spaCy

The most viable path is to use a powerful NLP library like [**spaCy**](https://spacy.io/) to train a custom model to recognize `DELIGHT_ATTRIBUTE` entities in review text.

**The development process would look like this:**

1.  **Data Annotation**: Create a high-quality, labeled dataset where specific product attributes in review sentences are annotated. This is the most critical and labor-intensive step.
2.  **Model Training**: Train a custom spaCy NER model on the annotated dataset. The model will learn the patterns, context, and phrasing associated with delight attributes specific to your domain.
3.  **Integration**: Replace the `extract_attributes_with_openai` function with a new function that loads the trained spaCy model and uses it to predict entities from the review text.
4.  **Open-Source Embeddings**: For clustering, replace OpenAI's embedding model with a high-performance open-source alternative from a library like `sentence-transformers`.

### Why Pursue This Vision?

-   **Zero Cost**: Eliminates all API costs, making the tool free to run at any scale.
-   **Full Control & Ownership**: You own the model and can fine-tune it precisely for your specific domain, without being subject to changes in external APIs.
-   **Speed and Efficiency**: A local, optimized NER model can be significantly faster than making network requests to an external API.
-   **Privacy**: All data is processed locally, ensuring complete privacy.

This roadmap transforms the tool from a powerful utility into a sustainable, independent, and highly specialized asset.
