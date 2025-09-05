# %% [markdown]
# # Finetuning Embeddings For Domain Specific Tasks

# %% [markdown]
# ## Imports and Class Definitions
#

# %%
# !pip install -r requirements.txt

# %%
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from huggingface_hub import login
import umap.umap_ as umap # Import UMAP


# %%

from llama_index.core.llms import LLM, CompletionResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

# %%
class HuggingFacePhi3LLM(LLM):
    tokenizer: any = None
    model: any = None
    generator: any = None

    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct", device="cuda"):
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto"))
        object.__setattr__(self, 'generator', pipeline("text-generation", model=self.model, tokenizer=self.tokenizer))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self.generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        # Strip prompt from response if model echoes it
        answer = response.replace(prompt, "").strip()
        return CompletionResponse(text=answer)

    def achat(self, *args, **kwargs):
        raise NotImplementedError()

    def acomplete(self, *args, **kwargs):
        raise NotImplementedError()

    def astream_chat(self, *args, **kwargs):
        raise NotImplementedError()

    def astream_complete(self, *args, **kwargs):
        raise NotImplementedError()

    def chat(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def metadata(self):
        return {}

    def stream_chat(self, *args, **kwargs):
        raise NotImplementedError()

    def stream_complete(self, *args, **kwargs):
        raise NotImplementedError()

# %% [markdown]
# ## Creation of DataSets
#

# %%
def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


# %%

def generate_and_split_qa_pairs(pdf_files: list, temp_dataset_path="full_qa_dataset.json", test_size=0.2):
    print("Generating QA embedding pairs and splitting into train/test sets...")
    phi3_llm = HuggingFacePhi3LLM()

    # Load all nodes from specified PDF files
    all_nodes = []
    for file in pdf_files:
        all_nodes.extend(load_corpus([file]))

    # Generate a single large QA embedding pair dataset
    # We'll save it temporarily to disk as generate_qa_embedding_pairs works with output_path
    if os.path.exists(temp_dataset_path):
        print(f"Using existing QA dataset from {temp_dataset_path}")
    else:
        temp_dataset = generate_qa_embedding_pairs(
            llm=phi3_llm,
            nodes=all_nodes,
            output_path=temp_dataset_path
        )

    # Free up memory used by the Phi3 LLM as it's no longer needed
    del phi3_llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the generated dataset into a Hugging Face Dataset object
    full_dataset = EmbeddingQAFinetuneDataset.from_json(temp_dataset_path)

    # Filter out generic queries like "Document:", "Question 1:", "Question:"
    filtered_queries = {}
    filtered_relevant_docs = {}
    for q_id, query_text in full_dataset.queries.items():
        if query_text.strip() != "Document:" and \
           not query_text.strip().startswith("Question 1:") and \
           not query_text.strip().startswith("Question:"):
            filtered_queries[q_id] = query_text
            filtered_relevant_docs[q_id] = full_dataset.relevant_docs[q_id]

    # Convert to Hugging Face datasets.Dataset for splitting
    df = pd.DataFrame({
        "id": list(filtered_queries.keys()),
        "anchor": ["search_query: " + q for q in filtered_queries.values()], # Add prefix
        "positive": ["search_document: " + full_dataset.corpus[filtered_relevant_docs[q_id][0]] for q_id in filtered_queries.keys()] # Add prefix
    })
    hf_dataset = Dataset.from_pandas(df)

    # Split into train and test sets
    train_test_split = hf_dataset.train_test_split(test_size=test_size)
    train_hf_dataset = train_test_split["train"]
    test_hf_dataset = train_test_split["test"]

    # Prepare data for evaluator from the test set
    test_queries = dict(zip(test_hf_dataset["id"], test_hf_dataset["anchor"]))
    test_corpus = dict(zip(test_hf_dataset["id"], test_hf_dataset["positive"])) # For evaluation, each positive is its own corpus entry
    test_relevant_docs = {}
    for q_id in test_queries:
        test_relevant_docs[q_id] = [q_id] # Query ID maps to its positive document ID

    print(f"Generated {len(train_hf_dataset)} training pairs and {len(test_hf_dataset)} test pairs.")

    return train_hf_dataset, test_queries, test_corpus, test_relevant_docs

# %% [markdown]
# ## Evaluator Functions

# %%
def setup_model(model_id="BAAI/bge-small-en-v1.5"):
    """
    Sets up the SentenceTransformer model.
    """
    print(f"Setting up model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"} if device == "cuda" else {},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="BGE Small English v1.5",
        ),
    )
    print("Model setup complete.")
    return model



# %%
def setup_evaluator(queries, corpus, relevant_docs, matryoshka_dimensions=[384]):
    """
    Sets up the sequential evaluator for different matryoshka dimensions.
    """
    print("Setting up evaluator...")
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    print("Evaluator setup complete.")
    return evaluator



# %%
def setup_trainer(model, train_dataset, evaluator, output_dir="nomic-embed-text-v1-finetuned", matryoshka_dimensions=[384], repo_id=None):
    """
    Sets up the SentenceTransformerTrainer.
    """
    print("Setting up trainer...")
    train_loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8, # Further reduced batch size to prevent OOM
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=16, # Further reduced batch size to prevent OOM
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_384_cosine_ndcg@10", # Updated metric
        push_to_hub=repo_id is not None,
        push_to_hub_model_id=repo_id,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["positive", "anchor"]),
        loss=train_loss,
        evaluator=evaluator,
    )
    print("Trainer setup complete.")
    return trainer, output_dir, args


# %%

def evaluate_model(model, evaluator, stage="Initial"):
    """
    Evaluates the model and prints the results.
    """
    print(f"Performing {stage} evaluation...")
    results = evaluator(model)
    print(f"\n--- {stage} Evaluation Results ---")
    # Dynamically get matryoshka dimensions from evaluator if possible
    matryoshka_dimensions = [e.truncate_dim for e in evaluator.evaluators if isinstance(e, InformationRetrievalEvaluator)]
    if not matryoshka_dimensions: # Fallback if not directly obtainable
        matryoshka_dimensions = [384] # Updated default dimension

    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        if key in results:
            print(f"{key}: {results[key]:.4f}")
        else:
            print(f"Warning: {key} not found in results.")
    print(f"--- End {stage} Evaluation Results ---\n")
    return results

# %% [markdown]
# ## Main Execution

# %%
def main():
    # Generate and split QA pairs
    pdf_files = ["mouch.pdf", "Production and Operations Management Systems.pdf"]
    train_dataset, test_queries, test_corpus, test_relevant_docs = generate_and_split_qa_pairs(pdf_files)

    # Setup model
    model = setup_model()

    # Setup evaluator using the test data
    evaluator = setup_evaluator(test_queries, test_corpus, test_relevant_docs)

    # Initial evaluation
    initial_results = evaluate_model(model, evaluator, stage="Initial")

    # Visualize initial embeddings
    print("Visualizing initial embeddings...")
    sample_texts = list(test_corpus.values())[:100] # Take first 100 documents for visualization
    initial_embeddings = model.encode(sample_texts, convert_to_numpy=True)
    visualize_embeddings(initial_embeddings, title="Initial Embeddings t-SNE", filename="initial_embeddings_tsne.png", method='tsne')
    visualize_embeddings(initial_embeddings, title="Initial Embeddings UMAP", filename="initial_embeddings_umap.png", method='umap')

    # Setup and train trainer
    repo_id = "bge-small-finetuned" # IMPORTANT: Replace with your desired model name (e.g., "my-embedding-model")
    trainer, output_dir, args = setup_trainer(model, train_dataset, evaluator, repo_id=repo_id)
    print("Starting training...")
    trainer.train()
    trainer.save_model() # Save the best model
    if args.push_to_hub:
        print(f"Pushing model to Hugging Face Hub: {args.push_to_hub_model_id}")
        trainer.push_to_hub()
    print("Training complete.")

    # Load the fine-tuned model for final evaluation
    print(f"Loading fine-tuned model from {output_dir} for final evaluation...")
    fine_tuned_model = SentenceTransformer(output_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    # Final evaluation
    final_results = evaluate_model(fine_tuned_model, evaluator, stage="Final")

    # Visualize fine-tuned embeddings
    print("Visualizing fine-tuned embeddings...")
    fine_tuned_embeddings = fine_tuned_model.encode(sample_texts, convert_to_numpy=True)
    visualize_embeddings(fine_tuned_embeddings, title="Fine-Tuned Embeddings t-SNE", filename="fine_tuned_embeddings_tsne.png", method='tsne')
    visualize_embeddings(fine_tuned_embeddings, title="Fine-Tuned Embeddings UMAP", filename="fine_tuned_embeddings_umap.png", method='umap')

    print("\n--- Performance Difference (NDCG@10) ---")
    # Ensure we use the actual dimensions evaluated
    matryoshka_dimensions = [e.truncate_dim for e in evaluator.evaluators if isinstance(e, InformationRetrievalEvaluator)]
    if not matryoshka_dimensions:
        matryoshka_dimensions = [384]

    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        initial_score = initial_results.get(key, 0)
        final_score = final_results.get(key, 0)
        print(f"Dimension {dim}: Initial {initial_score:.4f} -> Final {final_score:.4f} (Change: {final_score - initial_score:.4f})")




# %%
# !pip install accelerate

# Function for t-SNE/UMAP visualization
def visualize_embeddings(embeddings, title="Visualization", filename=None, method='tsne'):
    if embeddings.shape[0] < 2: # Need at least 2 samples for both t-SNE and UMAP
        print(f"Not enough samples ({embeddings.shape[0]}) to perform {method.upper()} visualization. Skipping for '{title}'.")
        return

    X_embedded = None
    if method == 'tsne':
        # Reduce to 2D using t-SNE
        perplexity_val = min(30, embeddings.shape[0]-1) # Ensure perplexity is less than n_samples
        if perplexity_val < 1: # Perplexity must be at least 1
            perplexity_val = 1
        tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
        X_embedded = tsne.fit_transform(embeddings)
    elif method == 'umap':
        # Reduce to 2D using UMAP
        # n_neighbors must be less than n_samples
        n_neighbors_val = min(15, embeddings.shape[0]-1)
        if n_neighbors_val < 2: # UMAP needs at least 2 neighbors for n_components=2
            n_neighbors_val = 2

        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors_val, random_state=42)
        X_embedded = reducer.fit_transform(embeddings)
    else:
        print(f"Unknown visualization method: {method}. Skipping visualization for '{title}'.")
        return

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5)
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close() # Close the plot to free up memory
# %%
if __name__ == "__main__":
    login()
    main()

# %%

# Function for t-SNE visualization
