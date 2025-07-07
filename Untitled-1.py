# %%
!pip install torch tensorboard

# %%
!pip install PyMuPDF sentence-transformers


# %%
!pip install sentence-transformers datasets transformers


# %%
!pip install ipywidgets
from huggingface_hub import login
login()

# %%
pip install -U datasets

# %%
!pip install pymupdf

# %%
!pip install accelerate

# %%


# %%
import sys
print(sys.executable)

# %%
!pip list

# %%
!pip freeze > requirements.txt

# %%
import accelerate
print(accelerate.__version__)

# %%
from transformers.utils import is_accelerate_available
print(is_accelerate_available())

# %%
!pip install llama_index

# %%
from llama_index.core.llms import LLM
from llama_index.core.llms import CompletionResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

# %%
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

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
!pip install llama-index-finetuning

# %%
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

# Load your text chunks (nodes) as before
train_nodes = load_corpus(["mouch.pdf"])
val_nodes = load_corpus(["Production and Operations Management Systems.pdf"])

# Instantiate your HF-based LLM
phi3_llm = HuggingFacePhi3LLM()

# Use it for QA pair generation
train_dataset = generate_qa_embedding_pairs(
    llm=phi3_llm,
    nodes=train_nodes,
    output_path="train_dataset.json"
)

val_dataset = generate_qa_embedding_pairs(
    llm=phi3_llm,
    nodes=val_nodes,
    output_path="val_dataset.json"
)

# Optional: load into dataset objects
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

# %%


# %%
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd

model_id = "BAAI/bge-small-en-v1.5"  # Hugging Face model ID
matryoshka_dimensions = [256, 128, 64] # Important: large to small

# Load a model
model = SentenceTransformer(
    model_id, device="cuda" if torch.cuda.is_available() else "cpu"
)

# load test dataset from json using pandas
test_df = pd.read_json("test_dataset.json", orient="records", lines=True)
train_df = pd.read_json("train_dataset.json", orient="records", lines=True)

# Convert pandas DataFrames to Hugging Face Datasets
test_dataset = Dataset.from_pandas(test_df)
train_dataset = Dataset.from_pandas(train_df)
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)
queries = dict(
    zip(test_dataset["id"], test_dataset["anchor"])
)  # Our queries (qid => question)

# Create a mapping of relevant document (1 in our case) to each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]


matryoshka_evaluators = []
# Iterate over the different dimensions
for dim in matryoshka_dimensions:
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,  # Truncate the embeddings to a certain dimension
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)

# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)

# %%
# Evaluate the model
results = evaluator(model)

# # COMMENT IN for full results
# print(results)

# Print the main score
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print
    print(f"{key}: {results[key]}")

# %%
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer

# Hugging Face model ID: https://huggingface.co/BAAI/bge-base-en-v1.5
model_id = "BAAI/bge-small-en-v1.5"

# load model with SDPA for using Flash Attention 2
model = SentenceTransformer(
    model_id,
    model_kwargs={"attn_implementation": "sdpa"},
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="BGE base Financial Matryoshka",
    ),
)

# %%
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

matryoshka_dimensions = [256, 128, 64]  # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

# %%
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

# load train dataset again
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")

# define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="bge-base-financial-matryoshka", # output directory and hugging face model ID
    num_train_epochs=4,                         # number of epochs
    per_device_train_batch_size=8,             # train batch size
    gradient_accumulation_steps=8,             # for a global batch size of 512
    per_device_eval_batch_size=16,              # evaluation batch size
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",                 # use constant learning rate scheduler
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    tf32=False,                                  # use tf32 precision
    bf16=True,                                  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",                      # evaluate after each epoch
    save_strategy="epoch",                      # save after each epoch
    logging_steps=10,                           # log every 10 steps
    save_total_limit=3,                         # save only the last 3 models
    load_best_model_at_end=True,                # load the best model when training ends
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
)

# %%
from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.select_columns(
        ["positive", "anchor"]
    ),
    loss=train_loss,
    evaluator=evaluator,
)

# %%
!pip install wandb

# %%
wandb login

# %%
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save the best model
trainer.save_model()

# push model to hub
# trainer.model.push_to_hub("bge-small-financial-matryoshka")

# %%
from sentence_transformers import SentenceTransformer

fine_tuned_model = SentenceTransformer(
    args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
)
# Evaluate the model
results = evaluator(fine_tuned_model)

# # COMMENT IN for full results
# print(results)

# Print the main score
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print(f"{key}: {results[key]}")

# %%
dim_256_cosine_ndcg@10: 0.7957342722206704
dim_128_cosine_ndcg@10: 0.7898308785293717
dim_64_cosine_ndcg@10: 0.7614769594274166


dim_256_cosine_ndcg@10: 0.7319604909788028
dim_128_cosine_ndcg@10: 0.7045758818102816
dim_64_cosine_ndcg@10: 0.6334755429440541

# %%



