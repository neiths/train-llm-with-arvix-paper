"""
Kubeflow Pipeline for arXiv LLM Training.

Defines the complete ML pipeline from data collection to model training.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


# ============================================================================
# Component Definitions
# ============================================================================

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["arxiv", "minio", "pyarrow"],
)
def collect_papers(
    categories: str,
    limit: int,
    output_dataset: Output[Dataset],
) -> NamedTuple("Outputs", [("num_papers", int)]):
    """Collect papers from arXiv API."""
    import json
    from pathlib import Path
    
    # Import would happen here in actual execution
    # from src.collectors.arxiv_client import ArxivClient
    
    # Placeholder for actual implementation
    output_path = Path(output_dataset.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    metadata = {
        "categories": categories.split(","),
        "limit": limit,
        "num_papers": limit,  # Placeholder
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return (limit,)


@dsl.component(
    base_image="bitnami/spark:3.5",
    packages_to_install=["pyspark", "minio"],
)
def process_data(
    input_dataset: Input[Dataset],
    threshold: float,
    output_dataset: Output[Dataset],
) -> NamedTuple("Outputs", [("processed_count", int)]):
    """Run data processing pipeline."""
    from pathlib import Path
    
    # Placeholder implementation
    input_path = Path(input_dataset.path)
    output_path = Path(output_dataset.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Would run actual spark jobs here
    processed_count = 100  # Placeholder
    
    return (processed_count,)


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["sentencepiece", "tokenizers"],
)
def train_tokenizer(
    input_dataset: Input[Dataset],
    vocab_size: int,
    output_model: Output[Model],
) -> NamedTuple("Outputs", [("vocab_size", int)]):
    """Train tokenizer on processed data."""
    from pathlib import Path
    
    output_path = Path(output_model.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Would train actual tokenizer here
    
    return (vocab_size,)


@dsl.component(
    base_image="bitnami/spark:3.5",
    packages_to_install=["pyspark", "sentencepiece"],
)
def tokenize_dataset(
    input_dataset: Input[Dataset],
    tokenizer_model: Input[Model],
    max_length: int,
    output_dataset: Output[Dataset],
) -> NamedTuple("Outputs", [("num_sequences", int)]):
    """Tokenize dataset using trained tokenizer."""
    from pathlib import Path
    
    output_path = Path(output_dataset.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Would run spark tokenization here
    num_sequences = 10000  # Placeholder
    
    return (num_sequences,)


@dsl.component(
    base_image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["wandb", "accelerate"],
)
def train_model(
    input_dataset: Input[Dataset],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_model: Output[Model],
) -> NamedTuple("Outputs", [("final_loss", float)]):
    """Train the LLM."""
    from pathlib import Path
    
    output_path = Path(output_model.path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Would train actual model here
    final_loss = 2.5  # Placeholder
    
    return (final_loss,)


@dsl.component(
    base_image="python:3.10",
)
def evaluate_model(
    model: Input[Model],
    test_dataset: Input[Dataset],
) -> NamedTuple("Outputs", [("perplexity", float), ("accuracy", float)]):
    """Evaluate the trained model."""
    # Would evaluate actual model here
    perplexity = 15.0  # Placeholder
    accuracy = 0.85  # Placeholder
    
    return (perplexity, accuracy)


# ============================================================================
# Pipeline Definition
# ============================================================================

@dsl.pipeline(
    name="arXiv LLM Training Pipeline",
    description="End-to-end pipeline for training LLMs on arXiv papers",
)
def arxiv_llm_pipeline(
    # Collection parameters
    categories: str = "cs.LG,cs.CL,cs.AI",
    collection_limit: int = 10000,
    
    # Processing parameters
    dedup_threshold: float = 0.85,
    
    # Tokenization parameters
    vocab_size: int = 32000,
    max_length: int = 2048,
    
    # Training parameters
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
):
    """
    Complete ML pipeline for training LLMs on arXiv data.
    
    Stages:
    1. Collect papers from arXiv API
    2. Process data (dedup, normalize, moderate)
    3. Train tokenizer
    4. Tokenize dataset
    5. Train model
    6. Evaluate model
    """
    # Stage 1: Collect papers
    collect_task = collect_papers(
        categories=categories,
        limit=collection_limit,
    )
    collect_task.set_display_name("Collect Papers")
    collect_task.set_caching_options(enable_caching=True)
    
    # Stage 2: Process data
    process_task = process_data(
        input_dataset=collect_task.outputs["output_dataset"],
        threshold=dedup_threshold,
    )
    process_task.set_display_name("Process Data")
    process_task.set_cpu_request("4")
    process_task.set_memory_request("16Gi")
    
    # Stage 3: Train tokenizer
    tokenizer_task = train_tokenizer(
        input_dataset=process_task.outputs["output_dataset"],
        vocab_size=vocab_size,
    )
    tokenizer_task.set_display_name("Train Tokenizer")
    
    # Stage 4: Tokenize dataset
    tokenize_task = tokenize_dataset(
        input_dataset=process_task.outputs["output_dataset"],
        tokenizer_model=tokenizer_task.outputs["output_model"],
        max_length=max_length,
    )
    tokenize_task.set_display_name("Tokenize Dataset")
    tokenize_task.set_cpu_request("8")
    tokenize_task.set_memory_request("32Gi")
    
    # Stage 5: Train model
    train_task = train_model(
        input_dataset=tokenize_task.outputs["output_dataset"],
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    train_task.set_display_name("Train Model")
    train_task.set_cpu_request("8")
    train_task.set_memory_request("64Gi")
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)
    
    # Stage 6: Evaluate model
    eval_task = evaluate_model(
        model=train_task.outputs["output_model"],
        test_dataset=tokenize_task.outputs["output_dataset"],
    )
    eval_task.set_display_name("Evaluate Model")


# ============================================================================
# Recurring Pipeline
# ============================================================================

@dsl.pipeline(
    name="arXiv Data Update Pipeline",
    description="Recurring pipeline for collecting new papers",
)
def arxiv_update_pipeline(
    categories: str = "cs.LG,cs.CL,cs.AI",
    days_back: int = 7,
):
    """
    Recurring pipeline for updating the training data.
    
    Runs periodically to collect new papers and update the dataset.
    """
    # Collect new papers from the last N days
    collect_task = collect_papers(
        categories=categories,
        limit=1000,  # Smaller batch for updates
    )
    collect_task.set_display_name("Collect New Papers")
    
    # Process with incremental deduplication
    process_task = process_data(
        input_dataset=collect_task.outputs["output_dataset"],
        threshold=0.85,
    )
    process_task.set_display_name("Process New Data")


if __name__ == "__main__":
    # Compile pipeline
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=arxiv_llm_pipeline,
        package_path="arxiv_llm_pipeline.yaml",
    )
    
    compiler.Compiler().compile(
        pipeline_func=arxiv_update_pipeline,
        package_path="arxiv_update_pipeline.yaml",
    )
    
    print("Pipelines compiled successfully!")
