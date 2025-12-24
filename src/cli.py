"""
CLI entry point for the arXiv LLM training pipeline.

Provides commands for data collection, processing, tokenization, and training.
"""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from config.settings import settings

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """arXiv LLM Training Pipeline CLI."""
    pass


# ============================================================================
# Collect Commands
# ============================================================================

@main.group()
def collect():
    """Data collection commands."""
    pass


@collect.command("arxiv")
@click.option("--limit", "-l", default=100, help="Maximum number of papers to collect")
@click.option("--category", "-c", multiple=True, help="arXiv categories (e.g., cs.LG)")
@click.option("--download-pdfs/--no-pdfs", default=True, help="Download PDF files")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def collect_arxiv(
    limit: int,
    category: tuple,
    download_pdfs: bool,
    output: Optional[str],
):
    """Collect papers from arXiv API."""
    from src.collectors.arxiv_client import ArxivClient
    
    console.print(f"[bold blue]Collecting up to {limit} papers from arXiv[/bold blue]")
    
    categories = list(category) if category else settings.arxiv.category_list
    console.print(f"Categories: {', '.join(categories)}")
    
    async def run():
        client = ArxivClient()
        papers = await client.collect_papers(
            categories=categories,
            limit=limit,
            download_pdfs=download_pdfs,
        )
        return papers
    
    papers = asyncio.run(run())
    console.print(f"[green]✓ Collected {len(papers)} papers[/green]")


@collect.command("ingest")
@click.option("--source", "-s", type=click.Path(exists=True), required=True, help="Source data path")
@click.option("--format", "-f", type=click.Choice(["json", "parquet"]), default="json")
def ingest_data(source: str, format: str):
    """Ingest collected data into storage."""
    from src.collectors.ingestion import IngestionPipeline
    
    console.print(f"[bold blue]Ingesting data from {source}[/bold blue]")
    
    async def run():
        pipeline = IngestionPipeline()
        if format == "json":
            return await pipeline.ingest_from_json(Path(source))
        else:
            console.print("[red]Parquet ingestion not yet implemented[/red]")
            return {}
    
    stats = asyncio.run(run())
    console.print(f"[green]✓ Ingested {stats.get('ingested', 0)} papers[/green]")


# ============================================================================
# Process Commands
# ============================================================================

@main.group()
def process():
    """Data processing commands."""
    pass


@process.command("pipeline")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
@click.option("--stage", "-s", multiple=True, help="Specific stages to run")
@click.option("--resume", "-r", help="Stage to resume from")
def run_pipeline(input: str, output: str, stage: tuple, resume: Optional[str]):
    """Run the full processing pipeline."""
    from src.processing.pipeline import ProcessingPipeline
    
    stages = list(stage) if stage else None
    
    console.print(f"[bold blue]Running processing pipeline[/bold blue]")
    console.print(f"Input: {input}")
    console.print(f"Output: {output}")
    
    pipeline = ProcessingPipeline()
    
    try:
        stats = pipeline.run(input, output, stages=stages, resume_from=resume)
        
        # Display results
        table = Table(title="Pipeline Results")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for stage_name, stage_stats in stats.get("stages", {}).items():
            status = stage_stats.get("status", "unknown")
            details = str(stage_stats.get("stats", {}))
            table.add_row(stage_name, status, details[:50] + "...")
        
        console.print(table)
        
    finally:
        pipeline.stop()


@process.command("dedup")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
@click.option("--threshold", "-t", default=0.85, help="Similarity threshold")
def run_dedup(input: str, output: str, threshold: float):
    """Run deduplication job."""
    from src.processing.deduplication import DeduplicationJob
    
    console.print(f"[bold blue]Running deduplication (threshold={threshold})[/bold blue]")
    
    job = DeduplicationJob(threshold=threshold)
    stats = job.run(input, output)
    
    console.print(f"[green]✓ Removed {stats['duplicates_removed']} duplicates[/green]")
    console.print(f"  Initial: {stats['initial_count']} → Final: {stats['final_count']}")


@process.command("normalize")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
def run_normalize(input: str, output: str):
    """Run normalization job."""
    from src.processing.normalization import NormalizationJob
    
    console.print("[bold blue]Running normalization[/bold blue]")
    
    job = NormalizationJob()
    stats = job.run(input, output)
    
    console.print(f"[green]✓ Normalized {stats['final_count']} documents[/green]")


@process.command("moderate")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
@click.option("--redact/--no-redact", default=True, help="Redact PII")
def run_moderate(input: str, output: str, redact: bool):
    """Run content moderation job."""
    from src.processing.content_moderation import ContentModerationJob
    
    console.print("[bold blue]Running content moderation[/bold blue]")
    
    job = ContentModerationJob(redact_pii=redact)
    stats = job.run(input, output)
    
    console.print(f"[green]✓ Moderated {stats['final_count']} documents[/green]")


# ============================================================================
# Tokenize Commands
# ============================================================================

@main.group()
def tokenize():
    """Tokenization commands."""
    pass


@tokenize.command("train")
@click.option("--input", "-i", required=True, help="Input text file(s)", multiple=True)
@click.option("--output", "-o", help="Output model path")
@click.option("--vocab-size", "-v", default=32000, help="Vocabulary size")
@click.option("--type", "-t", type=click.Choice(["sentencepiece", "bpe"]), default="sentencepiece")
def train_tokenizer(input: tuple, output: Optional[str], vocab_size: int, type: str):
    """Train a tokenizer."""
    from src.tokenization.trainer import TokenizerTrainer
    
    console.print(f"[bold blue]Training {type} tokenizer (vocab_size={vocab_size})[/bold blue]")
    
    trainer = TokenizerTrainer(vocab_size=vocab_size, model_type=type)
    input_files = [Path(f) for f in input]
    
    if type == "sentencepiece":
        model_path = trainer.train_sentencepiece(input_files)
    else:
        model_path = trainer.train_bpe(input_files)
    
    console.print(f"[green]✓ Tokenizer saved to {model_path}[/green]")


@tokenize.command("apply")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
@click.option("--tokenizer", "-t", required=True, help="Tokenizer model path")
@click.option("--max-length", "-m", default=2048, help="Maximum sequence length")
def apply_tokenizer(input: str, output: str, tokenizer: str, max_length: int):
    """Apply tokenizer to dataset."""
    from src.tokenization.spark_tokenizer import SparkTokenizer
    
    console.print(f"[bold blue]Tokenizing dataset (max_length={max_length})[/bold blue]")
    
    job = SparkTokenizer(tokenizer_path=Path(tokenizer), max_length=max_length)
    stats = job.run(input, output)
    
    console.print(f"[green]✓ Created {stats['total_sequences']} sequences[/green]")


# ============================================================================
# Train Commands
# ============================================================================

@main.group()
def train():
    """Training commands."""
    pass


@train.command("start")
@click.option("--data", "-d", required=True, help="Training data path")
@click.option("--output", "-o", default="checkpoints", help="Output directory")
@click.option("--config", "-c", type=click.Path(exists=True), help="Training config file")
@click.option("--epochs", "-e", default=3, help="Number of epochs")
@click.option("--batch-size", "-b", default=8, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--wandb/--no-wandb", default=True, help="Use W&B logging")
def start_training(
    data: str,
    output: str,
    config: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    wandb: bool,
):
    """Start model training."""
    from src.training.model import TransformerConfig, TransformerLM
    from src.training.trainer import LLMTrainer, TokenDataset, TrainingConfig
    from src.training.wandb_integration import WandBLogger
    
    console.print("[bold blue]Starting LLM training[/bold blue]")
    
    # Create model
    model_config = TransformerConfig.small()
    model = TransformerLM(model_config)
    console.print(f"Model parameters: {model.num_parameters():,}")
    
    # Create dataset
    dataset = TokenDataset(Path(data))
    
    # Create training config
    train_config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_dir=Path(output),
    )
    
    # Create W&B logger
    wandb_logger = None
    if wandb:
        wandb_logger = WandBLogger(
            config={
                "model": model_config.__dict__,
                "training": train_config.__dict__,
            }
        )
    
    # Create trainer
    trainer = LLMTrainer(
        model=model,
        train_config=train_config,
        train_dataset=dataset,
        wandb_logger=wandb_logger,
    )
    
    # Train
    stats = trainer.train()
    
    console.print(f"[green]✓ Training complete! Final loss: {stats['final_loss']:.4f}[/green]")
    
    if wandb_logger:
        wandb_logger.finish()


@train.command("resume")
@click.option("--checkpoint", "-c", required=True, help="Checkpoint path")
@click.option("--data", "-d", required=True, help="Training data path")
def resume_training(checkpoint: str, data: str):
    """Resume training from checkpoint."""
    from src.training.model import TransformerConfig, TransformerLM
    from src.training.trainer import LLMTrainer, TokenDataset, TrainingConfig
    
    console.print(f"[bold blue]Resuming training from {checkpoint}[/bold blue]")
    
    # Load config
    import json
    with open(Path(checkpoint) / "config.json") as f:
        saved_config = json.load(f)
    
    model_config = TransformerConfig(**saved_config["model_config"])
    model = TransformerLM(model_config)
    
    dataset = TokenDataset(Path(data))
    train_config = TrainingConfig(**{
        k: Path(v) if k == "output_dir" else v
        for k, v in saved_config["training_config"].items()
    })
    
    trainer = LLMTrainer(
        model=model,
        train_config=train_config,
        train_dataset=dataset,
    )
    
    trainer.load_checkpoint(Path(checkpoint))
    stats = trainer.train()
    
    console.print(f"[green]✓ Training complete! Final loss: {stats['final_loss']:.4f}[/green]")


# ============================================================================
# Status Commands
# ============================================================================

@main.command("status")
def show_status():
    """Show pipeline status."""
    console.print("[bold blue]Pipeline Status[/bold blue]")
    
    # Check storage
    try:
        from src.storage.minio_client import MinioStorage
        storage = MinioStorage()
        console.print("[green]✓ Storage: Connected[/green]")
    except Exception as e:
        console.print(f"[red]✗ Storage: {e}[/red]")
    
    # Show settings
    table = Table(title="Current Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Environment", settings.environment.value)
    table.add_row("MinIO Endpoint", settings.minio.endpoint)
    table.add_row("arXiv Categories", settings.arxiv.categories)
    table.add_row("Vocab Size", str(settings.tokenization.vocab_size))
    
    console.print(table)


if __name__ == "__main__":
    main()
