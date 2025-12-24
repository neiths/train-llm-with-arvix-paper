"""
Distributed tokenization using Spark.

Tokenizes large datasets efficiently across a Spark cluster.
"""

from pathlib import Path
from typing import Optional

import sentencepiece as spm
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType

from config.settings import settings
from src.logging_config import get_logger
from src.processing.base import BaseSparkJob

logger = get_logger(__name__)


class SparkTokenizer(BaseSparkJob):
    """
    Distributed tokenization job using Spark.
    
    Features:
    - Parallel tokenization across cluster
    - Support for SentencePiece and BPE
    - Sequence chunking for long documents
    - Output in training-ready format
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        tokenizer_path: Optional[Path] = None,
        max_length: Optional[int] = None,
        stride: int = 512,
    ):
        """
        Initialize Spark tokenizer.
        
        Args:
            spark: Spark session
            tokenizer_path: Path to trained tokenizer model
            max_length: Maximum sequence length
            stride: Stride for sliding window (overlap)
        """
        super().__init__(spark, "arxiv-tokenization")
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length or settings.training.max_seq_length
        self.stride = stride
        
        # Will be set during run
        self._tokenizer = None
        self._tokenizer_type = None
    
    def run(self, input_path: str, output_path: str) -> dict:
        """
        Run tokenization on input data.
        
        Args:
            input_path: Path to input Parquet data
            output_path: Path for tokenized output
            
        Returns:
            dict: Job statistics
        """
        if self.tokenizer_path is None:
            raise ValueError("Tokenizer path must be set before running")
        
        logger.info(
            "Starting tokenization",
            input_path=input_path,
            tokenizer_path=str(self.tokenizer_path),
            max_length=self.max_length,
        )
        
        # Determine tokenizer type
        self._tokenizer_type = "sentencepiece" if self.tokenizer_path.suffix == ".model" else "bpe"
        
        # Read input data
        df = self.read_parquet(input_path)
        initial_count = df.count()
        
        # Broadcast tokenizer path for workers
        tokenizer_path_broadcast = self.spark.sparkContext.broadcast(str(self.tokenizer_path))
        tokenizer_type_broadcast = self.spark.sparkContext.broadcast(self._tokenizer_type)
        max_length_broadcast = self.spark.sparkContext.broadcast(self.max_length)
        stride_broadcast = self.spark.sparkContext.broadcast(self.stride)
        
        # Define tokenization UDF
        def tokenize_text(text: str) -> list[list[int]]:
            """Tokenize text and split into chunks."""
            if not text:
                return []
            
            tokenizer_path = tokenizer_path_broadcast.value
            tokenizer_type = tokenizer_type_broadcast.value
            max_len = max_length_broadcast.value
            stride = stride_broadcast.value
            
            # Load tokenizer (cached per executor)
            if tokenizer_type == "sentencepiece":
                sp = spm.SentencePieceProcessor()
                sp.load(tokenizer_path)
                token_ids = sp.encode(text, out_type=int)
            else:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(tokenizer_path)
                encoding = tok.encode(text)
                token_ids = encoding.ids
            
            # Split into chunks with sliding window
            chunks = []
            if len(token_ids) <= max_len:
                chunks.append(token_ids)
            else:
                for i in range(0, len(token_ids), stride):
                    chunk = token_ids[i:i + max_len]
                    if len(chunk) >= stride:  # Don't keep tiny final chunks
                        chunks.append(chunk)
            
            return chunks
        
        # Register UDF
        tokenize_udf = F.udf(tokenize_text, ArrayType(ArrayType(IntegerType())))
        
        # Apply tokenization
        df_tokenized = df.withColumn(
            "token_chunks",
            tokenize_udf(F.col("text")),
        )
        
        # Explode chunks into separate rows
        df_exploded = df_tokenized.select(
            F.col("arxiv_id"),
            F.posexplode(F.col("token_chunks")).alias("chunk_idx", "token_ids"),
        )
        
        # Add attention mask (all 1s for actual tokens)
        df_exploded = df_exploded.withColumn(
            "attention_mask",
            F.transform(F.col("token_ids"), lambda x: F.lit(1)),
        )
        
        # Add metadata
        df_exploded = df_exploded.withColumn(
            "num_tokens",
            F.size(F.col("token_ids")),
        )
        
        final_count = df_exploded.count()
        
        # Write output
        self.write_parquet(df_exploded, output_path)
        
        stats = {
            "initial_documents": initial_count,
            "total_sequences": final_count,
            "avg_sequences_per_doc": final_count / initial_count if initial_count > 0 else 0,
            "max_length": self.max_length,
            "stride": self.stride,
        }
        
        logger.info("Tokenization complete", **stats)
        return stats
    
    def tokenize_batch(self, df: DataFrame) -> DataFrame:
        """
        Tokenize a batch of documents.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame: Tokenized DataFrame
        """
        if self.tokenizer_path is None:
            raise ValueError("Tokenizer path must be set")
        
        # Load tokenizer locally
        if self.tokenizer_path.suffix == ".model":
            sp = spm.SentencePieceProcessor()
            sp.load(str(self.tokenizer_path))
            
            def encode(text: str) -> list[int]:
                return sp.encode(text, out_type=int) if text else []
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(self.tokenizer_path))
            
            def encode(text: str) -> list[int]:
                return tok.encode(text).ids if text else []
        
        encode_udf = F.udf(encode, ArrayType(IntegerType()))
        
        df = df.withColumn("token_ids", encode_udf(F.col("text")))
        df = df.withColumn("num_tokens", F.size(F.col("token_ids")))
        
        return df
    
    def create_training_dataset(
        self,
        input_path: str,
        output_path: str,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Create a training-ready dataset.
        
        Args:
            input_path: Path to tokenized data
            output_path: Path for training dataset
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            dict: Dataset statistics
        """
        logger.info("Creating training dataset", input_path=input_path)
        
        df = self.read_parquet(input_path)
        
        if add_special_tokens:
            # Add BOS (2) and EOS (3) tokens
            df = df.withColumn(
                "token_ids",
                F.concat(
                    F.array(F.lit(2)),  # BOS
                    F.col("token_ids"),
                    F.array(F.lit(3)),  # EOS
                ),
            )
            
            df = df.withColumn(
                "attention_mask",
                F.concat(
                    F.array(F.lit(1)),
                    F.col("attention_mask"),
                    F.array(F.lit(1)),
                ),
            )
            
            df = df.withColumn("num_tokens", F.size(F.col("token_ids")))
        
        # Create labels (same as input for causal LM)
        df = df.withColumn("labels", F.col("token_ids"))
        
        total_samples = df.count()
        total_tokens = df.agg(F.sum("num_tokens")).collect()[0][0]
        
        self.write_parquet(df, output_path)
        
        stats = {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        }
        
        logger.info("Training dataset created", **stats)
        return stats
