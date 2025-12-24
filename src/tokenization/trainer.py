"""
Tokenizer training module.

Supports SentencePiece and HuggingFace tokenizers (BPE).
"""

import tempfile
from pathlib import Path
from typing import Iterator, Literal, Optional

import sentencepiece as spm
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class TokenizerTrainer:
    """
    Trains tokenizers for the LLM pipeline.
    
    Supports:
    - SentencePiece (Unigram, BPE)
    - HuggingFace Tokenizers (BPE, WordPiece)
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        model_type: Literal["sentencepiece", "bpe"] = "sentencepiece",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Target vocabulary size
            model_type: Type of tokenizer (sentencepiece or bpe)
            output_dir: Directory for saving models
        """
        self.vocab_size = vocab_size or settings.tokenization.vocab_size
        self.model_type = model_type
        self.output_dir = output_dir or settings.data_dir / "tokenizers"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "TokenizerTrainer initialized",
            vocab_size=self.vocab_size,
            model_type=self.model_type,
        )
    
    def train_sentencepiece(
        self,
        input_files: list[Path],
        model_prefix: str = "arxiv_sp",
        character_coverage: float = 0.9995,
        model_type: str = "unigram",
        special_tokens: Optional[list[str]] = None,
    ) -> Path:
        """
        Train a SentencePiece tokenizer.
        
        Args:
            input_files: List of text files for training
            model_prefix: Output model prefix
            character_coverage: Character coverage ratio
            model_type: SentencePiece model type (unigram, bpe, char, word)
            special_tokens: Additional special tokens
            
        Returns:
            Path: Path to trained model file
        """
        special_tokens = special_tokens or []
        default_special = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        all_special = default_special + [t for t in special_tokens if t not in default_special]
        
        model_path = self.output_dir / f"{model_prefix}.model"
        
        logger.info(
            "Training SentencePiece model",
            input_files=[str(f) for f in input_files],
            vocab_size=self.vocab_size,
            model_type=model_type,
        )
        
        # Join input files
        input_arg = ",".join(str(f) for f in input_files)
        
        # Train the model
        spm.SentencePieceTrainer.train(
            input=input_arg,
            model_prefix=str(self.output_dir / model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols=all_special[4:],  # Skip default special tokens
            num_threads=8,
            train_extremely_large_corpus=True,
        )
        
        logger.info("SentencePiece training complete", model_path=str(model_path))
        return model_path
    
    def train_bpe(
        self,
        input_files: list[Path],
        model_name: str = "arxiv_bpe",
        min_frequency: int = 2,
        special_tokens: Optional[list[str]] = None,
    ) -> Path:
        """
        Train a BPE tokenizer using HuggingFace tokenizers.
        
        Args:
            input_files: List of text files for training
            model_name: Output model name
            min_frequency: Minimum frequency for tokens
            special_tokens: Additional special tokens
            
        Returns:
            Path: Path to trained tokenizer
        """
        special_tokens = special_tokens or []
        default_special = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        all_special = default_special + [t for t in special_tokens if t not in default_special]
        
        model_path = self.output_dir / f"{model_name}.json"
        
        logger.info(
            "Training BPE tokenizer",
            input_files=[str(f) for f in input_files],
            vocab_size=self.vocab_size,
        )
        
        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # Add normalizer
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])
        
        # Add pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ])
        
        # Train
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_special,
            show_progress=True,
        )
        
        tokenizer.train([str(f) for f in input_files], trainer)
        
        # Add post-processor for special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )
        
        # Save
        tokenizer.save(str(model_path))
        
        logger.info("BPE training complete", model_path=str(model_path))
        return model_path
    
    def train_from_iterator(
        self,
        texts: Iterator[str],
        model_name: str = "arxiv_tokenizer",
    ) -> Path:
        """
        Train tokenizer from an iterator of texts.
        
        Args:
            texts: Iterator yielding text strings
            model_name: Output model name
            
        Returns:
            Path: Path to trained model
        """
        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text + "\n")
            temp_path = Path(f.name)
        
        try:
            if self.model_type == "sentencepiece":
                return self.train_sentencepiece([temp_path], model_name)
            else:
                return self.train_bpe([temp_path], model_name)
        finally:
            temp_path.unlink()
    
    def load_sentencepiece(self, model_path: Path) -> spm.SentencePieceProcessor:
        """Load a trained SentencePiece model."""
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        logger.info("Loaded SentencePiece model", path=str(model_path))
        return sp
    
    def load_bpe(self, model_path: Path) -> Tokenizer:
        """Load a trained BPE tokenizer."""
        tokenizer = Tokenizer.from_file(str(model_path))
        logger.info("Loaded BPE tokenizer", path=str(model_path))
        return tokenizer


class VocabularyManager:
    """
    Manages vocabulary files and metadata.
    """
    
    def __init__(self, vocab_dir: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            vocab_dir: Directory for vocabulary files
        """
        self.vocab_dir = vocab_dir or settings.data_dir / "vocab"
        self.vocab_dir.mkdir(parents=True, exist_ok=True)
    
    def export_vocabulary(
        self,
        tokenizer_path: Path,
        output_name: str = "vocabulary.txt",
    ) -> Path:
        """
        Export vocabulary to a text file.
        
        Args:
            tokenizer_path: Path to tokenizer model
            output_name: Output filename
            
        Returns:
            Path: Path to vocabulary file
        """
        output_path = self.vocab_dir / output_name
        
        if tokenizer_path.suffix == ".model":
            # SentencePiece
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path))
            
            with open(output_path, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    piece = sp.id_to_piece(i)
                    f.write(f"{piece}\t{i}\n")
        else:
            # HuggingFace tokenizer
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            vocab = tokenizer.get_vocab()
            
            with open(output_path, "w", encoding="utf-8") as f:
                for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\t{idx}\n")
        
        logger.info("Exported vocabulary", path=str(output_path))
        return output_path
    
    def get_vocab_stats(self, tokenizer_path: Path) -> dict:
        """
        Get vocabulary statistics.
        
        Args:
            tokenizer_path: Path to tokenizer model
            
        Returns:
            dict: Vocabulary statistics
        """
        if tokenizer_path.suffix == ".model":
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path))
            vocab_size = sp.get_piece_size()
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            vocab_size = tokenizer.get_vocab_size()
        
        return {
            "vocab_size": vocab_size,
            "model_path": str(tokenizer_path),
            "model_type": "sentencepiece" if tokenizer_path.suffix == ".model" else "bpe",
        }
