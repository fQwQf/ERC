"""
FAISS-based Retrieval Module for In-Context Demonstrations

This module builds a semantic index over training data and retrieves
the most similar examples to use as in-context demonstrations.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..data.data_processor import DialogueSample


@dataclass
class RetrievedExample:
    """A retrieved in-context demonstration"""
    sample_id: str
    dialogue_history: List[str]
    target_utterance: str
    emotion: str
    similarity_score: float
    
    def format_as_demonstration(self) -> str:
        """Format as a demonstration for the prompt"""
        history_str = "\n".join(
            f"[Turn {i+1}] {utt}" 
            for i, utt in enumerate(self.dialogue_history[-3:])  # Last 3 turns
        ) if self.dialogue_history else "[No previous context]"
        
        return f"""Example:
Context:
{history_str}
Target Utterance: {self.target_utterance}
Emotion: {self.emotion}
"""


class Retriever:
    """FAISS-based semantic retrieval for in-context demonstrations"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: str = "./cache/retrieval",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS index
        self.index: Optional[faiss.IndexFlatIP] = None
        self.samples: List[DialogueSample] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _get_text_for_embedding(self, sample: DialogueSample) -> str:
        """Create text representation for embedding"""
        # Combine dialogue context and target for semantic matching
        history = " ".join(sample.dialogue_history[-3:])  # Last 3 turns
        return f"{history} [SEP] {sample.target_utterance}"
    
    def build_index(self, samples: List[DialogueSample], batch_size: int = 128):
        """Build FAISS index from training samples"""
        print(f"Building retrieval index for {len(samples)} samples...")
        self.samples = samples
        
        # Generate embeddings
        texts = [self._get_text_for_embedding(s) for s in tqdm(samples, desc="Preparing texts")]
        
        print("Generating embeddings...")
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )
        
        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))
        
        print(f"Index built with {self.index.ntotal} samples")
    
    def retrieve(
        self,
        query_sample: DialogueSample,
        k: int = 3,
        exclude_same_dialogue: bool = True,
    ) -> List[RetrievedExample]:
        """Retrieve top-k most similar examples"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get query embedding
        query_text = self._get_text_for_embedding(query_sample)
        query_embedding = self.encoder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search (retrieve more than k to allow filtering)
        k_search = min(k * 3, len(self.samples))
        distances, indices = self.index.search(query_embedding, k_search)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            sample = self.samples[idx]
            
            # Optionally exclude samples from same dialogue
            if exclude_same_dialogue and query_sample.sample_id.startswith(sample.sample_id[:10]):
                continue
            
            results.append(RetrievedExample(
                sample_id=sample.sample_id,
                dialogue_history=sample.dialogue_history,
                target_utterance=sample.target_utterance,
                emotion=sample.emotion,
                similarity_score=float(dist),
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def retrieve_by_text(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        k: int = 3,
    ) -> List[RetrievedExample]:
        """Retrieve by raw text input"""
        temp_sample = DialogueSample(
            sample_id="query",
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            emotion="",
            emotion_idx=0,
            speaker="",
        )
        return self.retrieve(temp_sample, k=k, exclude_same_dialogue=False)
    
    def save_index(self, path: str):
        """Save index and samples to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss_index.bin"))
        
        # Save samples metadata
        samples_data = [s.to_dict() for s in self.samples]
        with open(path / "samples.json", "w") as f:
            json.dump(samples_data, f)
        
        # Save embeddings
        np.save(str(path / "embeddings.npy"), self.embeddings)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load index and samples from disk"""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss_index.bin"))
        
        # Load samples metadata
        with open(path / "samples.json", "r") as f:
            samples_data = json.load(f)
        
        self.samples = [
            DialogueSample(
                sample_id=s["sample_id"],
                dialogue_history=s["dialogue_history"],
                target_utterance=s["target_utterance"],
                emotion=s["emotion"],
                emotion_idx=s["emotion_idx"],
                speaker=s["speaker"],
                prev_speaker=s.get("prev_speaker"),
                prev_emotion=s.get("prev_emotion"),
                dataset=s.get("dataset", ""),
            )
            for s in samples_data
        ]
        
        # Load embeddings
        self.embeddings = np.load(str(path / "embeddings.npy"))
        
        print(f"Index loaded with {self.index.ntotal} samples from {path}")


class BatchRetriever:
    """Batch retrieval for efficient processing during training"""
    
    def __init__(self, retriever: Retriever, k: int = 3):
        self.retriever = retriever
        self.k = k
    
    def retrieve_batch(
        self, 
        samples: List[DialogueSample],
        show_progress: bool = True,
    ) -> List[List[RetrievedExample]]:
        """Retrieve demonstrations for a batch of samples"""
        results = []
        iterator = tqdm(samples, desc="Retrieving") if show_progress else samples
        
        for sample in iterator:
            retrieved = self.retriever.retrieve(sample, k=self.k)
            results.append(retrieved)
        
        return results
