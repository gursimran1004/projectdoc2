"""
Offline caching module for document storage and retrieval.
Allows the app to work without internet connection.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


class OfflineCache:
    """Manages offline storage of documents and embeddings."""
    
    CACHE_DIR = Path(".offline_cache")
    DOCS_DIR = CACHE_DIR / "documents"
    EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
    METADATA_FILE = CACHE_DIR / "metadata.json"
    
    def __init__(self):
        """Initialize offline cache directories."""
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.DOCS_DIR.mkdir(exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.METADATA_FILE.exists():
            try:
                with open(self.METADATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def cache_document(self, doc_name: str, doc_text: str, sections: List[Dict[str, str]]) -> bool:
        """
        Cache a document locally for offline access.
        
        Args:
            doc_name: Document name/filename
            doc_text: Full document text
            sections: List of document sections with metadata
        
        Returns:
            True if caching successful, False otherwise
        """
        try:
            # Create unique ID based on document name
            doc_id = self._generate_doc_id(doc_name)
            doc_file = self.DOCS_DIR / f"{doc_id}.pkl"
            
            # Prepare document data
            doc_data = {
                "name": doc_name,
                "text": doc_text,
                "sections": sections,
                "cached_at": datetime.now().isoformat(),
                "text_length": len(doc_text),
            }
            
            # Save document
            with open(doc_file, 'wb') as f:
                pickle.dump(doc_data, f)
            
            # Update metadata
            self.metadata[doc_id] = {
                "name": doc_name,
                "doc_id": doc_id,
                "cached_at": doc_data["cached_at"],
                "text_length": doc_data["text_length"],
                "file_path": str(doc_file),
            }
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error caching document {doc_name}: {e}")
            return False
    
    def get_cached_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document data if found, None otherwise
        """
        try:
            if doc_id not in self.metadata:
                return None
            
            doc_file = Path(self.metadata[doc_id]["file_path"])
            if not doc_file.exists():
                return None
            
            with open(doc_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error retrieving cached document {doc_id}: {e}")
            return None
    
    def list_cached_documents(self) -> List[Dict[str, Any]]:
        """Get list of all cached documents."""
        docs = []
        for doc_id, meta in self.metadata.items():
            doc_data = self.get_cached_document(doc_id)
            if doc_data:
                docs.append({
                    "id": doc_id,
                    "name": meta["name"],
                    "cached_at": meta["cached_at"],
                    "size_kb": meta["text_length"] / 1024,
                })
        return docs
    
    def delete_cached_document(self, doc_id: str) -> bool:
        """Delete a cached document."""
        try:
            if doc_id in self.metadata:
                file_path = Path(self.metadata[doc_id]["file_path"])
                if file_path.exists():
                    file_path.unlink()
                del self.metadata[doc_id]
                self._save_metadata()
                return True
        except Exception as e:
            print(f"Error deleting cached document {doc_id}: {e}")
        return False
    
    def cache_embeddings(self, doc_id: str, embeddings: Dict[int, List[float]]) -> bool:
        """
        Cache embeddings for faster offline retrieval.
        
        Args:
            doc_id: Document ID
            embeddings: Dictionary mapping chunk IDs to embedding vectors
        
        Returns:
            True if caching successful
        """
        try:
            emb_file = self.EMBEDDINGS_DIR / f"{doc_id}_embeddings.pkl"
            with open(emb_file, 'wb') as f:
                pickle.dump(embeddings, f)
            return True
        except Exception as e:
            print(f"Error caching embeddings for {doc_id}: {e}")
            return False
    
    def get_cached_embeddings(self, doc_id: str) -> Optional[Dict[int, List[float]]]:
        """Get cached embeddings for a document."""
        try:
            emb_file = self.EMBEDDINGS_DIR / f"{doc_id}_embeddings.pkl"
            if not emb_file.exists():
                return None
            
            with open(emb_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error retrieving embeddings for {doc_id}: {e}")
            return None
    
    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        for file in self.CACHE_DIR.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)
    
    def clear_all_cache(self) -> bool:
        """Clear all cached data."""
        try:
            for file in self.CACHE_DIR.rglob("*"):
                if file.is_file():
                    file.unlink()
            self.metadata = {}
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def _generate_doc_id(self, doc_name: str) -> str:
        """Generate unique document ID from name."""
        import hashlib
        return hashlib.md5(doc_name.encode()).hexdigest()[:12]
    
    def export_for_offline(self, output_path: str = "offline_package.zip") -> bool:
        """
        Export all cached data as a portable package.
        
        Args:
            output_path: Path to save the ZIP file
        
        Returns:
            True if export successful
        """
        try:
            import zipfile
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in self.CACHE_DIR.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(self.CACHE_DIR.parent)
                        zipf.write(file, arcname)
            
            print(f"Cache exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting cache: {e}")
            return False
    
    def import_from_offline(self, input_path: str) -> bool:
        """
        Import cached data from a package.
        
        Args:
            input_path: Path to the ZIP file
        
        Returns:
            True if import successful
        """
        try:
            import zipfile
            
            with zipfile.ZipFile(input_path, 'r') as zipf:
                zipf.extractall(self.CACHE_DIR.parent)
            
            # Reload metadata
            self.metadata = self._load_metadata()
            print(f"Cache imported from {input_path}")
            return True
        except Exception as e:
            print(f"Error importing cache: {e}")
            return False


def is_offline_mode() -> bool:
    """Check if app should run in offline mode."""
    return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def get_offline_cache() -> OfflineCache:
    """Get singleton offline cache instance."""
    if not hasattr(get_offline_cache, "_instance"):
        get_offline_cache._instance = OfflineCache()
    return get_offline_cache._instance
