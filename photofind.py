#!/usr/bin/env python3.13
import sys
import os
import shutil
import torch
import json
import argparse
import numpy as np
import subprocess
import threading
import queue
import gc
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple, Callable
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QListWidgetItem, 
    QVBoxLayout, QWidget, QLineEdit, QLabel, QStatusBar, 
    QFileDialog, QMessageBox, QToolBar, QHBoxLayout, QPushButton, 
    QMenu, QDialog, QSplitter, QAbstractItemView, QFrame, QTextEdit,
    QProgressBar, QCompleter
)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction, QKeySequence, QShortcut, QDesktopServices
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QFile, QPoint, QTimer

# --- Helper Functions ---

def format_duration(seconds: float) -> str:
    """Formats seconds into a human readable string like '2m 15s'"""
    if seconds < 60:
        return f"{int(seconds)}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours = int(mins // 60)
    mins = int(mins % 60)
    return f"{hours}h {mins}m"

# --- Backend Logic ---

class PhotoSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Info] Loading CLIP model on {self.device}...")
        
        model_id = "openai/clip-vit-large-patch14"
        
        try:
            print("[Info] Attempting offline load from cache...")
            self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=True)
            print("[Info] Model loaded successfully (Offline).")
        except OSError:
            print("-" * 40)
            print("[Warning] Model files not found in cache.")
            print("[Info] Connecting to Hugging Face Hub to download...")
            print("-" * 40)
            self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
            self.model = CLIPModel.from_pretrained(model_id)
            print("\n[Info] Download complete. Future runs will be offline.")
        
        if self.device == "cuda":
            self.model = self.model.half()
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.projection_dim
        
        self.cache_dir = Path.home() / ".cache" / "photofind"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_cache_dir = self.cache_dir / "thumbs"
        self.thumb_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / "photo_index.json"
        self.embeddings_file = self.cache_dir / "photo_embeddings.pt"
        self.metadata_file = self.cache_dir / "photo_metadata.json"
        self.history_file = self.cache_dir / "search_history.json"
        self.stats_file = self.cache_dir / "stats.json"
        
        self.image_paths: List[str] = []
        self.image_embeddings: Optional[torch.Tensor] = None
        self.image_metadata: Dict[str, Dict[str, Any]] = {}
        self.indexed_set: Set[str] = set()
        self.stats = self._load_stats()
        
        self._lock = threading.Lock()
        self._embeddings_on_gpu: Optional[torch.Tensor] = None
        self._embeddings_device: str = 'cpu'
        self._embeddings_dirty: bool = True
        
        # Dynamic Batch Size State for Self-Optimization
        self.current_batch_size = 32
        self.batch_size_lock = threading.Lock()

    def _load_stats(self) -> Dict[str, Any]:
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f: return json.load(f)
            except Exception: pass
        return {}

    def save_index_stats(self, duration: float, count: int) -> None:
        self.stats["last_index_duration"] = format_duration(duration)
        self.stats["last_index_count"] = count
        self._save_stats()

    def save_dedupe_stats(self, duration: float, count: int) -> None:
        self.stats["last_dedupe_duration"] = format_duration(duration)
        self.stats["last_dedupe_count"] = count
        self._save_stats()

    def _save_stats(self) -> None:
        try:
            with open(self.stats_file, "w") as f: json.dump(self.stats, f)
        except Exception: pass

    def load_index(self) -> bool:
        if self.index_file.exists() and self.embeddings_file.exists():
            try:
                with open(self.index_file, "r", errors='surrogateescape') as f:
                    raw_paths = json.load(f)
                self.image_paths = raw_paths
                
                loaded_embeddings = torch.load(str(self.embeddings_file), map_location='cpu', weights_only=False)
                
                if loaded_embeddings.shape[1] != self.embedding_dim:
                    print(f"[Backend] Incompatible index detected. Re-index required.")
                    return False

                self.image_embeddings = loaded_embeddings
                self.indexed_set = set(raw_paths)
                
                if self.metadata_file.exists():
                    with open(self.metadata_file, "r", errors='surrogateescape') as f:
                        self.image_metadata = json.load(f)
                else:
                    self.image_metadata = {}
                
                for p in self.image_paths:
                    if p not in self.image_metadata:
                        self.image_metadata[p] = {"std_dev": 50.0, "edge_score": 50.0}

                self._embeddings_dirty = True
                print(f"[Backend] Loaded {len(self.image_paths)} image embeddings.")
                return True
            except Exception as e:
                print(f"[Backend] Error loading index: {e}")
        return False

    def _get_cached_thumbnail(self, path: str, size: Tuple[int, int] = (200, 200)) -> Optional[Image.Image]:
        thumb_hash = hashlib.md5(path.encode('utf-8', errors='surrogateescape')).hexdigest()
        thumb_path = self.thumb_cache_dir / f"{thumb_hash}.jpg"
        
        if thumb_path.exists():
            try:
                return Image.open(thumb_path).copy()
            except Exception:
                pass
        
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                thumb = img.copy()
                thumb.thumbnail(size, Image.Resampling.LANCZOS)
                thumb.save(thumb_path, "JPEG", quality=85)
                return thumb
        except Exception:
            return None

    def _load_single_image(self, file_path: str) -> Tuple[Optional[Image.Image], Optional[str], Optional[Dict[str, float]]]:
        try:
            img = Image.open(file_path)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            
            if min(w, h) < 32 or max(w, h) < 64:
                return None, None, None

            if img.mode != "RGB":
                img = img.convert("RGB")
            else:
                img.load() 

            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            gray = np.array(img.convert("L"))
            std_dev = float(np.std(gray))
            dx = np.abs(gray[:, 2:] - gray[:, :-2])
            dy = np.abs(gray[2:, :] - gray[:-2, :])
            edge_score = float(np.mean(dx) + np.mean(dy))

            return img, os.path.realpath(file_path), {
                "std_dev": std_dev,
                "edge_score": edge_score
            }
        except Exception:
            return None, None, None

    def _prepare_inputs(self, batch_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        numpy_images = [np.array(img, dtype=np.float32) for img in batch_images]
        batch_array = np.stack(numpy_images, axis=0)
        del numpy_images
        
        inputs = self.processor(images=batch_array, return_tensors="pt", do_resize=False, do_center_crop=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs['pixel_values'] = inputs['pixel_values'].half()
        return inputs

    def index_photos(self, source_dir: str, progress_callback: Optional[Callable[[str, int, int], None]] = None, 
                     cancel_check: Optional[Callable[[], bool]] = None) -> None:
        source_path = os.path.realpath(source_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # --- 1. Initialization ---
        print(f"[Info] Scanning {source_path} for images...")
        files = []
        for root, dirs, filenames in os.walk(source_path):
            if cancel_check and cancel_check():
                print("[Info] Indexing cancelled.")
                return
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    files.append(os.path.join(root, filename))
        
        with self._lock:
            if self.indexed_set:
                initial_count = len(files)
                files = [f for f in files if f not in self.indexed_set]
                skipped = initial_count - len(files)
                if skipped > 0:
                    print(f"[Info] Skipped {skipped} already indexed images.")

        if not files:
            print("[Info] No new images to index.")
            return

        total_files = len(files)
        print(f"[Info] Processing {total_files} new images...")

        # --- 2. Self-Optimization Setup ---
        if self.device == "cuda":
            try:
                free_vram, _ = torch.cuda.mem_get_info()
                free_vram_mb = free_vram / (1024**2)
                self.current_batch_size = max(16, min(64, int(free_vram_mb / 25)))
                print(f"[Info] GPU detected. Initial batch size: {self.current_batch_size}")
            except Exception:
                self.current_batch_size = 16
        else:
            self.current_batch_size = 8 

        max_loaders = min(12, os.cpu_count() or 4)
        load_queue = queue.Queue(maxsize=500)
        gpu_queue = queue.Queue(maxsize=4)
        
        print(f"[Info] Pipeline: {max_loaders} loaders -> preprocess -> GPU")

        def loader_worker(file_chunk: List[str]) -> None:
            for f in file_chunk:
                if cancel_check and cancel_check(): break
                img, path, meta = self._load_single_image(f)
                if img is not None:
                    load_queue.put((img, path, meta))
            load_queue.put(None)

        def preprocess_worker(num_loaders: int) -> None:
            active_loaders = num_loaders
            batch_images: List[Image.Image] = []
            batch_map: List[str] = []
            batch_meta: List[Dict[str, float]] = []
            
            while True:
                with self.batch_size_lock:
                    current_bs = self.current_batch_size

                while len(batch_images) < current_bs:
                    if cancel_check and cancel_check():
                        if batch_images:
                            inputs = self._prepare_inputs(batch_images)
                            gpu_queue.put((inputs, batch_map, batch_meta))
                        gpu_queue.put(None)
                        return
                    try:
                        item = load_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        active_loaders -= 1
                        if active_loaders == 0:
                            if batch_images:
                                inputs = self._prepare_inputs(batch_images)
                                gpu_queue.put((inputs, batch_map, batch_meta))
                            gpu_queue.put(None)
                            return
                        continue
                    img, path, meta = item
                    batch_images.append(img)
                    batch_map.append(path)
                    batch_meta.append(meta)
                
                if batch_images:
                    inputs = self._prepare_inputs(batch_images)
                    for img in batch_images: img.close()
                    gpu_queue.put((inputs, batch_map, batch_meta))
                    batch_images, batch_map, batch_meta = [], [], []

        chunk_size = max(1, (total_files + max_loaders - 1) // max_loaders)
        chunks = [files[i:i+chunk_size] for i in range(0, total_files, chunk_size)]
        
        loader_threads = [threading.Thread(target=loader_worker, args=(chunk,), daemon=True) for chunk in chunks]
        for t in loader_threads: t.start()
        preprocess_thread = threading.Thread(target=preprocess_worker, args=(len(loader_threads),), daemon=True)
        preprocess_thread.start()

        new_embeddings: List[torch.Tensor] = []
        new_paths: List[str] = []
        new_metadata: List[Dict[str, float]] = []
        processed_count = 0
        first_batch_processed = False
        
        # --- 3. GPU Consumption Loop with Recovery ---
        while True:
            if cancel_check and cancel_check(): break
            try:
                item = gpu_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if item is None: break
            
            inputs, batch_map, batch_meta = item
            batch_size_actual = len(batch_map)
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    with torch.no_grad():
                        raw_output = self.model.get_image_features(**inputs)
                        if hasattr(raw_output, 'pooler_output'):
                            features = raw_output.pooler_output
                        elif isinstance(raw_output, tuple):
                            features = raw_output[0]
                        else:
                            features = raw_output
                            
                        features = features / features.norm(p=2, dim=-1, keepdim=True)
                        
                        if self.device == "cuda" and not first_batch_processed:
                            torch.cuda.synchronize()
                            mem_used = torch.cuda.max_memory_allocated() / (1024**2)
                            bytes_per_img = mem_used / batch_size_actual
                            free_vram, _ = torch.cuda.mem_get_info()
                            target_bs = int(((free_vram * 0.85) / (1024**2)) / bytes_per_img)
                            
                            with self.batch_size_lock:
                                self.current_batch_size = max(1, min(256, target_bs))
                            
                            print(f"[Opt] Calibration: {mem_used:.1f}MB used. Est. {bytes_per_img:.2f}MB/img. New batch size: {self.current_batch_size}")
                            first_batch_processed = True

                        new_embeddings.append(features.cpu())
                        new_paths.extend(batch_map)
                        new_metadata.extend(batch_meta)
                    
                    break 
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[Warn] CUDA OOM on batch {batch_size_actual}. Reducing batch size and retrying...")
                        torch.cuda.empty_cache()
                        
                        with self.batch_size_lock:
                            if self.current_batch_size > 1:
                                self.current_batch_size = max(1, int(self.current_batch_size * 0.5))
                            print(f"[Opt] Reduced batch size to {self.current_batch_size}")
                        
                        if attempt < max_retries - 1:
                            print("[Opt] Skipping current batch to recover. You can re-index later to catch these.")
                            break 
                        else:
                            raise
                    else:
                        raise

            del inputs
            processed_count += batch_size_actual
            if progress_callback:
                progress_callback(f"Indexing... {processed_count}/{total_files} images (BS: {self.current_batch_size})", processed_count, total_files)

        for t in loader_threads: t.join(timeout=2.0)
        preprocess_thread.join(timeout=2.0)
        gc.collect()
        
        if cancel_check and cancel_check():
            print("[Info] Indexing cancelled by user.")
            return
            
        if new_embeddings:
            print(f"[Info] Saving index ({len(self.image_paths) + len(new_paths)} total embeddings)...")
            new_embeddings_tensor = torch.cat(new_embeddings, dim=0)
            del new_embeddings
            gc.collect()
            
            with self._lock:
                if self.image_embeddings is not None:
                    all_embeddings = torch.cat([self.image_embeddings, new_embeddings_tensor], dim=0)
                    all_paths = self.image_paths + new_paths
                    del self.image_embeddings
                else:
                    all_embeddings = new_embeddings_tensor
                    all_paths = new_paths

                for i, p in enumerate(new_paths):
                    self.image_metadata[p] = new_metadata[i]
                
                temp_emb = str(self.embeddings_file) + ".tmp"
                temp_idx = str(self.index_file) + ".tmp"
                temp_meta = str(self.metadata_file) + ".tmp"
                
                torch.save(all_embeddings, temp_emb)
                with open(temp_idx, "w", errors='surrogateescape') as f: json.dump(all_paths, f)
                with open(temp_meta, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
                    
                os.replace(temp_emb, str(self.embeddings_file))
                os.replace(temp_idx, str(self.index_file))
                os.replace(temp_meta, str(self.metadata_file))

                self.image_embeddings = all_embeddings
                self.image_paths = all_paths
                self.indexed_set = set(self.image_paths)
                self._embeddings_dirty = True
                del all_embeddings
            gc.collect()
            print("[Info] Indexing complete.")
        else:
            print("[Info] No valid images processed.")

    def clean_database(self, progress_callback: Optional[Callable[[str], None]] = None) -> int:
        if not self.image_paths: return 0
        valid_paths, valid_indices = [], []
        removed_count = 0
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if os.path.exists(path):
                valid_paths.append(path)
                valid_indices.append(i)
            else:
                removed_count += 1
                if path in self.image_metadata: del self.image_metadata[path]
            if progress_callback and i % 1000 == 0: progress_callback(f"Verifying... {i}/{total}")
        
        if removed_count > 0:
            print(f"[Info] Cleaning database: Removing {removed_count} missing files.")
            with self._lock:
                self.image_paths = valid_paths
                self.indexed_set = set(self.image_paths)
                if self.image_embeddings is not None:
                    self.image_embeddings = self.image_embeddings[valid_indices] if valid_indices else None
                self._embeddings_dirty = True
                
                if self.image_embeddings is not None:
                    torch.save(self.image_embeddings, str(self.embeddings_file))
                elif self.embeddings_file.exists():
                    os.remove(self.embeddings_file)
                with open(self.index_file, "w", errors='surrogateescape') as f: json.dump(self.image_paths, f)
                with open(self.metadata_file, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
        return removed_count

    def mark_photo_deleted(self, file_path: str) -> None:
        with self._lock:
            if file_path in self.image_metadata: self.image_metadata[file_path]['deleted'] = True
            with open(self.metadata_file, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
            if file_path in self.indexed_set: self.indexed_set.remove(file_path)

    def _get_embeddings_gpu(self) -> torch.Tensor:
        if self._embeddings_dirty or self._embeddings_device != self.device or self._embeddings_on_gpu is None:
            self._embeddings_on_gpu = self.image_embeddings.to(self.device)
            if self.device == "cuda": self._embeddings_on_gpu = self._embeddings_on_gpu.half()
            self._embeddings_device = self.device
            self._embeddings_dirty = False
        return self._embeddings_on_gpu

    def search(self, query: str, top_k: int = 68) -> List[Dict[str, Any]]:
        if self.image_embeddings is None or len(self.image_embeddings) == 0: return []
        
        search_text = f"a photo of a {query}" if len(query.split()) < 4 else query
        inputs = self.processor(text=[search_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            raw_output = self.model.get_text_features(**inputs)
            text_features = raw_output.pooler_output if hasattr(raw_output, 'pooler_output') else (raw_output[0] if isinstance(raw_output, tuple) else raw_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        similarities = (self._get_embeddings_gpu() @ text_features.T).squeeze(1)
        requested_k = min(top_k * 10, len(similarities))
        values, indices = torch.topk(similarities, requested_k)
        
        scores, idxs = values.cpu().numpy(), indices.cpu().numpy()
        if len(scores) == 0: return []
        
        effective_floor = max(0.19, scores[0] * 0.55)
        results = []
        for i in range(len(scores)):
            if scores[i] < effective_floor: continue
            fp = self.image_paths[idxs[i]]
            meta = self.image_metadata.get(fp, {})
            if not meta.get('deleted') and not self._is_garbage(meta):
                results.append({"file": fp, "score": float(scores[i])})
            if len(results) >= top_k: break
        return results

    def _is_garbage(self, meta: Dict[str, Any]) -> bool:
        std_dev = meta.get("std_dev", 100.0)
        edge_score = meta.get("edge_score")
        if edge_score is None: return std_dev < 25.0 
        return std_dev < 10.0 or (std_dev < 25.0 and edge_score < 15.0)

    def get_garbage_photos(self) -> List[Dict[str, Any]]:
        return [{"file": p, "score": 0.0, "is_garbage": True} for p, m in self.image_metadata.items() if os.path.exists(p) and not m.get('deleted') and self._is_garbage(m)]
    
    def load_search_history(self) -> List[str]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f: return json.load(f)
            except Exception: pass
        return []
    
    def save_search_history(self, history: List[str]) -> None:
        try:
            with open(self.history_file, "w") as f: json.dump(history[-100:], f)
        except Exception: pass

# --- Threading Workers ---

class IndexWorker(QThread):
    progress = pyqtSignal(str, int, int)
    result_ready = pyqtSignal(int, float) 
    
    def __init__(self, searcher: PhotoSearch, directory: str):
        super().__init__()
        self.searcher, self.directory, self._cancelled = searcher, directory, False
    def cancel(self): self._cancelled = True
    def run(self):
        start_time = time.time()
        self.searcher.index_photos(self.directory, lambda m,c,t: self.progress.emit(m,c,t), lambda: self._cancelled)
        duration = time.time() - start_time
        self.result_ready.emit(0, duration)

class SearchWorker(QThread):
    result = pyqtSignal(list)
    def __init__(self, searcher: PhotoSearch, query: str):
        super().__init__()
        self.searcher, self.query = searcher, query
    def run(self): self.result.emit(self.searcher.search(self.query))

class GarbageWorker(QThread):
    result = pyqtSignal(list)
    def __init__(self, searcher: PhotoSearch):
        super().__init__()
        self.searcher = searcher
    def run(self): self.result.emit(self.searcher.get_garbage_photos())

class CleanWorker(QThread):
    clean_complete = pyqtSignal(int)
    def __init__(self, searcher: PhotoSearch):
        super().__init__()
        self.searcher = searcher
    def run(self): self.clean_complete.emit(self.searcher.clean_database())

class ThumbnailWorker(QThread):
    thumbnail_loaded = pyqtSignal(str, QImage)
    def __init__(self, files: List[str], searcher: Optional[PhotoSearch] = None):
        super().__init__()
        self.files, self.searcher = files, searcher
        self._cancelled = False
        
    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        for path in self.files:
            if self._cancelled: 
                break
            try:
                thumb = self.searcher._get_cached_thumbnail(path) if self.searcher else None
                if not thumb:
                    with Image.open(path) as img:
                        img = ImageOps.exif_transpose(img)
                        if img.mode != "RGB": img = img.convert("RGB")
                        img.thumbnail((200, 200)); thumb = img
                data = thumb.tobytes("raw", "RGB")
                self.thumbnail_loaded.emit(path, QImage(data, thumb.width, thumb.height, 3 * thumb.width, QImage.Format.Format_RGB888).copy())
            except Exception: pass

# --- Deduplication Worker & Dialog ---

class DupesWorker(QThread):
    chunk_ready = pyqtSignal(list)
    scan_complete = pyqtSignal(float, int)      
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, target_file: Optional[str] = None, scan_dir: Optional[str] = None, indexed_set: Optional[Set[str]] = None):
        super().__init__()
        self.target_file, self.scan_dir, self.indexed_set = target_file, scan_dir, indexed_set or set()
        self.process: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self.temp_dir: Optional[str] = None

    def stop(self) -> None:
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            try: self.process.terminate(); self.process.wait(timeout=2)
            except Exception:
                try: self.process.kill()
                except Exception: pass

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            try: shutil.rmtree(self.temp_dir)
            except Exception: pass
            self.temp_dir = None

    def run(self) -> None:
        start_time = time.time()
        try:
            files_to_check: List[str] = []
            if self.target_file and os.path.exists(self.target_file):
                files_to_check = list(self.indexed_set)
                print(f"[Dupes] Scanning entire index ({len(files_to_check)} files)...")
            elif self.scan_dir:
                real_scan_dir = os.path.realpath(self.scan_dir)
                if not real_scan_dir.endswith(os.sep): real_scan_dir += os.sep
                files_to_check = [p for p in self.indexed_set if p.startswith(real_scan_dir)]
                print(f"[Dupes] Scanning directory '{self.scan_dir}'. Found {len(files_to_check)} indexed files in scope.")
            
            if not files_to_check:
                duration = time.time() - start_time
                self.scan_complete.emit(duration, 0); return

            self.temp_dir = tempfile.mkdtemp(prefix="photofind_scan_")
            link_to_real: Dict[str, str] = {}
            
            print(f"[Dupes] Linking {len(files_to_check)} files to temporary directory...")
            self.status_update.emit(f"Linking {len(files_to_check)} files...")
            
            for i, fpath in enumerate(files_to_check):
                if self._stop_requested: self._cleanup_temp_dir(); self.cancelled.emit(); return
                
                if not os.path.exists(fpath):
                    continue
                    
                try:
                    link_path = os.path.join(self.temp_dir, f"{i:06d}_{os.path.basename(fpath)}")
                    os.symlink(fpath, link_path)
                    link_to_real[link_path] = fpath
                except Exception: pass
            
            cmd = ['jdupes', '-r', '-s', self.temp_dir]
            print(f"[Dupes] Running: {' '.join(cmd)}")
            self.status_update.emit("Running jdupes (this may take a while)...")
            
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            stdout_bytes, stderr_bytes = self.process.communicate()
            
            if self._stop_requested: self._cleanup_temp_dir(); self.cancelled.emit(); return

            stderr = stderr_bytes.decode('utf-8', errors='surrogateescape')
            if stderr.strip(): print(f"[Dupes] jdupes stderr:\n{stderr}")

            if self.process.returncode == 2:
                self._cleanup_temp_dir()
                duration = time.time() - start_time
                self.error.emit(f"jdupes error (code {self.process.returncode}):\n{stderr}"); return
            
            stdout = stdout_bytes.decode('utf-8', errors='surrogateescape')
            groups: List[List[str]] = []
            current_group: List[str] = []
            
            for line in stdout.splitlines():
                if line.strip() == "":
                    if len(current_group) > 1:
                        groups.append(current_group)
                        if len(groups) >= 50: self.chunk_ready.emit(groups); groups = []
                    current_group = []
                else:
                    real_path = link_to_real.get(line.strip())
                    if real_path and real_path in self.indexed_set: current_group.append(real_path)
            
            if len(current_group) > 1: groups.append(current_group)
            if groups: self.chunk_ready.emit(groups)
            
            duration = time.time() - start_time
            total_dupes = sum(len(g) for g in groups)
            self.scan_complete.emit(duration, total_dupes)
            
        except FileNotFoundError: 
            self.error.emit("'jdupes' is not installed.")
        except Exception as e:
            if not self._stop_requested: self.error.emit(f"Error: {str(e)}"); import traceback; traceback.print_exc()
        finally: self._cleanup_temp_dir()


class DuplicateDialog(QDialog):
    _thumb_signal = pyqtSignal(str, QImage)

    def __init__(self, worker: DupesWorker, searcher: PhotoSearch, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.searcher = searcher
        self.worker = worker
        self.all_groups: List[List[str]] = []
        self.page_offset = 0
        self.PAGE_SIZE = 50
        self.is_streaming = True
        self.current_group_index = -1
        
        self.setWindowTitle("Duplicate Manager (Scanning...)")
        self.resize(1050, 700)
        self._thumb_signal.connect(self.update_thumbnail)
        
        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.group_list = QListWidget()
        self.group_list.setMaximumWidth(250)
        self.group_list.currentRowChanged.connect(self.load_group_images)
        left_layout.addWidget(self.group_list)
        
        self.next_page_btn = QPushButton("Load Next 50 Sets...")
        self.next_page_btn.clicked.connect(self.load_next_page)
        self.next_page_btn.setEnabled(False)
        left_layout.addWidget(self.next_page_btn)
        content_layout.addLayout(left_layout)
        
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(200, 200))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Fixed)
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.currentItemChanged.connect(self._handle_dupes_image_selection)
        self.image_list.itemDoubleClicked.connect(self.open_in_explorer)
        self.image_list.itemSelectionChanged.connect(self.update_trash_btn_state)
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        content_layout.addWidget(self.image_list)
        
        btn_layout = QVBoxLayout()
        self.dup_folders_btn = QPushButton("Show Duplicate Folders")
        self.dup_folders_btn.clicked.connect(self.show_duplicate_folders)
        self.trash_btn = QPushButton("Move Selected to Trash")
        self.trash_btn.clicked.connect(self.trash_selected)
        self.trash_btn.setEnabled(False)
        self.trash_rest_btn = QPushButton("Keep First, Trash Rest")
        self.trash_rest_btn.clicked.connect(self.trash_rest)
        self.trash_rest_btn.setEnabled(False)
        
        btn_layout.addWidget(self.dup_folders_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.trash_rest_btn)
        btn_layout.addWidget(self.trash_btn)
        content_layout.addLayout(btn_layout)
        main_layout.addLayout(content_layout)
        
        bottom_area = QHBoxLayout()
        self.info_label = QLabel("Waiting for jdupes to find duplicates...")
        self.info_label.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.info_label.setMargin(5)
        
        self.stop_scan_btn = QPushButton("Stop Scan")
        self.stop_scan_btn.clicked.connect(self.stop_scan)
        self.stop_scan_btn.setVisible(True) 
        
        bottom_area.addWidget(self.info_label, 1)
        bottom_area.addWidget(self.stop_scan_btn, 0)
        main_layout.addLayout(bottom_area)

    def stop_scan(self):
        if self.worker:
            self.info_label.setText("Stopping...")
            self.worker.stop()

    def add_groups_chunk(self, groups: List[List[str]]) -> None:
        self.all_groups.extend(groups)
        if self.page_offset >= len(self.all_groups) - self.PAGE_SIZE - len(groups):
            self.page_offset = len(self.all_groups) - len(groups)
            self.render_current_page()
        self.update_next_button_state()

    def stream_finished(self, duration: float, total_dupes: int) -> None:
        self.is_streaming = False
        self.stop_scan_btn.setVisible(False)
        self.update_next_button_state()
        
        dur_str = format_duration(duration)
        count = len(self.all_groups)
        self.setWindowTitle(f"Duplicate Manager ({count} Total Sets Found)")
        self.info_label.setText(f"Scan complete ({dur_str}). Found {count} duplicate sets ({total_dupes} total files)." if count else f"Scan complete ({dur_str}). No duplicates found.")

    def update_status(self, msg: str) -> None:
        self.info_label.setText(msg)

    def update_next_button_state(self) -> None:
        has_more = (self.page_offset + self.PAGE_SIZE) < len(self.all_groups)
        self.next_page_btn.setEnabled(has_more or self.is_streaming)
        self.next_page_btn.setVisible(has_more or self.is_streaming)

    def load_next_page(self) -> None:
        self.page_offset += self.PAGE_SIZE
        self.render_current_page()

    def render_current_page(self) -> None:
        self.group_list.clear(); self.image_list.clear()
        self.trash_btn.setEnabled(False); self.trash_rest_btn.setEnabled(False)
        self.info_label.setText("Select a set on the left")
        page_groups = self.all_groups[self.page_offset:self.page_offset + self.PAGE_SIZE]
        for i, group in enumerate(page_groups):
            self.group_list.addItem(QListWidgetItem(f"Set {self.page_offset + i+1}: {len(group)} files"))
        if page_groups: self.group_list.setCurrentRow(0)
        self.update_next_button_state()

    def update_trash_btn_state(self) -> None:
        self.trash_btn.setEnabled(len(self.image_list.selectedItems()) > 0)

    def load_group_images(self, list_row: int) -> None:
        self.image_list.clear()
        self.trash_btn.setEnabled(False); self.trash_rest_btn.setEnabled(False)
        self.info_label.setText("Select an image to see details")
        actual_index = self.page_offset + list_row
        if actual_index < 0 or actual_index >= len(self.all_groups): return
        self.current_group_index = actual_index
        group = self.all_groups[actual_index]
        
        for i, path in enumerate(group):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(200, 200))
            label = os.path.basename(path) + ("\n(Original)" if i == 0 else "")
            item.setToolTip(label)
            self.image_list.addItem(item)
            threading.Thread(target=self._load_thumb, args=(path,), daemon=True).start()
        if len(group) > 1: self.trash_rest_btn.setEnabled(True)

    def _handle_dupes_image_selection(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current:
            self.show_image_info(current)

    def show_image_info(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            size_bytes = os.path.getsize(path)
            size_str = f"{size_bytes} B" if size_bytes < 1024 else (f"{size_bytes / 1024:.2f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB")
            self.info_label.setText(f"{path}  |  Size: {size_str}")
        except Exception as e:
            self.info_label.setText(f"{path}  |  Error: {e}")

    def _load_thumb(self, path: str) -> None:
        try:
            thumb = self.searcher._get_cached_thumbnail(path)
            if not thumb:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != "RGB": img = img.convert("RGB")
                    img.thumbnail((200, 200)); thumb = img
            data = thumb.tobytes("raw", "RGB")
            self._thumb_signal.emit(path, QImage(data, thumb.width, thumb.height, 3 * thumb.width, QImage.Format.Format_RGB888).copy())
        except Exception: pass

    def update_thumbnail(self, path: str, q_img: QImage) -> None:
        pixmap = QPixmap.fromImage(q_img)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path: item.setIcon(QIcon(pixmap)); break

    def open_in_explorer(self, path: str) -> None:
        try:
            # DEVNULL prevents file browsers from spamming your terminal
            if os.path.exists("/usr/bin/dolphin"): 
                subprocess.Popen(['dolphin', '--select', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif os.path.exists("/usr/bin/nautilus"): 
                subprocess.Popen(['nautilus', '--select', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else: 
                subprocess.Popen(['xdg-open', os.path.dirname(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception: 
            pass

    def open_image_viewer(self, path: str) -> None:
        if not os.path.exists(path): return
        # Launch image viewer detached from terminal to suppress stderr spam
        try:
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def show_image_context_menu(self, position: int) -> None:
        item = self.image_list.itemAt(position)
        if not item: return
        if not item.isSelected(): self.image_list.clearSelection(); item.setSelected(True)
        selected_items = self.image_list.selectedItems()
        if not selected_items: return
            
        menu = QMenu()
        open_action = menu.addAction("Open in File Browser")
        open_action.triggered.connect(lambda: self.open_in_explorer(selected_items[0]))
        menu.addSeparator()
        
        delete_action = menu.addAction(f"Move {len(selected_items)} Photos to Trash")
        paths_to_delete = [i.data(Qt.ItemDataRole.UserRole) for i in selected_items]
        items_to_remove = selected_items
        delete_action.triggered.connect(lambda: self._delete_paths(paths_to_delete, items_to_remove))
        menu.exec(self.image_list.viewport().mapToGlobal(position))

    def trash_selected(self) -> None:
        sel = self.image_list.selectedItems()
        if sel: self._delete_paths([i.data(Qt.ItemDataRole.UserRole) for i in sel], sel)

    def trash_rest(self) -> None:
        if self.current_group_index < 0: return
        paths_to_delete = self.all_groups[self.current_group_index][1:]
        items_to_remove = [i for i in [self.image_list.item(r) for r in range(self.image_list.count())] if i.data(Qt.ItemDataRole.UserRole) in paths_to_delete]
        self._delete_paths(paths_to_delete, items_to_remove)

    def purge_file_from_all_groups(self, file_path: str) -> None:
        new_all_groups, changed = [], False
        for group in self.all_groups:
            if file_path in group: changed = True; group.remove(file_path)
            if len(group) > 1: new_all_groups.append(group)
        if changed:
            self.all_groups = new_all_groups
            self.page_offset = min(self.page_offset, max(0, len(self.all_groups) - self.PAGE_SIZE))
            self.render_current_page()
            self.setWindowTitle(f"Duplicate Manager ({len(self.all_groups)} Total Sets Found)")

    def _delete_paths(self, paths_to_delete: List[str], items_to_remove: List[QListWidgetItem]) -> None:
        success_paths = []
        fail_paths = []
        for p in paths_to_delete:
            if QFile.moveToTrash(p):
                self.searcher.mark_photo_deleted(p)
                success_paths.append(p)
            else:
                fail_paths.append(p)
                
        for p in success_paths: self.purge_file_from_all_groups(p)
        if fail_paths: QMessageBox.warning(self, "Error", f"Could not trash {len(fail_paths)} files.")

    def _open_in_explorer(self, path: str) -> None:
        try:
            if os.path.exists("/usr/bin/dolphin"): subprocess.Popen(['dolphin', '--select', path])
            elif os.path.exists("/usr/bin/nautilus"): subprocess.Popen(['nautilus', '--select', path])
            else: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception: 
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def show_duplicate_folders(self) -> None:
        dir_pairs: Dict[Tuple[str, str], int] = {}
        for group in self.all_groups:
            dirs = list({os.path.dirname(f) for f in group})
            if len(dirs) > 1:
                for i in range(len(dirs)):
                    for j in range(i + 1, len(dirs)):
                        pair = tuple(sorted([dirs[i], dirs[j]]))
                        dir_pairs[pair] = dir_pairs.get(pair, 0) + 1
        
        if not dir_pairs: 
            QMessageBox.information(self, "Duplicate Folders", "No overlapping duplicate folders found."); 
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Duplicate Folders Analysis")
        dialog.resize(700, 500)
        
        layout = QVBoxLayout(dialog)
        info_label = QLabel("Double-click a row or use the button below to open the folder in your file browser.")
        layout.addWidget(info_label)
        
        list_widget = QListWidget()
        for (d1, d2), count in sorted(dir_pairs.items(), key=lambda x: x[1], reverse=True)[:100]:
            item_text = f"[{count} files]\n  {d1}\n  \u2194\n  {d2}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, (d1, d2))
            list_widget.addItem(item)
            
        list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        def on_context_menu(position: int) -> None:
            item = list_widget.itemAt(position)
            if not item: return
            d1, d2 = item.data(Qt.ItemDataRole.UserRole)
            menu = QMenu(list_widget)
            menu.addAction(f"Open: {d1}").triggered.connect(lambda _, p=d1: self._open_in_explorer(p))
            menu.addAction(f"Open: {d2}").triggered.connect(lambda _, p=d2: self._open_in_explorer(p))
            menu.exec(list_widget.viewport().mapToGlobal(position))

        list_widget.customContextMenuRequested.connect(on_context_menu)
        list_widget.itemDoubleClicked.connect(lambda item: self._open_in_explorer(item.data(Qt.ItemDataRole.UserRole)[0]))
        
        layout.addWidget(list_widget)
        
        btn_layout = QHBoxLayout()
        open_btn = QPushButton("Open Selected Folder")
        
        def on_open_clicked() -> None:
            item = list_widget.currentItem()
            if not item: return
            d1, d2 = item.data(Qt.ItemDataRole.UserRole)
            menu = QMenu(open_btn)
            menu.addAction(f"Open: {d1}").triggered.connect(lambda _, p=d1: self._open_in_explorer(p))
            menu.addAction(f"Open: {d2}").triggered.connect(lambda _, p=d2: self._open_in_explorer(p))
            menu.exec(open_btn.mapToGlobal(QPoint(0, open_btn.height())))
                
        open_btn.clicked.connect(on_open_clicked)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        
        btn_layout.addStretch()
        btn_layout.addWidget(open_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()


# --- GUI Application ---

class PhotoOrganizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoFind")
        self.resize(1200, 800)
        
        self.has_jdupes = shutil.which('jdupes') is not None
        if not self.has_jdupes: print("[Info] 'jdupes' not found. Duplicate detection disabled.")
        
        self.searcher = PhotoSearch()
        self.current_results: List[Dict[str, Any]] = []
        self.thumb_worker: Optional[ThumbnailWorker] = None
        self.dupes_worker: Optional[DupesWorker] = None
        self.dupes_dialog: Optional[DuplicateDialog] = None 
        self.index_worker: Optional[IndexWorker] = None
        self.search_worker: Optional[SearchWorker] = None
        self.garbage_worker: Optional[GarbageWorker] = None
        self.clean_worker: Optional[CleanWorker] = None
        self.search_history: List[str] = []
        
        self.setup_ui(); self.setup_menu(); self.setup_shortcuts(); self.setup_search_history()
        
        if self.searcher.load_index():
            self.statusBar().showMessage(self._get_idle_status_message())

    def _get_idle_status_message(self) -> str:
        """Builds the permanent 'Ready...' string with historical stats."""
        parts = [f"Ready. {len(self.searcher.indexed_set)} images indexed."]
        
        stats = self.searcher.stats
        if "last_index_count" in stats:
            parts.append(f"(last scan {stats['last_index_count']} images took {stats['last_index_duration']})")
            
        if "last_dedupe_count" in stats:
            parts.append(f"(last dedupe {stats['last_dedupe_count']} files took {stats['last_dedupe_duration']})")
            
        return " ".join(parts)

    def _show_temp_status(self, message: str, duration_ms: int = 5000) -> None:
        """Interrupts status bar, then safely falls back to permanent stats."""
        self.statusBar().showMessage(message)
        QTimer.singleShot(duration_ms, lambda: self.statusBar().showMessage(self._get_idle_status_message()))

    def closeEvent(self, event):
        """Gracefully shut down all background threads before closing the window."""
        try:
            if self.index_worker and self.index_worker.isRunning():
                self.index_worker.cancel()
            if self.dupes_worker and self.dupes_worker.isRunning():
                self.dupes_worker.stop()
        except RuntimeError:
            pass
            
        if self.dupes_dialog and self.dupes_dialog.isVisible():
            self.dupes_dialog.close()
            
        workers = [
            self.index_worker, self.search_worker, 
            self.garbage_worker, self.clean_worker, self.dupes_worker
        ]
        
        for w in workers:
            if w is not None:
                try:
                    if w.isRunning():
                        w.wait(2000)  
                        if w.isRunning():
                            w.terminate()  
                            w.wait()
                except RuntimeError:
                    pass 
                    
        try:
            if self.thumb_worker and self.thumb_worker.isRunning():
                self.thumb_worker.cancel()
                self.thumb_worker.wait(1000)
        except RuntimeError:
            pass

        # Critical: Flush pending deleteLater() signals to unlock Qt's internal mutexes
        QApplication.processEvents()
        QApplication.processEvents()
            
        event.accept()

    def setup_ui(self) -> None:
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Type a description..."); self.search_input.returnPressed.connect(self.start_search)
        search_btn = QPushButton("Search"); search_btn.clicked.connect(self.start_search)
        garbage_btn = QPushButton("Find Bad Photos"); garbage_btn.clicked.connect(self.find_garbage)
        search_layout.addWidget(self.search_input); search_layout.addWidget(search_btn); search_layout.addWidget(garbage_btn)
        layout.addLayout(search_layout)
        
        bottom_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setTextVisible(True)
        self.stop_index_btn = QPushButton("Stop Indexing"); self.stop_index_btn.setVisible(False); self.stop_index_btn.clicked.connect(self.cancel_current_operation)
        bottom_layout.addWidget(self.progress_bar, 1)
        bottom_layout.addWidget(self.stop_index_btn, 0)
        layout.addLayout(bottom_layout)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode); self.list_widget.setIconSize(QSize(200, 200))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust); self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.itemDoubleClicked.connect(self.open_image)
        self.list_widget.currentItemChanged.connect(self.show_path_in_status)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.list_widget)
        self.setStatusBar(QStatusBar())

    def setup_menu(self) -> None:
        toolbar = QToolBar("Main"); self.addToolBar(toolbar)
        index_action = QAction("Index Folder", self); index_action.triggered.connect(self.select_folder_to_index); toolbar.addAction(index_action)

        self.dupes_action = QAction("Find Duplicates", self)
        self.dupes_action.setToolTip("Scan directories for exact byte-for-byte duplicates"); self.dupes_action.triggered.connect(self.start_global_dedupe)
        if not self.has_jdupes: self.dupes_action.setEnabled(False); self.dupes_action.setToolTip("jdupes is not installed")
        toolbar.addAction(self.dupes_action)

        clean_action = QAction("Clean Database", self); clean_action.setToolTip("Remove entries for files that no longer exist")
        clean_action.triggered.connect(self.clean_database); toolbar.addAction(clean_action)

    def setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+F"), self, self.search_input.setFocus)
        QShortcut(QKeySequence("Ctrl+A"), self, self.list_widget.selectAll)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence("Escape"), self, self.cancel_current_operation)

    def setup_search_history(self) -> None:
        self.search_history = self.searcher.load_search_history()
        if self.search_history:
            self.search_completer = QCompleter(self.search_history, self.search_input)
            self.search_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self.search_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self.search_input.setCompleter(self.search_completer)

    def _add_to_search_history(self, query: str) -> None:
        if not query or query in self.search_history: return
        self.search_history.append(query)
        self.searcher.save_search_history(self.search_history)
        
        self.search_completer = QCompleter(self.search_history, self.search_input)
        self.search_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.search_input.setCompleter(self.search_completer)

    def _format_file_size(self, path: str) -> str:
        try:
            size_bytes = os.path.getsize(path)
            return f"{size_bytes} B" if size_bytes < 1024 else (f"{size_bytes / 1024:.2f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB")
        except Exception: return "Unknown size"

    def delete_selected(self) -> None:
        sel = self.list_widget.selectedItems()
        if sel: self.delete_photos_action([i.data(Qt.ItemDataRole.UserRole) for i in sel], sel)

    def cancel_current_operation(self) -> None:
        try:
            if self.index_worker and self.index_worker.isRunning(): 
                self.index_worker.cancel(); 
                self.statusBar().showMessage("Cancelling indexing...")
            elif self.dupes_worker and self.dupes_worker.isRunning(): 
                if self.dupes_dialog:
                    self.dupes_dialog.stop_scan()
        except RuntimeError:
            pass

    def clean_database(self) -> None:
        if QMessageBox.question(self, 'Clean Database', 'Remove entries for files that no longer exist on disk?\n\nContinue?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Cleaning database...")
            self.clean_worker = CleanWorker(self.searcher)
            self.clean_worker.clean_complete.connect(self.clean_finished)
            self.clean_worker.finished.connect(self.clean_worker.deleteLater)
            self.clean_worker.start()

    def clean_finished(self, removed_count: int) -> None:
        self._show_temp_status(f"Clean complete. Removed {removed_count} missing files.")
        self.searcher.load_index()

    def show_context_menu(self, position: int) -> None:
        item = self.list_widget.itemAt(position)
        if not item: return
        if not item.isSelected(): self.list_widget.clearSelection(); item.setSelected(True)
        selected_items = self.list_widget.selectedItems(); selected_count = len(selected_items)
        menu = QMenu()
        
        if selected_count == 1:
            path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            open_action = menu.addAction("Open in Image Viewer")
            open_action.triggered.connect(lambda: self.open_image_viewer(path))
            
            open_folder_action = menu.addAction("Open in File Browser")
            open_folder_action.triggered.connect(lambda: self.open_in_explorer(path))
            
            menu.addSeparator()

        delete_action = menu.addAction(f"Move {selected_count} Photos to Trash" if selected_count > 1 else "Move to Trash")
        delete_action.triggered.connect(lambda: self.delete_photos_action([i.data(Qt.ItemDataRole.UserRole) for i in selected_items], selected_items))
        menu.exec(self.list_widget.viewport().mapToGlobal(position))

    def delete_photos_action(self, paths: List[str], items: List[QListWidgetItem]) -> None:
        count = len(paths)
        if count == 0: return
        msg = f'Move this file to Trash?\n{paths[0]}' if count == 1 else f'Move {count} selected files to Trash?'
        if QMessageBox.question(self, 'Move to Trash', msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            success, fail = [], []
            for p in paths:
                if QFile.moveToTrash(p): self.searcher.mark_photo_deleted(p); success.append(p)
                else: fail.append(p)
            for item in reversed([i for i in items if i.data(Qt.ItemDataRole.UserRole) in success]): self.list_widget.takeItem(self.list_widget.row(item))
            if fail: QMessageBox.warning(self, "Error", f"Could not move {len(fail)} files to trash.")
            else: self._show_temp_status(f"Moved {count} photos to Trash.")

    def open_in_explorer(self, path: str) -> None:
        try:
            if os.path.exists("/usr/bin/dolphin"): subprocess.Popen(['dolphin', '--select', path])
            elif os.path.exists("/usr/bin/nautilus"): subprocess.Popen(['nautilus', '--select', path])
            else: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        except Exception: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def open_image_viewer(self, path: str) -> None:
        if not os.path.exists(path): return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def start_global_dedupe(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Root Photo Folder to Scan")
        if folder:
            self.statusBar().showMessage("Running jdupes..."); self.dupes_action.setEnabled(False)
            self.dupes_worker = DupesWorker(scan_dir=folder, indexed_set=self.searcher.indexed_set)
            self.dupes_dialog = DuplicateDialog(self.dupes_worker, self.searcher, self); self.dupes_dialog.show()
            
            self.dupes_worker.chunk_ready.connect(self.dupes_dialog.add_groups_chunk)
            self.dupes_worker.scan_complete.connect(self.dupes_finished)
            self.dupes_worker.error.connect(self.dupes_error)
            self.dupes_worker.cancelled.connect(self.dupes_cancelled)
            self.dupes_worker.status_update.connect(self.dupes_dialog.update_status)
            
            self.dupes_worker.finished.connect(self.dupes_worker.deleteLater)
            
            self.dupes_worker.start()

    def dupes_finished(self, duration: float, total_dupes: int, message_override: Optional[str] = None) -> None:
        if self.dupes_dialog: self.dupes_dialog.stream_finished(duration, total_dupes)
        self.reset_dupes_ui_state()
        
        self.searcher.save_dedupe_stats(duration, total_dupes)
        
        if message_override:
            self._show_temp_status(message_override)
        else:
            self.statusBar().showMessage(self._get_idle_status_message())

    def dupes_cancelled(self) -> None:
        self.dupes_finished(0.0, 0, message_override="Duplicate scan cancelled.")

    def dupes_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Deduplication Error", msg); 
        if self.dupes_dialog: self.dupes_dialog.close()
        self.reset_dupes_ui_state()
        self._show_temp_status("Duplicate scan failed.")

    def reset_dupes_ui_state(self) -> None:
        self.dupes_action.setEnabled(self.has_jdupes); 
        self.dupes_dialog = None

    def select_folder_to_index(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.statusBar().showMessage(f"Indexing {folder}..."); 
            self.progress_bar.setVisible(True); self.stop_index_btn.setVisible(True)
            self.progress_bar.setRange(0, 0); self.progress_bar.setFormat("Scanning...")
            self.index_worker = IndexWorker(self.searcher, folder)
            self.index_worker.progress.connect(self.indexing_progress)
            self.index_worker.result_ready.connect(self.indexing_finished)
            self.index_worker.finished.connect(self.index_worker.deleteLater)
            self.index_worker.start()

    def indexing_progress(self, msg: str, current: int, total: int) -> None:
        self.progress_bar.setRange(0, total); self.progress_bar.setValue(current); self.progress_bar.setFormat(f"{msg} (%p%)"); self.statusBar().showMessage(msg)

    def indexing_finished(self, count: int, duration: float) -> None:
        self.progress_bar.setVisible(False)
        self.stop_index_btn.setVisible(False)
        
        dur_str = format_duration(duration)
        total_count = len(self.searcher.indexed_set)
        
        self.searcher.save_index_stats(duration, count)
        self.statusBar().showMessage(self._get_idle_status_message())

    def start_search(self) -> None:
        query = self.search_input.text().strip()
        if not query: return
        self._add_to_search_history(query); self.list_widget.clear(); self.statusBar().showMessage("Searching...")
        self.search_worker = SearchWorker(self.searcher, query)
        self.search_worker.result.connect(self.display_results)
        self.search_worker.finished.connect(self.search_worker.deleteLater)
        self.search_worker.start()

    def find_garbage(self) -> None:
        self.list_widget.clear(); self.statusBar().showMessage("Scanning metadata for bad photos...")
        self.garbage_worker = GarbageWorker(self.searcher)
        self.garbage_worker.result.connect(self.display_results)
        self.garbage_worker.finished.connect(self.garbage_worker.deleteLater)
        self.garbage_worker.start()

    def display_results(self, hits: List[Dict[str, Any]]) -> None:
        self.list_widget.clear(); self.current_results = hits
        if not hits: 
            self._show_temp_status("No results found.")
            return
        self.statusBar().showMessage(f"Found {len(hits)} results. Loading thumbnails...")
        paths_to_load = []
        for hit in hits:
            path, score = hit['file'], hit.get('score', 0.0)
            item = QListWidgetItem(); item.setData(Qt.ItemDataRole.UserRole, path); item.setSizeHint(QSize(200, 200))
            tooltip = f"{os.path.basename(path)}\nSize: {self._format_file_size(path)}\nScore: {score:.3f}"
            if hit.get('is_garbage'): tooltip += "\n[LOW QUALITY]"
            item.setToolTip(tooltip); self.list_widget.addItem(item); paths_to_load.append(path)
            
        # FIX: Catch the race condition where deleteLater already killed the C++ object
        try:
            if self.thumb_worker and self.thumb_worker.isRunning(): 
                self.thumb_worker.cancel()
                self.thumb_worker.wait(500)
        except RuntimeError:
            pass  # Object was already deleted by Qt, safe to ignore and proceed
            
        self.thumb_worker = ThumbnailWorker(paths_to_load, self.searcher)
        self.thumb_worker.thumbnail_loaded.connect(self.update_thumbnail)
        self.thumb_worker.finished.connect(lambda: self.statusBar().showMessage(self._get_idle_status_message()))
        self.thumb_worker.finished.connect(self.thumb_worker.deleteLater)
        self.thumb_worker.start()

    def update_thumbnail(self, path: str, q_img: QImage) -> None:
        pixmap = QPixmap.fromImage(q_img)
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == path: self.list_widget.item(i).setIcon(QIcon(pixmap)); return

    def show_path_in_status(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if not current:
            self.statusBar().showMessage("Ready")
            return
            
        selected_count = len(self.list_widget.selectedItems())
        if selected_count > 1:
            self.statusBar().showMessage(f"{selected_count} items selected")
        else:
            path = current.data(Qt.ItemDataRole.UserRole)
            self.statusBar().showMessage(f"{path}  |  Size: {self._format_file_size(path)}")

    def open_image(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        self.open_image_viewer(path)

# --- Main Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Photo Search")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Clear index")
    parser.add_argument("--top", type=int, default=5, help="Results count")
    parser.add_argument("--find-garbage", action="store_true", help="Find low quality images")
    args, _ = parser.parse_known_args()

    cache_path = Path.home() / ".cache" / "photofind"
    if args.index or args.search or args.find_garbage:
        searcher = PhotoSearch()
        if args.reindex:
            for p in [cache_path / "photo_index.json", cache_path / "photo_embeddings.pt", cache_path / "photo_metadata.json"]:
                if p.exists(): p.unlink()
        searcher.load_index()
        if args.index: searcher.index_photos(args.index)
        elif args.find_garbage:
            hits = searcher.get_garbage_photos(); print(f"\nFound {len(hits)} low quality images:")
            for h in hits: print(f"[Garbage] {h['file']}")
        elif args.search:
            hits = searcher.search(args.search, top_k=args.top); print(f"\nTop {args.top} results for '{args.search}':")
            for h in hits: print(f"[Score: {h['score']:.3f}] {h['file']}")
    else:
        app = QApplication(sys.argv)
        if args.reindex:
            for p in [cache_path / "photo_index.json", cache_path / "photo_embeddings.pt", cache_path / "photo_metadata.json"]:
                if p.exists(): p.unlink()
        window = PhotoOrganizerWindow(); window.show(); ret = app.exec(); os._exit(ret)
