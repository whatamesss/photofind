#!/usr/bin/env python
"""
PhotoFind: Semantic CLIP Image Search, Duplicate Detection, and Quality Sorting
Requirements: torch, transformers, PyQt6, Pillow, jdupes (optional)
"""

import os
# Configure CUDA memory allocator before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Silence benign KDE/KIO noise emitted by the native (KDE) folder chooser:
# "kf.kio.widgets.kdirmodel: No node found for item that was just removed".
# Must be set before Qt/KIO initialize their logging rules.
os.environ.setdefault("QT_LOGGING_RULES", "kf.kio.widgets.kdirmodel=false")

import sys
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
import random
import logging
import warnings
import re
import shlex
import configparser
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple, Callable
from PIL import Image, ImageOps

# Transformers & PyTorch
from transformers import CLIPProcessor, CLIPModel

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QListWidgetItem,
    QVBoxLayout, QWidget, QLineEdit, QLabel, QStatusBar,
    QFileDialog, QMessageBox, QToolBar, QHBoxLayout, QPushButton,
    QMenu, QDialog, QAbstractItemView, QFrame, QProgressBar,
    QToolButton, QTabWidget, QDoubleSpinBox, QSpinBox, QFormLayout,
    QGroupBox, QDialogButtonBox, QComboBox, QSplashScreen
)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction, QKeySequence, QShortcut, QDesktopServices
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QFile, QPoint, QTimer, QLibraryInfo, QLoggingCategory

# Suppress SDPA warnings on older GPUs (e.g., GTX 1050Ti)
warnings.filterwarnings("ignore", message="Torch was not compiled with memory efficient attention")
warnings.filterwarnings("ignore", message="1Torch was not compiled with memory efficient attention")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# --- Configuration ---
# Default search-accuracy and quality thresholds. Editable at runtime via the
# "Search Tuning" dialog and persisted to ~/.cache/photofind/settings.json.
DEFAULT_SEARCH_SETTINGS: Dict[str, Any] = {
    "top_k": 60,            # max results returned
    "floor_min": 0.15,      # absolute cosine-similarity floor
    "floor_ratio": 0.45,    # required fraction of the best hit's score
    "dedupe_cos": 0.99,     # suppress results more similar than this to a kept hit
    "candidate_mult": 40,   # candidate pool multiplier (advanced)
    "black_lum_p95": 10.0,  # hide near-black photos at/below this luminance
}

DEFAULT_VIEWER_SETTINGS: Dict[str, Any] = {
    "image_viewer": "gwenview",  # "system" = desktop default; else a command line
}

# Bumped when the default viewer changed from "system" to "gwenview"; used to
# migrate settings files written before then.
SETTINGS_VERSION = 2

# Geeqie always launches windowed unless told otherwise; add its fullscreen
# flag. PhotoFind also sets GQ_NEW_INSTANCE so each launch is a fresh window
# (see _launch_image_viewer).
GEEQIE_EXTRA_ARGS: List[str] = ["--fullscreen"]


# Launched viewers are kept so we can reap them; when a viewer closes it would
# otherwise linger as a defunct/zombie child, since nothing calls wait() on it.
_VIEWER_PROCESSES: List["subprocess.Popen"] = []


def reap_viewer_processes() -> None:
    alive = []
    for proc in _VIEWER_PROCESSES:
        try:
            if proc.poll() is None:
                alive.append(proc)
        except Exception:
            pass
    _VIEWER_PROCESSES[:] = alive


# --- Dynamic image-viewer discovery ---------------------------------------
# We discover installed applications from freedesktop .desktop entries rather
# than hard-coding a list, so anything registered to open image/* shows up.

_DESKTOP_FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


def _application_dirs() -> List[str]:
    """Standard directories that hold .desktop application entries."""
    dirs: List[str] = []
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    dirs.append(os.path.join(data_home, "applications"))
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    for base in data_dirs.split(":"):
        if base:
            dirs.append(os.path.join(base, "applications"))
    dirs += [
        os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
        "/var/lib/flatpak/exports/share/applications",
        "/var/lib/snapd/desktop/applications",
    ]
    seen: Set[str] = set(); out: List[str] = []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def _clean_desktop_exec(exec_line: str) -> str:
    """Turn a .desktop Exec line into a launch command (field codes removed).

    Drops leading ``env``/``VAR=value`` wrappers so the result starts with the
    real executable. The path to open is appended by the caller.
    """
    parts = shlex.split(_DESKTOP_FIELD_CODES.sub(" ", exec_line))
    if parts and os.path.basename(parts[0]) == "env":
        parts = parts[1:]
        while parts and "=" in parts[0]:
            parts = parts[1:]
    return " ".join(parts)


def _find_executable(command: str) -> bool:
    """True if the command's executable can be resolved (PATH or absolute)."""
    try:
        exe = shlex.split(command)[0]
    except (ValueError, IndexError):
        return False
    return bool(shutil.which(exe) or (os.path.isabs(exe) and os.path.exists(exe)))


# Discovery is dynamic (only what is installed is listed), but restricted to
# this set so generic image handlers (web browsers, office/drawing apps) that
# merely register for image/* do not clutter the menu.
_PHOTO_VIEWER_NAMES: Set[str] = {
    # photo viewers
    "gwenview", "geeqie", "gqview", "eog", "eom", "ristretto", "feh", "nomacs",
    "mirage", "viewnior", "gpicview", "gthumb", "shotwell", "pix", "xviewer",
    "lximage-qt", "qiv", "sxiv", "nsxiv", "loupe", "fim", "pqiv", "photoqt",
    # editors, RAW processors and organizers
    "gimp", "gimp-2.10", "gimp-2.99", "gimp-3.0", "krita", "darktable",
    "rawtherapee", "luminance-hdr", "digikam", "showfoto",
}

# Preferred order in the toolbar menu; anything else sorts after these.
_VIEWER_PREFERENCE: List[str] = ["gwenview", "geeqie", "gimp"]


def _photo_viewer_name(command: str) -> Optional[str]:
    """Return the allow-listed viewer name for ``command``, or None.

    Matches the executable's base name. For launchers such as Flatpak the app
    id's final component is used (``flatpak run org.gnome.Loupe`` -> ``loupe``).
    """
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    exe = os.path.basename(parts[0]).lower()
    if exe in _PHOTO_VIEWER_NAMES:
        return exe
    if exe in ("flatpak", "snap"):
        for tok in parts[1:]:
            if "." in tok:
                candidate = tok.rsplit(".", 1)[-1].lower()
                if candidate in _PHOTO_VIEWER_NAMES:
                    return candidate
    return None


def _viewer_sort_key(name: str, label: str, command: str) -> Tuple[int, str, str]:
    rank = len(_VIEWER_PREFERENCE)
    for i, pref in enumerate(_VIEWER_PREFERENCE):
        if name == pref or name.startswith(pref):
            rank = i
            break
    return (rank, label.lower(), command)


def detect_image_viewers() -> List[Tuple[str, str]]:
    """Return (command, label) for installed photo viewers/editors.

    Derived from freedesktop .desktop entries, newest-namespace entries win;
    entries are restricted to known photo apps (see _PHOTO_VIEWER_NAMES).
    """
    found: Dict[str, Tuple[str, str]] = {}
    for d in _application_dirs():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for fn in names:
            if not fn.endswith(".desktop"): continue
            try:
                cp = configparser.ConfigParser(interpolation=None, strict=False)
                cp.read(os.path.join(d, fn), encoding="utf-8")
                sec = cp["Desktop Entry"]
            except (configparser.Error, KeyError, OSError, UnicodeDecodeError):
                continue
            if sec.get("Type", "Application") != "Application": continue
            if sec.getboolean("Hidden", fallback=False) or sec.getboolean("NoDisplay", fallback=False):
                continue
            mimes = sec.get("MimeType", "")
            if not any(tok.startswith("image/") for tok in mimes.split(";")):
                continue
            exec_line = sec.get("Exec")
            if not exec_line: continue
            command = _clean_desktop_exec(exec_line)
            if not command or not _find_executable(command): continue
            viewer_name = _photo_viewer_name(command)
            if not viewer_name: continue
            found.setdefault(command, (sec.get("Name") or shlex.split(command)[0], viewer_name))
    return [(cmd, label) for cmd, (label, _name) in
            sorted(found.items(), key=lambda kv: _viewer_sort_key(kv[1][1], kv[1][0], kv[0]))]

DEFAULT_QUALITY_SETTINGS: Dict[str, Any] = {
    "dark_lum_p95": 25.0, "dark_span": 25.0,
    # Blur is measured by the variance of the Laplacian (lap_var); images with
    # almost no edges (edge_score <= blur_min_edge) are featureless and are not
    # judged for blur at all.
    "blur_lap_var": 20.0, "blur_span": 30.0, "blur_min_edge": 2.0,
    # Blank/solid-colour frames: near-zero pixel variation (std_dev).
    "blank_std_max": 2.5, "blank_span": 2.5,
    "over_lum_p95": 235.0, "over_lum_span": 20.0,
    "over_mean_lum": 200.0, "over_mean_span": 55.0,
    "folder_mirror_coverage": 0.90,
}

# Directories that are poor places to "keep" a duplicate copy from.
DUPE_JUNK_FRAGMENTS = ("/downloads/", "/tmp/", "/cache/", "/trash/", "/.trash")

INDEX_EXCLUDE_FRAGMENTS = ("/www.", "/html/", "/textures/", "/res/", "nopic", "/screenshots/", "/downloads/", "/cache/", "/tmp/", "cake_wallet")
VRAM_TARGET_RATIO = 0.90
MAX_BATCH_SIZE = 512
VRAM_RESERVE_MB = 512

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def format_duration(seconds: float) -> str:
    if seconds < 60: return f"{int(seconds)}s"
    mins = int(seconds // 60); secs = int(seconds % 60)
    if mins < 60: return f"{mins}m {secs}s"
    hours = int(mins // 60); mins = int(mins % 60)
    return f"{hours}h {mins}m"

def open_in_file_browser(path: str) -> None:
    """
    Robust file opener that uses DBus to avoid Qt6.10 segfaults on KDE.
    Keeps the 'Select File' functionality without crashing the desktop.
    """
    real_path = os.path.realpath(path)
    file_uri = QUrl.fromLocalFile(real_path).toString()

    # Try DBus first (Standard, stable, supports selection)
    try:
        import dbus
        bus = dbus.SessionBus()
        try:
            # Standard FreeDesktop interface used by Dolphin/KDE
            interface = dbus.Interface(
                bus.get_object('org.freedesktop.FileManager1', '/org/freedesktop/FileManager1'),
                'org.freedesktop.FileManager1'
            )
            # This calls Dolphin's DBus method directly.
            interface.ShowItems([file_uri], "")
            logging.info(f"Opened via DBus: {file_uri}")
            return
        except dbus.exceptions.DBusException:
            pass
    except ImportError:
        pass

    # Fallback: Use Qt's internal desktop services (Opens folder, no selection)
    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(real_path)))
    except Exception:
        pass

def _pil_image_to_qimage(img: Image.Image) -> QImage:
    """Copy a Pillow image into a standalone RGBA QImage.

    RGBA is used unconditionally: it keeps every scanline 4-byte aligned (a
    QImage requirement) and preserves transparency. ``.copy()`` detaches the
    result from the temporary Python buffer.
    """
    if img.mode != "RGBA": img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, img.width * 4,
                  QImage.Format.Format_RGBA8888).copy()

def load_image_for_clipboard(path: str) -> QImage:
    """Decode an image the way it is displayed (EXIF orientation applied)."""
    with Image.open(path) as opened:
        opened.load()
        return _pil_image_to_qimage(ImageOps.exif_transpose(opened))

def copy_image_to_clipboard(path: str) -> Optional[str]:
    """Copy the pixels of the image at ``path`` to the system clipboard.

    Returns None on success, or a human-readable error message on failure.
    Unlike putting a file URL on the clipboard, this lets the user paste the
    actual picture into chat, document, and image-editing apps (the feature
    Geeqie has and Gwenview lacks). Must be called from the GUI thread.
    """
    try:
        qimg = load_image_for_clipboard(path)
    except Exception as e:
        logging.warning(f"Failed to load image for clipboard {path}: {type(e).__name__}: {e}")
        return f"Could not read image:\n{e}"
    if qimg.isNull(): return "Could not decode image."
    QApplication.clipboard().setImage(qimg)
    return None

def copy_image_with_ui(parent: QWidget, path: str, status_cb: Callable[[str], None]) -> None:
    """Copy ``path`` to the clipboard, showing a busy cursor and any error."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        err = copy_image_to_clipboard(path)
    finally:
        QApplication.restoreOverrideCursor()
    if err: QMessageBox.warning(parent, "Copy Image to Clipboard", err)
    else: status_cb(f"Copied {os.path.basename(path)} to the clipboard.")

def _numpy_to_clip_arrays(img: Image.Image) -> Tuple[np.ndarray, float, float, float, float, float, float]:
    arr_uint8 = np.array(img)
    gray = arr_uint8.mean(axis=2).astype(np.float32)
    std_dev = float(np.std(gray))
    dx = np.abs(gray[:, 2:] - gray[:, :-2])
    dy = np.abs(gray[2:, :] - gray[:-2, :])
    edge_score = float(np.mean(dx) + np.mean(dy))
    mean_lum = float(gray.mean()); lap = (-4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]); lap_var = float(lap.var()); lum_p95 = float(np.percentile(gray, 95)); lap_p95 = float(np.percentile(np.abs(lap), 95))
    # CPU float32 math (optimal for overlapping PCIe transfer)
    arr = arr_uint8.astype(np.float32) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return arr, std_dev, edge_score, mean_lum, lap_var, lum_p95, lap_p95

# --- Core Search Engine ---

class PhotoSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading CLIP model on {self.device}...")
        model_id = "openai/clip-vit-large-patch14"
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id, backend="torchvision", local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=True)
            logging.info("Model loaded successfully (Offline).")
        except OSError:
            logging.warning("Model files not found in cache. Downloading...")
            self.processor = CLIPProcessor.from_pretrained(model_id, backend="torchvision")
            self.model = CLIPModel.from_pretrained(model_id)
            logging.info("Download complete.")

        self.model = self.model.to(self.device)
        self._use_fp16 = False
        if self.device == "cuda":
            cap_major, _ = torch.cuda.get_device_capability()
            self._use_fp16 = (cap_major >= 6)
            if self._use_fp16: self.model = self.model.half()
            logging.info(f"Compute precision: {'fp16' if self._use_fp16 else 'fp32'} (compute capability {cap_major}.x)")
        self.model.eval()

        # POWER MOVE: PyTorch 2.0 Compilation
        # Checks GPU capability to avoid crashes on older cards (GTX 10xx).
        can_compile = False
        if hasattr(torch, "compile") and self.device == "cuda":
            # FIX: get_device_capability() returns current device by default, or takes an int index, not a string.
            cap_major, _ = torch.cuda.get_device_capability() 
            if cap_major >= 7.0:
                can_compile = True
            else:
                logging.info(f"Skipping torch.compile (GPU Compute Cap {cap_major}.x < 7.0). Triton requires newer hardware.")

        if can_compile:
            try:
                logging.info("Compiling CLIP Vision Model with torch.compile (this may take a minute)...")
                self.model.vision_model = torch.compile(
                    self.model.vision_model, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                logging.info("Model compiled successfully.")
            except Exception as e:
                logging.warning(f"torch.compile failed (fallback to eager mode): {e}")

        self._cudnn_ok = None
        self.embedding_dim = self.model.config.projection_dim
        self._gpu_fully_oom = False

        self.cache_dir = Path.home() / ".cache" / "photofind"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_cache_dir = self.cache_dir / "thumbs"
        self.thumb_cache_dir.mkdir(parents=True, exist_ok=True)

        self._NUM_THUMB_STRIPES = 64
        self._thumb_locks = [threading.Lock() for _ in range(self._NUM_THUMB_STRIPES)]

        self.index_file = self.cache_dir / "photo_index.json"
        self.embeddings_file = self.cache_dir / "photo_embeddings.pt"
        self.metadata_file = self.cache_dir / "photo_metadata.json"
        self.stats_file = self.cache_dir / "stats.json"
        self.excluded_file = self.cache_dir / "excluded_dirs.json"
        self.excluded_dirs: Set[str] = set()
        try:
            if self.excluded_file.exists():
                with open(self.excluded_file, "r") as f: self.excluded_dirs = set(json.load(f))
        except Exception: self.excluded_dirs = set()

        self.settings_file = self.cache_dir / "settings.json"
        self.search_settings: Dict[str, Any] = dict(DEFAULT_SEARCH_SETTINGS)
        self.quality_settings: Dict[str, Any] = dict(DEFAULT_QUALITY_SETTINGS)
        self.viewer_settings: Dict[str, Any] = dict(DEFAULT_VIEWER_SETTINGS)
        self._load_settings()

        self.image_paths, self.image_embeddings, self.image_metadata, self.indexed_set = [], None, {}, set()
        self.last_search_warnings: List[str] = []
        self.stats = self._load_stats()
        self._lock = threading.Lock()
        self._model_lock = threading.RLock()
        self._embeddings_on_gpu, self._embeddings_device, self._embeddings_dirty = None, 'cpu', True
        self.current_batch_size, self.batch_size_lock = 32, threading.Lock()
        self._dir_counts: Optional[Dict[str, int]] = None

    def _load_settings(self) -> None:
        """Merge persisted settings over the built-in defaults."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r") as f: data = json.load(f)
                for k, v in (data.get("search") or {}).items():
                    if k in self.search_settings: self.search_settings[k] = v
                for k, v in (data.get("quality") or {}).items():
                    if k in self.quality_settings: self.quality_settings[k] = v
                for k, v in (data.get("viewer") or {}).items():
                    if k in self.viewer_settings: self.viewer_settings[k] = v
                try: ver = int(data.get("version", 1))
                except (TypeError, ValueError): ver = 1
                if ver < SETTINGS_VERSION and self.viewer_settings.get("image_viewer") == "system":
                    # "system" was only ever the old default; adopt the new one
                    # and persist the migration so it does not repeat.
                    self.viewer_settings["image_viewer"] = DEFAULT_VIEWER_SETTINGS["image_viewer"]
                    self.save_settings()
        except Exception as e:
            logging.warning(f"Could not read settings: {type(e).__name__}: {e}. Using defaults.")

    def save_settings(self, search: Optional[Dict[str, Any]] = None,
                      quality: Optional[Dict[str, Any]] = None,
                      viewer: Optional[Dict[str, Any]] = None) -> None:
        """Persist the given (partial) search/quality/viewer settings updates."""
        if search: self.search_settings.update(search)
        if quality: self.quality_settings.update(quality)
        if viewer: self.viewer_settings.update(viewer)
        try:
            tmp = str(self.settings_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"version": SETTINGS_VERSION, "search": self.search_settings,
                           "quality": self.quality_settings, "viewer": self.viewer_settings},
                          f, indent=2)
            os.replace(tmp, str(self.settings_file))
        except Exception as e:
            logging.warning(f"Could not save settings: {type(e).__name__}: {e}")

    def _load_stats(self) -> Dict[str, Any]:
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f: return json.load(f)
            except Exception: pass
        return {}

    def save_index_stats(self, duration: float, count: int) -> None:
        self.stats["last_index_duration"] = format_duration(duration)
        self.stats["last_index_count"] = count; self._save_stats()

    def save_dedupe_stats(self, duration: float, count: int) -> None:
        self.stats["last_dedupe_duration"] = format_duration(duration)
        self.stats["last_dedupe_count"] = count; self._save_stats()

    def _save_stats(self) -> None:
        try:
            with open(self.stats_file, "w") as f: json.dump(self.stats, f)
        except Exception: pass

    def load_index(self) -> bool:
        if not self.index_file.exists() or not self.embeddings_file.exists():
            return False

        try:
            with open(
                self.index_file,
                "r",
                errors="surrogateescape"
            ) as f:
                raw_paths = json.load(f)

            if not isinstance(raw_paths, list):
                raise ValueError("Index paths are not a list")

            loaded_embeddings = torch.load(
                str(self.embeddings_file),
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(loaded_embeddings, torch.Tensor):
                raise ValueError(
                    "Embedding index does not contain a Tensor"
                )

            if loaded_embeddings.ndim != 2:
                raise ValueError(
                    f"Embedding index must be 2-D, "
                    f"got {loaded_embeddings.ndim}-D"
                )

            if loaded_embeddings.shape[1] != self.embedding_dim:
                logging.warning(
                    "Incompatible index dimension: "
                    f"{loaded_embeddings.shape[1]} "
                    f"(expected {self.embedding_dim}). "
                    "Please re-index."
                )
                return False

            if loaded_embeddings.shape[0] != len(raw_paths):
                raise ValueError(
                    "Corrupt index: "
                    f"{len(raw_paths)} paths but "
                    f"{loaded_embeddings.shape[0]} embeddings"
                )

            if self.device == "cuda":
                loaded_embeddings = loaded_embeddings.half()
            else:
                loaded_embeddings = loaded_embeddings.float()

            self.image_paths = raw_paths
            self.image_embeddings = loaded_embeddings
            self.indexed_set = set(raw_paths)

            if self.metadata_file.exists():
                with open(
                    self.metadata_file,
                    "r",
                    errors="surrogateescape"
                ) as f:
                    self.image_metadata = json.load(f)

                if not isinstance(self.image_metadata, dict):
                    raise ValueError(
                        "Metadata index is not a dictionary"
                    )
            else:
                self.image_metadata = {}

            for p in self.image_paths:
                if p not in self.image_metadata:
                    self.image_metadata[p] = {
                        "std_dev": 50.0,
                        "edge_score": 50.0
                    }

            self._embeddings_dirty = True

            logging.info(
                f"Loaded {len(self.image_paths)} image embeddings."
            )
            return True

        except Exception as e:
            logging.error(
                f"Error loading index: {type(e).__name__}: {e}"
            )

            self.image_paths = []
            self.image_embeddings = None
            self.image_metadata = {}
            self.indexed_set = set()

        return False

    def _get_thumb_lock(self, path: str) -> threading.Lock:
        return self._thumb_locks[hash(path) % self._NUM_THUMB_STRIPES]

    def _get_cached_thumbnail(self, path: str, size: Tuple[int, int] = (200, 200)) -> Optional[Image.Image]:
        thumb_hash = hashlib.md5(path.encode('utf-8', errors='surrogateescape')).hexdigest()
        thumb_path = self.thumb_cache_dir / f"{thumb_hash}.jpg"
        if thumb_path.exists():
            try: return Image.open(thumb_path).copy()
            except Exception: pass
        lock = self._get_thumb_lock(path)
        with lock:
            if thumb_path.exists():
                try: return Image.open(thumb_path).copy()
                except Exception: pass
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != "RGB": img = img.convert("RGB")
                    thumb = img.copy(); thumb.thumbnail(size, Image.Resampling.LANCZOS)
                    tmp = thumb_path.with_suffix('.tmp')
                    thumb.save(tmp, "JPEG", quality=85); os.replace(str(tmp), str(thumb_path))
                    return thumb
            except Exception: return None

    def _load_single_image(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict[str, float]]]:
        try:
            img = Image.open(file_path); img = ImageOps.exif_transpose(img); w, h = img.size
            if min(w, h) < 32 or max(w, h) < 64: return None, None, None
            if w * h > 89_478_485: logging.warning(f"Large image ({w}x{h}, {w*h/1e6:.0f}MP), indexing anyway: {file_path}")
            if img.mode != "RGB": img = img.convert("RGB")
            else: img.load()
            img = ImageOps.fit(img, (224, 224), Image.Resampling.BICUBIC)
            arr, std_dev, edge_score, mean_lum, lap_var, lum_p95, lap_p95 = _numpy_to_clip_arrays(img)

            img.close()

            return arr, os.path.realpath(file_path), {
                "std_dev": std_dev,
                "edge_score": edge_score, "mean_lum": mean_lum, "lap_var": lap_var, "lum_p95": lum_p95, "lap_p95": lap_p95,
            }

        except Exception as e: logging.warning(f"Failed to load image {file_path}: {type(e).__name__}: {e}"); return None, None, None

    def _release_gpu_embeddings(self) -> None:
        with self._lock:
            if self._embeddings_on_gpu is not None:
                del self._embeddings_on_gpu; self._embeddings_on_gpu = None; self._embeddings_dirty = True
        if self.device == "cuda": torch.cuda.empty_cache()

    def _compute_vision_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        is_cuda = (pixel_values.device.type == 'cuda')
        with self._model_lock, torch.no_grad():
            if is_cuda:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = self.model.vision_model(pixel_values=pixel_values)
                    pooled = out.pooler_output if hasattr(out, 'pooler_output') else out[1]
                    features = self.model.visual_projection(pooled)
            else:
                out = self.model.vision_model(pixel_values=pixel_values)
                pooled = out.pooler_output if hasattr(out, 'pooler_output') else out[1]
                features = self.model.visual_projection(pooled)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().half()

    def _calibrate_batch_size(self) -> int:
        """Synthetic calibration with adaptive probing. Skips unsupported cuDNN on older GPUs."""
        if self.device != "cuda": return 8
        gc.collect(); torch.cuda.empty_cache()
        free_vram, _ = torch.cuda.mem_get_info()
        
        cap_major, _ = torch.cuda.get_device_capability()
        
        logging.info(f"Calibrating batch size (free VRAM: {free_vram / (1024**2):.0f}MB, Compute Cap: {cap_major}.x)...")

        probe_sequence = [True, False]

        for try_cudnn in probe_sequence:
            torch.backends.cudnn.enabled = try_cudnn
            torch.backends.cudnn.benchmark = False
            dummy_pv = None
            try:
                dummy_pv = torch.randn(2, 3, 224, 224, device=self.device, dtype=torch.float16)
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with self._model_lock:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            out = self.model.vision_model(pixel_values=dummy_pv)
                            pooled = out.pooler_output if hasattr(out, 'pooler_output') else out[1]
                            features = self.model.visual_projection(pooled)
                torch.cuda.synchronize()
                peak_delta = torch.cuda.max_memory_allocated() - mem_before
                bytes_per_img_2 = max(peak_delta / 2, 1)
                del dummy_pv, out, pooled, features; torch.cuda.empty_cache()

                probe_bs = min(16, int(free_vram * 0.15 / bytes_per_img_2), MAX_BATCH_SIZE)
                if probe_bs >= 4:
                    probe_pv = torch.randn(probe_bs, 3, 224, 224, device=self.device, dtype=torch.float16)
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    with self._model_lock:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                out = self.model.vision_model(pixel_values=probe_pv)
                                pooled = out.pooler_output if hasattr(out, 'pooler_output') else out[1]
                                features = self.model.visual_projection(pooled)
                    torch.cuda.synchronize()
                    peak_delta_big = torch.cuda.max_memory_allocated() - mem_before
                    bytes_per_img_large = peak_delta_big / probe_bs
                    del probe_pv, out, pooled, features; torch.cuda.empty_cache()
                    bytes_per_img = bytes_per_img_large
                else:
                    bytes_per_img = bytes_per_img_2

                reserve_bytes = VRAM_RESERVE_MB * (1024**2)
                safe_available = max(free_vram - reserve_bytes, 0)

                target_bs = int(safe_available / bytes_per_img)
                target_bs = max(2, min(target_bs, MAX_BATCH_SIZE))

                logging.info(f"Calibration (cuDNN={'ON' if try_cudnn else 'OFF'}): "
                             f"{bytes_per_img / (1024**2):.1f}MB/img. "
                             f"Reserving {VRAM_RESERVE_MB}MB. "
                             f"Batch size: {target_bs}")
                self._cudnn_ok = try_cudnn
                return target_bs
            except RuntimeError as e:
                err_str = str(e).lower()
                if "unable to find an engine" in err_str or "out of memory" in err_str:
                    logging.warning(f"cuDNN={'ON' if try_cudnn else 'OFF'} failed: {e}")
                    if dummy_pv is not None:
                        del dummy_pv
                        dummy_pv = None
                    torch.cuda.empty_cache()
                    if not try_cudnn:
                        logging.warning("GPU calibration failed entirely. Using CPU fallback.")
                        self._cudnn_ok = False; self._gpu_fully_oom = True; return 16
                    logging.info("Disabling cuDNN (not supported on this GPU)...")
                    continue
                raise
        self._cudnn_ok = False; self._gpu_fully_oom = True; return 16

    def _process_batch_cpu(self, arr: np.ndarray, batch_map: List[str], batch_meta: List[Dict],
                           cancel_check: Optional[Callable[[], bool]],
                           new_embeddings: List[torch.Tensor], new_paths: List[str],
                           new_metadata: List[Dict]) -> bool:
        chunk_sz = 4
        for i in range(0, len(batch_map), chunk_sz):
            if cancel_check and cancel_check(): return True
            chunk_arr = arr[i:i+chunk_sz]
            pv = torch.from_numpy(chunk_arr).float()
            features = self._compute_vision_embeddings(pv)
            new_embeddings.append(features); new_paths.extend(batch_map[i:i+chunk_sz])
            new_metadata.extend(batch_meta[i:i+chunk_sz]); del pv, features
        return False

    def _ensure_model_on_cpu(self) -> None:
        with self._model_lock:
            if self._use_fp16: self.model.vision_model.float().to('cpu'); self.model.visual_projection.float().to('cpu')
            else: self.model.vision_model.to('cpu'); self.model.visual_projection.to('cpu')
        if self.device == "cuda": torch.cuda.empty_cache()

    def _ensure_model_on_gpu(self) -> None:
        with self._model_lock:
            if self._use_fp16: self.model.vision_model.half().to(self.device); self.model.visual_projection.half().to(self.device)
            else: self.model.vision_model.to(self.device); self.model.visual_projection.to(self.device)
        if self.device == "cuda": torch.cuda.empty_cache()

    def index_photos(self, source_dir: str, progress_callback: Optional[Callable[[str, int, int], None]] = None,
                     cancel_check: Optional[Callable[[], bool]] = None) -> None:
        self._release_gpu_embeddings()
        self._ips_last_t, self._ips_ema = None, 0.0
        source_path = os.path.realpath(source_dir); extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        logging.info(f"Scanning {source_path} for images..."); files = []
        for root, dirs, filenames in os.walk(source_path):
            if cancel_check and cancel_check():
                logging.info("Indexing cancelled."); gc.collect();
                if self.device == "cuda": torch.cuda.empty_cache()
                return
            dirs[:] = [d for d in dirs if not d.startswith('.') and not self._dir_excluded(os.path.join(root, d))
                       and not self._fragment_skipped(os.path.join(root, d))]
            if self._dir_excluded(root) or self._fragment_skipped(root): continue
            for filename in filenames:
                fpath = os.path.join(root, filename)
                if Path(filename).suffix.lower() in extensions and not self._fragment_skipped(fpath):
                    files.append(fpath)
        with self._lock:
            if self.indexed_set:
                initial_count = len(files); files = [f for f in files if f not in self.indexed_set]
                if initial_count - len(files) > 0: logging.info(f"Skipped {initial_count - len(files)} already indexed images.")
        if not files: logging.info("No new images to index."); return
        total_files = len(files); logging.info(f"Processing {total_files} new images...")

        if self.device == "cuda":
            self.current_batch_size = self._calibrate_batch_size()
            if cancel_check and cancel_check(): return
            if self._gpu_fully_oom:
                self._ensure_model_on_cpu()
                logging.warning("GPU too small for CLIP. Using CPU for all batches.")
        else: self.current_batch_size = 8

        max_loaders = min(16, max(6, (os.cpu_count() or 4) * 2))
        load_queue = queue.Queue(maxsize=600)
        gpu_queue = queue.Queue(maxsize=4)
        mode_str = 'CPU' if self._gpu_fully_oom else f'GPU (cuDNN={"ON" if self._cudnn_ok else "OFF"})'
        logging.info(f"Pipeline: {max_loaders} loaders -> preprocess -> {mode_str}")

        def loader_worker(file_chunk: List[str]) -> None:
            for f in file_chunk:
                if cancel_check and cancel_check(): break
                result = self._load_single_image(f)
                if result[0] is not None:
                    while not (cancel_check and cancel_check()):
                        try: load_queue.put(result, timeout=0.5); break
                        except queue.Full: continue
            load_queue.put(None)

        def preprocess_worker(num_loaders: int) -> None:
            active_loaders = num_loaders; batch_arrays, batch_map, batch_meta = [], [], []
            while True:
                with self.batch_size_lock: current_bs = self.current_batch_size
                while len(batch_arrays) < current_bs:
                    with self.batch_size_lock: current_bs = self.current_batch_size
                    if cancel_check and cancel_check():
                        if batch_arrays: gpu_queue.put((np.stack(batch_arrays), batch_map, batch_meta))
                        gpu_queue.put("DONE"); return
                    try: item = load_queue.get(timeout=0.1)
                    except queue.Empty: continue
                    if item is None:
                        active_loaders -= 1
                        if active_loaders == 0:
                            if batch_arrays: gpu_queue.put((np.stack(batch_arrays), batch_map, batch_meta))
                            gpu_queue.put("DONE"); return
                        continue
                    arr, path, meta = item
                    batch_arrays.append(arr); batch_map.append(path); batch_meta.append(meta)
                if batch_arrays:
                    try:
                        gpu_queue.put((np.stack(batch_arrays), batch_map, batch_meta))
                        batch_arrays, batch_map, batch_meta = [], [], []
                    except Exception as e:
                        logging.error(f"Preprocess worker crashed: {e}"); gpu_queue.put("DONE"); return

        chunk_size = max(1, (total_files + max_loaders - 1) // max_loaders)
        chunks = [files[i:i+chunk_size] for i in range(0, total_files, chunk_size)]
        loader_threads = [threading.Thread(target=loader_worker, args=(chunk,), daemon=True) for chunk in chunks]
        for t in loader_threads: t.start()
        preprocess_thread = threading.Thread(target=preprocess_worker, args=(len(loader_threads),), daemon=True)
        preprocess_thread.start()

        new_embeddings, new_paths, new_metadata = [], [], []
        processed_count = 0; cancelled = False

        while True:
            if cancel_check and cancel_check(): cancelled = True; break
            try: item = gpu_queue.get(timeout=1.0)
            except queue.Empty: continue
            if item == "DONE": break
            arr, batch_map, batch_meta = item

            batch_size_actual = len(batch_map)
            if self._gpu_fully_oom:
                if self._process_batch_cpu(arr, batch_map, batch_meta, cancel_check,
                                           new_embeddings, new_paths, new_metadata):
                    cancelled = True; break
                processed_count += batch_size_actual
                if progress_callback:
                    progress_callback(f"Indexing (CPU)... {processed_count}/{total_files}", processed_count, total_files)
                continue

            pv = torch.from_numpy(arr).pin_memory().to(self.device, non_blocking=True)
            
            try:
                features = self._compute_vision_embeddings(pv)
                new_embeddings.append(features); new_paths.extend(batch_map); new_metadata.extend(batch_meta)
                del pv, features; processed_count += batch_size_actual
                if progress_callback:
                    now = time.time()
                    rate_str = ""
                    if self._ips_last_t is not None:
                        dt = now - self._ips_last_t
                        if dt > 0.05:
                            inst = batch_size_actual / dt
                            self._ips_ema = inst if self._ips_ema == 0.0 else 0.8 * self._ips_ema + 0.2 * inst
                            rate_str = f", {self._ips_ema:.1f} img/s"
                    self._ips_last_t = now
                    progress_callback(f"Indexing... {processed_count}/{total_files} (BS: {self.current_batch_size}{rate_str})", processed_count, total_files)
            except RuntimeError as e:
                err_str = str(e).lower()
                if "out of memory" in err_str or "unable to find an engine" in err_str:
                    logging.warning(f"CUDA OOM on batch of {batch_size_actual}. Reducing...")
                    try: del pv
                    except Exception: pass
                    torch.cuda.empty_cache()
                    with self.batch_size_lock:
                        self.current_batch_size = max(1, self.current_batch_size // 2); new_bs = self.current_batch_size
                    if new_bs <= 1:
                        logging.warning("Switching to CPU fallback for all remaining batches.")
                        self._gpu_fully_oom = True; self._ensure_model_on_cpu()
                        if self._process_batch_cpu(arr, batch_map, batch_meta, cancel_check,
                                                   new_embeddings, new_paths, new_metadata):
                            cancelled = True; break
                    else:
                        chunk_ok = True
                        for i in range(0, batch_size_actual, new_bs):
                            if cancel_check and cancel_check(): chunk_ok = False; break
                            chunk_arr = arr[i:i+new_bs]; chunk_pv = None
                            try:
                                chunk_pv = torch.from_numpy(chunk_arr).pin_memory().to(self.device, non_blocking=True)
                                features = self._compute_vision_embeddings(chunk_pv)
                                new_embeddings.append(features)
                                new_paths.extend(batch_map[i:i+new_bs]); new_metadata.extend(batch_meta[i:i+new_bs])
                                del chunk_pv, features
                            except RuntimeError as e2:
                                if "out of memory" in str(e2).lower() or "unable to find an engine" in str(e2).lower():
                                    try: del chunk_pv
                                    except Exception: pass
                                    torch.cuda.empty_cache()
                                    logging.warning("Still OOM after reduction. Switching to CPU fallback.")
                                    self._gpu_fully_oom = True; self._ensure_model_on_cpu()
                                    if self._process_batch_cpu(arr[i:], batch_map[i:], batch_meta[i:],
                                                               cancel_check, new_embeddings, new_paths, new_metadata):
                                        chunk_ok = False
                                    break
                                else:
                                    try: del chunk_pv
                                    except Exception: pass
                                    raise
                        if not chunk_ok: cancelled = True; break
                    processed_count += batch_size_actual
                    if progress_callback:
                        mode = "CPU" if self._gpu_fully_oom else "GPU"
                        progress_callback(f"Indexing ({mode})... {processed_count}/{total_files}", processed_count, total_files)
                else: del pv; raise

        if self._gpu_fully_oom and self.device == "cuda":
            self._ensure_model_on_gpu(); self._gpu_fully_oom = False
        while not gpu_queue.empty():
            try: gpu_queue.get_nowait()
            except queue.Empty: break
        for t in loader_threads: t.join(timeout=2.0)
        preprocess_thread.join(timeout=2.0); gc.collect()
        if cancelled:
            logging.info("Indexing cancelled by user."); gc.collect()
            if self.device == "cuda": torch.cuda.empty_cache()
            return
        if new_embeddings:
            if progress_callback: progress_callback("Saving index...", total_files, total_files)
            logging.info(f"Saving index ({len(self.image_paths) + len(new_paths)} total embeddings)...")
            new_embeddings_tensor = torch.cat(new_embeddings, dim=0); del new_embeddings; gc.collect()
            with self._lock:
                if self.image_embeddings is not None:
                    all_embeddings = torch.cat([self.image_embeddings, new_embeddings_tensor], dim=0)
                    all_paths = self.image_paths + new_paths; del self.image_embeddings
                else: all_embeddings, all_paths = new_embeddings_tensor, new_paths
                for i, p in enumerate(new_paths): self.image_metadata[p] = new_metadata[i]
                temp_emb = str(self.embeddings_file)+".tmp"
                temp_idx = str(self.index_file)+".tmp"
                temp_meta = str(self.metadata_file)+".tmp"
                torch.save(all_embeddings, temp_emb)
                with open(temp_idx, "w", errors='surrogateescape') as f: json.dump(all_paths, f)
                with open(temp_meta, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
                os.replace(temp_emb, str(self.embeddings_file))
                os.replace(temp_idx, str(self.index_file))
                os.replace(temp_meta, str(self.metadata_file))
                self.image_embeddings, self.image_paths = all_embeddings, all_paths
                self.indexed_set = set(self.image_paths); self._embeddings_dirty = True
                self._dir_counts = None; del all_embeddings
            gc.collect()
            if self.device == "cuda": torch.cuda.empty_cache()
            logging.info("Indexing complete.")
        else: logging.info("No valid images processed.")

    def clean_database(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> int:
        if not self.image_paths:
            return 0

        valid_paths = []
        valid_indices = []
        removed_count = 0
        new_metadata = dict(self.image_metadata)

        dir_map: Dict[str, List[int]] = {}

        for i, path in enumerate(self.image_paths):
            directory = os.path.dirname(path)
            dir_map.setdefault(directory, []).append(i)

        total_dirs = len(dir_map)

        for dir_idx, (directory, indices) in enumerate(
            dir_map.items()
        ):
            if progress_callback and dir_idx % 500 == 0:
                progress_callback(
                    f"Verifying directories... "
                    f"{dir_idx}/{total_dirs}"
                )

            # A directory which genuinely disappeared can safely have
            # its indexed entries removed.
            if not os.path.exists(directory):
                access_err = self._check_file_access(directory)
                if access_err and "not available" in access_err:
                    logging.warning(
                        f"Directory {directory!r} is on an unavailable "
                        f"filesystem; keeping {len(indices)} entries."
                    )
                    valid_paths.extend(self.image_paths[i] for i in indices)
                    valid_indices.extend(indices)
                    continue
                removed_count += len(indices)

                for i in indices:
                    p = self.image_paths[i]
                    new_metadata.pop(p, None)

                continue

            try:
                actual_files = set(os.listdir(directory))

            except (PermissionError, OSError) as e:
                # IMPORTANT:
                # Inaccessible does NOT mean empty.
                #
                # Keep all entries because we cannot determine whether
                # the files are actually missing.
                logging.warning(
                    f"Cannot inspect directory {directory!r}: "
                    f"{type(e).__name__}: {e}. "
                    f"Keeping {len(indices)} indexed entries."
                )

                valid_paths.extend(
                    self.image_paths[i]
                    for i in indices
                )
                valid_indices.extend(indices)
                continue

            for i in indices:
                path = self.image_paths[i]
                filename = os.path.basename(path)

                if filename in actual_files:
                    valid_paths.append(path)
                    valid_indices.append(i)

                else:
                    removed_count += 1
                    new_metadata.pop(path, None)

        if removed_count <= 0:
            return 0

        logging.info(
            f"Cleaning database: Removing "
            f"{removed_count} missing files."
        )

        with self._lock:
            backup = (self.image_paths, self.indexed_set, self.image_embeddings, dict(self.image_metadata), self._embeddings_dirty)
            self.image_paths = valid_paths
            self.indexed_set = set(valid_paths)
            self.image_metadata = new_metadata

            if self.image_embeddings is not None:
                if valid_indices:
                    self.image_embeddings = (
                        self.image_embeddings[valid_indices]
                        .contiguous()
                    )
                else:
                    self.image_embeddings = None

            self._embeddings_dirty = True
            self._dir_counts = None

            # Write temporary files first so a failed write doesn't leave
            # a partially-written JSON file.
            temp_emb = str(self.embeddings_file) + ".tmp"
            temp_idx = str(self.index_file) + ".tmp"
            temp_meta = str(self.metadata_file) + ".tmp"

            try:
                if self.image_embeddings is not None:
                    torch.save(
                        self.image_embeddings,
                        temp_emb
                    )

                with open(
                    temp_idx,
                    "w",
                    errors="surrogateescape"
                ) as f:
                    json.dump(self.image_paths, f)

                with open(
                    temp_meta,
                    "w",
                    errors="surrogateescape"
                ) as f:
                    json.dump(self.image_metadata, f)

                if self.image_embeddings is not None:
                    os.replace(
                        temp_emb,
                        str(self.embeddings_file)
                    )
                elif self.embeddings_file.exists():
                    os.remove(self.embeddings_file)

                os.replace(
                    temp_idx,
                    str(self.index_file)
                )

                os.replace(
                    temp_meta,
                    str(self.metadata_file)
                )

            except Exception:
                for tmp in (
                    temp_emb,
                    temp_idx,
                    temp_meta,
                ):
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass

                self.image_paths, self.indexed_set, self.image_embeddings, self.image_metadata, self._embeddings_dirty = backup
                raise

        return removed_count

    def mark_photo_deleted(self, file_path: str) -> None:
        self.mark_photos_deleted([file_path])

    def mark_photos_deleted(self, file_paths: List[str]) -> None:
        with self._lock:
            changed = False
            for fp in file_paths:
                if fp in self.image_metadata:
                    self.image_metadata[fp]['deleted'] = True; changed = True
            if changed:
                with open(self.metadata_file, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)

    def _dir_excluded(self, dir_path: str) -> bool:
        if not self.excluded_dirs: return False
        rp = os.path.realpath(dir_path)
        return any(rp == x or rp.startswith(x + os.sep) for x in self.excluded_dirs)

    def _fragment_skipped(self, path: str) -> bool:
        if not INDEX_EXCLUDE_FRAGMENTS: return False
        return any(fragment in os.path.realpath(path).lower() for fragment in INDEX_EXCLUDE_FRAGMENTS)

    def exclude_directory(self, directory: str) -> int:
        """Exclude a directory and everything under it: removes its indexed
        entries now and skips it in all future indexing. Returns removed count."""
        rp = os.path.realpath(directory); prefix = rp + os.sep
        self.excluded_dirs.add(rp)
        try:
            with open(self.excluded_file, "w") as f: json.dump(sorted(self.excluded_dirs), f)
        except Exception: pass
        with self._lock:
            backup = (self.image_paths, self.indexed_set, self.image_embeddings, dict(self.image_metadata), self._embeddings_dirty)
            keep_indices = [i for i, p in enumerate(self.image_paths) if not p.startswith(prefix)]
            removed = len(self.image_paths) - len(keep_indices)
            if removed == 0: return 0
            self.image_paths = [self.image_paths[i] for i in keep_indices]
            self.indexed_set = set(self.image_paths)
            for p in [p for p in self.image_metadata if p.startswith(prefix)]: self.image_metadata.pop(p, None)
            if self.image_embeddings is not None:
                self.image_embeddings = self.image_embeddings[keep_indices].contiguous() if keep_indices else None
            self._embeddings_dirty = True; self._dir_counts = None
            temp_emb, temp_idx, temp_meta = str(self.embeddings_file)+".tmp", str(self.index_file)+".tmp", str(self.metadata_file)+".tmp"
            try:
                if self.image_embeddings is not None: torch.save(self.image_embeddings, temp_emb)
                with open(temp_idx, "w", errors='surrogateescape') as f: json.dump(self.image_paths, f)
                with open(temp_meta, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
                if self.image_embeddings is not None: os.replace(temp_emb, str(self.embeddings_file))
                elif self.embeddings_file.exists(): os.remove(self.embeddings_file)
                os.replace(temp_idx, str(self.index_file)); os.replace(temp_meta, str(self.metadata_file))
            except Exception:
                for tmp in (temp_emb, temp_idx, temp_meta):
                    try:
                        if os.path.exists(tmp): os.remove(tmp)
                    except OSError: pass
                self.image_paths, self.indexed_set, self.image_embeddings, self.image_metadata, self._embeddings_dirty = backup
                raise
        return removed

    def purge_fragment_entries(self) -> int:
        """Remove already-indexed entries matching INDEX_EXCLUDE_FRAGMENTS."""
        if not INDEX_EXCLUDE_FRAGMENTS: return 0
        with self._lock:
            keep_indices = [i for i, p in enumerate(self.image_paths) if not self._fragment_skipped(p)]
            removed = len(self.image_paths) - len(keep_indices)
            if removed == 0: return 0
            backup = (self.image_paths, self.indexed_set, self.image_embeddings, dict(self.image_metadata), self._embeddings_dirty)
            self.image_paths = [self.image_paths[i] for i in keep_indices]
            self.indexed_set = set(self.image_paths)
            for p in [p for p in self.image_metadata if self._fragment_skipped(p)]: self.image_metadata.pop(p, None)
            if self.image_embeddings is not None:
                self.image_embeddings = self.image_embeddings[keep_indices].contiguous() if keep_indices else None
            self._embeddings_dirty = True; self._dir_counts = None
            temp_emb, temp_idx, temp_meta = str(self.embeddings_file)+".tmp", str(self.index_file)+".tmp", str(self.metadata_file)+".tmp"
            try:
                if self.image_embeddings is not None: torch.save(self.image_embeddings, temp_emb)
                with open(temp_idx, "w", errors='surrogateescape') as f: json.dump(self.image_paths, f)
                with open(temp_meta, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
                if self.image_embeddings is not None: os.replace(temp_emb, str(self.embeddings_file))
                elif self.embeddings_file.exists(): os.remove(self.embeddings_file)
                os.replace(temp_idx, str(self.index_file)); os.replace(temp_meta, str(self.metadata_file))
            except Exception:
                for tmp in (temp_emb, temp_idx, temp_meta):
                    try:
                        if os.path.exists(tmp): os.remove(tmp)
                    except OSError: pass
                self.image_paths, self.indexed_set, self.image_embeddings, self.image_metadata, self._embeddings_dirty = backup
                raise
        return removed

    def _get_embeddings_gpu(self) -> torch.Tensor:
        with self._lock:
            if self._embeddings_dirty or self._embeddings_device != self.device or self._embeddings_on_gpu is None:
                if self.image_embeddings is None: raise ValueError("No embeddings loaded")
                self._embeddings_on_gpu = self.image_embeddings.to(self.device)
                if self.device == "cuda" and self._embeddings_on_gpu.dtype != torch.float16:
                    self._embeddings_on_gpu = self._embeddings_on_gpu.half()
                self._embeddings_device, self._embeddings_dirty = self.device, False
            return self._embeddings_on_gpu

    def _check_file_access(self, path: str) -> Optional[str]:
        """Return None if `path` is readable, else a human-readable reason."""
        if os.path.exists(path): return None
        p = Path(path)
        removable_roots = ('/run/media', '/media', '/mnt')
        root = next((Path(m) for m in removable_roots if p.is_relative_to(m)), None)
        if root is not None:
            # Walk down the removable root looking for an actual mountpoint. A
            # mounted drive means the file was deleted/moved; an unmounted one
            # (no mountpoint found) means the drive is disconnected.
            mounted = os.path.ismount(root)
            candidate = root
            try: rel_parts = p.relative_to(root).parts
            except ValueError: rel_parts = ()
            for part in rel_parts:
                candidate = candidate / part
                try:
                    if candidate.exists() and os.path.ismount(candidate): mounted = True
                except OSError: pass
            if not mounted:
                return f"Filesystem under {root} not mounted (drive may be disconnected)"
        return "File not found"

    def search(self, query: str, top_k: Optional[int] = None,
               settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        s = self.search_settings if settings is None else settings
        if top_k is None: top_k = int(s.get("top_k", 60))
        if self.image_embeddings is None or len(self.image_embeddings) == 0: return []
        if self.image_embeddings.shape[1] != self.embedding_dim:
            logging.error(f"Embedding dimension mismatch. Please re-index."); return []
        search_text = f"a photo of a {query}" if len(query.split()) < 4 else query
        
        with self._model_lock:
            inputs = self.processor(text=[search_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self._model_lock, torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if self.device == "cuda" else 'cpu', dtype=torch.float16):
                text_outputs = self.model.text_model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
                pooled = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs[1]
                text_features = self.model.text_projection(pooled)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            if self.device == "cuda":
                text_features = text_features.half()

        with torch.no_grad():
            embeddings = self._get_embeddings_gpu()
            similarities = torch.mm(embeddings, text_features.T).squeeze().float()
            requested_k = min(max(top_k * int(s.get("candidate_mult", 40)), 1000), len(similarities))
            values, indices = torch.topk(similarities, requested_k)

        cand = embeddings[indices].float().cpu()
        scores_np = values.cpu().numpy()
        idxs_np = indices.cpu().numpy()
        
        if len(scores_np) == 0: return []
        
        effective_floor = max(s.get("floor_min", 0.15), float(scores_np[0]) * s.get("floor_ratio", 0.45))
        dedupe_cos = s.get("dedupe_cos", 0.99)
        results = []
        accepted_refs: List[torch.Tensor] = []
        skipped_dup = 0
        unmounted: Set[str] = set()
        self.last_search_warnings = []
        
        for i in range(len(scores_np)):
            score = float(scores_np[i])
            if score < effective_floor: break
            
            idx = int(idxs_np[i])
            fp = self.image_paths[idx]
            
            meta = self.image_metadata.get(fp)
            if meta and meta.get('deleted'): continue
            if meta and self._is_black(meta, s): continue
            err = self._check_file_access(fp)
            if err is not None:
                if "not available" in err: unmounted.add(err)
                continue
            
            vec = cand[i]
            if accepted_refs:
                if bool((vec @ torch.stack(accepted_refs).t() > dedupe_cos).any()):
                    skipped_dup += 1
                    continue
            accepted_refs.append(vec)
            results.append({"file": fp, "score": score})
            if len(results) >= top_k: break
            
        if skipped_dup:
            logging.info(f"Search dedup: skipped {skipped_dup} near-duplicate photos.")
        if unmounted:
            self.last_search_warnings = sorted(unmounted)
            for w in self.last_search_warnings:
                logging.warning(f"Search skipped results: {w}")
        return results

    def _darkness(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> float:
        """[0,1] darkness: 1 when the 95th-percentile luminance is far below the cutoff."""
        q = self.quality_settings if q is None else q
        lum_p95 = meta.get("lum_p95")
        if lum_p95 is None: return 0.0
        cut = float(q.get("dark_lum_p95", 25.0)); span = max(1e-6, float(q.get("dark_span", 25.0)))
        return max(0.0, min(1.0, (cut - lum_p95) / span))

    def _is_blank(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> bool:
        """True for (near) solid-colour / featureless frames (std_dev ~ 0)."""
        q = self.quality_settings if q is None else q
        std = meta.get("std_dev")
        return std is not None and std <= float(q.get("blank_std_max", 2.5))

    def _is_featureless(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> bool:
        """True when there is too little edge content to judge blur at all."""
        q = self.quality_settings if q is None else q
        edge = meta.get("edge_score")
        return edge is not None and edge <= float(q.get("blur_min_edge", 2.0))

    def _blur(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> float:
        """[0,1] blur from the variance of the Laplacian; featureless frames are
        excluded (they are blank, not blurry)."""
        q = self.quality_settings if q is None else q
        if self._is_blank(meta, q) or self._is_featureless(meta, q): return 0.0
        lap_var = meta.get("lap_var")
        if lap_var is None: return 0.0
        cut = float(q.get("blur_lap_var", 20.0)); span = max(1e-6, float(q.get("blur_span", 30.0)))
        return max(0.0, min(1.0, (cut - lap_var) / span))

    def _blankness(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> float:
        """[0,1] blankness: 1 for a solid-colour frame, 0 once variation is normal."""
        q = self.quality_settings if q is None else q
        std = meta.get("std_dev")
        if std is None: return 0.0
        cut = float(q.get("blank_std_max", 2.5)); span = max(1e-6, float(q.get("blank_span", 5.0)))
        return max(0.0, min(1.0, (cut - std) / span))

    def _overexposure(self, meta: Dict[str, Any], q: Optional[Dict[str, Any]] = None) -> float:
        """[0,1] overexposure from blown highlights and/or a high mean luminance."""
        q = self.quality_settings if q is None else q
        score = 0.0
        lum_p95, mean_lum = meta.get("lum_p95"), meta.get("mean_lum")
        if lum_p95 is not None:
            cut = float(q.get("over_lum_p95", 235.0)); span = max(1e-6, float(q.get("over_lum_span", 20.0)))
            score = max(score, max(0.0, min(1.0, (lum_p95 - cut) / span)))
        if mean_lum is not None:
            cut = float(q.get("over_mean_lum", 200.0)); span = max(1e-6, float(q.get("over_mean_span", 55.0)))
            score = max(score, max(0.0, min(1.0, (mean_lum - cut) / span)))
        return score

    def _badness(self, meta: Dict[str, Any]) -> float:
        """Badness in [0, 2]: 0 = healthy; near-black and/or blurry raises it."""
        return self._darkness(meta) + self._blur(meta)

    def _problem_severity(self, kind: str, meta: Dict[str, Any],
                          q: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Unclipped severity for a problem kind: >0 means past the cutoff,
        larger = worse, <=0 = healthier. Spans normalise the kinds so they are
        loosely comparable. Returns None when the needed metadata is missing."""
        q = self.quality_settings if q is None else q
        sev: Optional[float] = None

        def bump(v: float) -> None:
            nonlocal sev
            sev = v if sev is None else max(sev, v)

        if kind in ("dark", "all"):
            lum_p95 = meta.get("lum_p95")
            if lum_p95 is not None:
                bump((float(q.get("dark_lum_p95", 25.0)) - float(lum_p95))
                     / max(1e-6, float(q.get("dark_span", 25.0))))
        if kind in ("blurry", "all"):
            lap_var = meta.get("lap_var")
            # Featureless/blank frames have no edges to blur, so they are never
            # counted as "blurry" (they belong to the blank category instead).
            if lap_var is not None and not (self._is_blank(meta, q) or self._is_featureless(meta, q)):
                bump((float(q.get("blur_lap_var", 20.0)) - float(lap_var))
                     / max(1e-6, float(q.get("blur_span", 30.0))))
        if kind in ("blank", "all"):
            std = meta.get("std_dev")
            if std is not None:
                bump((float(q.get("blank_std_max", 2.5)) - float(std))
                     / max(1e-6, float(q.get("blank_span", 5.0))))
        if kind in ("overexposed", "all"):
            lum_p95 = meta.get("lum_p95")
            if lum_p95 is not None:
                bump((float(lum_p95) - float(q.get("over_lum_p95", 235.0)))
                     / max(1e-6, float(q.get("over_lum_span", 20.0))))
            mean_lum = meta.get("mean_lum")
            if mean_lum is not None:
                bump((float(mean_lum) - float(q.get("over_mean_lum", 200.0)))
                     / max(1e-6, float(q.get("over_mean_span", 55.0))))
        return sev

    def _problem_score(self, kind: str, meta: Dict[str, Any],
                       q: Optional[Dict[str, Any]] = None) -> float:
        sev = self._problem_severity(kind, meta, q)
        return 0.0 if sev is None else max(0.0, min(1.0, sev))

    def _is_garbage(self, meta: Dict[str, Any]) -> bool:
        return self._badness(meta) >= 1.5

    def _is_black(self, meta: Dict[str, Any], s: Optional[Dict[str, Any]] = None) -> bool:
        """Near-black frame hidden from search results (configurable)."""
        s = self.search_settings if s is None else s
        lum_p95 = meta.get("lum_p95")
        return lum_p95 is not None and lum_p95 <= float(s.get("black_lum_p95", 10.0))

    def get_problem_photos(self, kind: str = "all", offset: int = 0, limit: int = 50,
                           q: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Worst photos for a problem `kind` (dark/blurry/overexposed/all), worst first.

        Always returns the worst-ranked photos (even when none cross the absolute
        cutoff) so the filters stay useful on libraries with no obvious offenders.
        Entries past the cutoff are flagged with `below_threshold`. ``q`` supplies
        candidate quality thresholds (for live previews) instead of the saved ones.
        """
        scored = []
        for p, m in self.image_metadata.items():
            if m.get('deleted'): continue
            sev = self._problem_severity(kind, m, q)
            if sev is not None: scored.append((sev, p))
        scored.sort(reverse=True)
        total = len(scored)
        if total: offset %= total
        out = []
        for sev, p in scored:
            if len(out) >= limit: break
            if offset > 0: offset -= 1; continue
            if self._check_file_access(p) is None:
                out.append({"file": p, "score": round(max(0.0, min(1.0, sev)), 2),
                            "severity": round(sev, 3), "below_threshold": sev <= 0.0,
                            "is_garbage": True, "kind": kind})
        return out

    def get_garbage_photos(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        return self.get_problem_photos("all", offset, limit)

    def count_problem_photos(self, kinds: Tuple[str, ...] = ("dark", "blurry", "overexposed", "blank"),
                             q: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Count indexed photos flagged by each problem kind under (candidate) thresholds."""
        counts = {k: 0 for k in kinds}
        for m in self.image_metadata.values():
            if m.get('deleted'): continue
            for k in kinds:
                if self._problem_score(k, m, q) > 0.0: counts[k] += 1
        return counts

    # --- Duplicate ranking helpers ---

    def _photo_counts_by_dir(self) -> Dict[str, int]:
        """Number of indexed photos per exact directory (cached; invalidated on index changes)."""
        if self._dir_counts is None:
            counts: Dict[str, int] = {}
            for p in self.image_paths:
                d = os.path.dirname(p); counts[d] = counts.get(d, 0) + 1
            self._dir_counts = counts
        return self._dir_counts

    def rank_duplicate_group(self, group: List[str]) -> List[str]:
        """Order a duplicate group best-copy-first (prefer real libraries over junk dirs)."""
        if not group: return list(group)
        counts = self._photo_counts_by_dir()

        def keyf(p: str) -> Tuple:
            rp = os.path.realpath(p); low = rp.lower()
            junk = any(frag in low for frag in DUPE_JUNK_FRAGMENTS)
            try: mtime = os.path.getmtime(rp)
            except OSError: mtime = 0.0
            return (not junk, counts.get(os.path.dirname(rp), 0),
                    -rp.count(os.sep), -len(rp), -mtime)

        return sorted(group, key=keyf, reverse=True)

    def rename_indexed_path(self, old: str, new: str) -> bool:
        """Update the index in memory after a file was moved on disk."""
        with self._lock:
            try: idx = self.image_paths.index(old)
            except ValueError: return False
            self.image_paths[idx] = new
            self.indexed_set.discard(old); self.indexed_set.add(new)
            if old in self.image_metadata:
                self.image_metadata[new] = self.image_metadata.pop(old)
            self._dir_counts = None; self._embeddings_dirty = True
            return True

    def save_index_files(self) -> None:
        """Atomically persist the current paths/metadata/embeddings."""
        with self._lock:
            temp_emb = str(self.embeddings_file) + ".tmp"
            temp_idx = str(self.index_file) + ".tmp"
            temp_meta = str(self.metadata_file) + ".tmp"
            if self.image_embeddings is not None: torch.save(self.image_embeddings, temp_emb)
            with open(temp_idx, "w", errors='surrogateescape') as f: json.dump(self.image_paths, f)
            with open(temp_meta, "w", errors='surrogateescape') as f: json.dump(self.image_metadata, f)
            if self.image_embeddings is not None: os.replace(temp_emb, str(self.embeddings_file))
            os.replace(temp_idx, str(self.index_file)); os.replace(temp_meta, str(self.metadata_file))

    @staticmethod
    def _unique_dest_path(dst_dir: str, filename: str) -> str:
        target = os.path.join(dst_dir, filename)
        if not os.path.exists(target): return target
        stem, ext = os.path.splitext(filename); i = 1
        while True:
            cand = os.path.join(dst_dir, f"{stem}_{i}{ext}")
            if not os.path.exists(cand): return cand
            i += 1

    def _indexed_photos_in_dir(self, directory: str) -> List[str]:
        prefix = os.path.realpath(directory) + os.sep
        return [p for p in self.image_paths if p.startswith(prefix)
                and not self.image_metadata.get(p, {}).get('deleted')]

    def plan_merge(self, src_dir: str, dst_dir: str, duplicate_groups: List[List[str]]) -> Tuple[List[str], List[str]]:
        """Return (unique_src_files, redundant_src_files) for merging src into dst."""
        src_prefix = os.path.realpath(src_dir) + os.sep
        dst_dir = os.path.realpath(dst_dir); dst_prefix = dst_dir + os.sep
        src_files = self._indexed_photos_in_dir(src_dir)
        redundant: Set[str] = set()
        for group in duplicate_groups:
            if any(f.startswith(dst_prefix) for f in group):
                for f in group:
                    if f.startswith(src_prefix): redundant.add(f)
        unique = [f for f in src_files if f not in redundant]
        return unique, sorted(redundant)

    def merge_directories(self, src_dir: str, dst_dir: str,
                          duplicate_groups: List[List[str]],
                          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Move src's unique indexed images into dst and trash src's redundant copies.

        Only indexed image files are touched; other files are left untouched so a
        non-mirror source directory is never silently emptied.
        """
        if progress: progress("Planning merge...")
        unique, redundant = self.plan_merge(src_dir, dst_dir, duplicate_groups)
        moved, trashed, failed = [], [], []
        for i, f in enumerate(unique):
            if progress and i % 25 == 0: progress(f"Moving unique files... {i}/{len(unique)}")
            dest = self._unique_dest_path(dst_dir, os.path.basename(f))
            try:
                shutil.move(f, dest)
                if self.rename_indexed_path(f, dest): moved.append((f, dest))
                else: moved.append((f, dest))
            except Exception as e:
                logging.warning(f"Merge move failed for {f!r}: {type(e).__name__}: {e}"); failed.append(f)
        for f in redundant:
            if QFile.moveToTrash(f): trashed.append(f)
            else: failed.append(f)
        if trashed: self.mark_photos_deleted(trashed)
        if moved or trashed: self.save_index_files()
        return {"moved": moved, "trashed": trashed, "failed": failed}

    def trash_directory_indexed(self, directory: str,
                                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Trash every indexed image in a directory (used for redundant mirror folders)."""
        targets = self._indexed_photos_in_dir(directory)
        trashed, failed = [], []
        for i, f in enumerate(targets):
            if progress and i % 25 == 0: progress(f"Moving to trash... {i}/{len(targets)}")
            if QFile.moveToTrash(f): trashed.append(f)
            else: failed.append(f)
        if trashed: self.mark_photos_deleted(trashed)
        return {"trashed": trashed, "failed": failed}

    def compute_folder_duplicate_stats(self, all_groups: List[List[str]]) -> List[Dict[str, Any]]:
        """Folder-pair duplicate overlap: shared files, wasted bytes, and coverage."""
        thr = float(self.quality_settings.get("folder_mirror_coverage", 0.90))
        dir_total = self._photo_counts_by_dir()
        pair: Dict[Tuple[str, str], Dict[str, int]] = {}
        for group in all_groups:
            by_dir: Dict[str, List[str]] = {}
            for f in group: by_dir.setdefault(os.path.dirname(f), []).append(f)
            dirs = list(by_dir.keys())
            for i in range(len(dirs)):
                a = dirs[i]
                for j in range(i + 1, len(dirs)):
                    b = dirs[j]
                    key = tuple(sorted([a, b]))
                    try: size = os.path.getsize(by_dir[a][0])
                    except OSError: size = 0
                    shared = min(len(by_dir[a]), len(by_dir[b]))
                    rec = pair.setdefault(key, {"shared": 0, "bytes": 0})
                    rec["shared"] += shared; rec["bytes"] += shared * size
        out = []
        for (a, b), rec in pair.items():
            ta, tb = dir_total.get(a, 0), dir_total.get(b, 0)
            cov_a = rec["shared"] / ta if ta else 0.0
            cov_b = rec["shared"] / tb if tb else 0.0
            if cov_a >= thr and cov_b >= thr: kind = "mirror"
            elif cov_a >= thr or cov_b >= thr: kind = "subset"
            else: kind = "partial"
            out.append({"a": a, "b": b, "shared": rec["shared"], "bytes": rec["bytes"],
                        "cov_a": cov_a, "cov_b": cov_b, "kind": kind,
                        "total_a": ta, "total_b": tb})
        out.sort(key=lambda r: r["bytes"], reverse=True)
        return out


# --- Threading Workers ---

class IndexWorker(QThread):
    progress = pyqtSignal(str, int, int); result_ready = pyqtSignal(int, float)
    def __init__(self, searcher: PhotoSearch, directory: str):
        super().__init__(); self.searcher, self.directory, self._cancelled = searcher, directory, False
    def cancel(self): self._cancelled = True
    def run(self):
        start_time = time.time()
        try:
            self.searcher.index_photos(self.directory, lambda m,c,t: self.progress.emit(m,c,t), lambda: self._cancelled)
        except Exception as e:
            logging.error(f"Indexing aborted due to error: {e}")
        self.result_ready.emit(0, time.time() - start_time)


class SearchWorker(QThread):
    result = pyqtSignal(list, list)

    def __init__(
        self,
        searcher: PhotoSearch,
        query: str,
        settings: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.searcher = searcher
        self.query = query
        self.settings = None if settings is None else dict(settings)

    def run(self):
        try:
            hits = self.searcher.search(self.query, settings=self.settings)
            self.result.emit(hits, list(self.searcher.last_search_warnings))
        except Exception as e:
            logging.exception(
                f"Search failed for query {self.query!r}: "
                f"{type(e).__name__}: {e}"
            )
            self.result.emit([], [])


class GarbageWorker(QThread):
    result = pyqtSignal(list)
    def __init__(self, searcher: PhotoSearch, kind: str = "all"):
        super().__init__(); self.searcher, self.kind = searcher, kind
    def run(self):
        self.result.emit(self.searcher.get_problem_photos(self.kind, getattr(self.searcher, "_garbage_offset", 0)))


class QualityPreviewWorker(QThread):
    """Counts photos matching each problem kind under candidate quality thresholds."""
    done = pyqtSignal(dict)
    def __init__(self, searcher: PhotoSearch, settings: Dict[str, Any]):
        super().__init__(); self.searcher, self.settings = searcher, dict(settings)
    def run(self):
        try: self.done.emit(self.searcher.count_problem_photos(q=self.settings))
        except Exception: self.done.emit({})


class QualityPhotoPreviewWorker(QThread):
    """Returns the worst photos for one kind under candidate quality thresholds."""
    result = pyqtSignal(list)
    def __init__(self, searcher: PhotoSearch, kind: str, settings: Dict[str, Any]):
        super().__init__(); self.searcher, self.kind, self.settings = searcher, kind, dict(settings)
    def run(self):
        try: self.result.emit(self.searcher.get_problem_photos(self.kind, 0, 50, q=self.settings))
        except Exception: self.result.emit([])


class CleanWorker(QThread):
    clean_complete = pyqtSignal(int)
    def __init__(self, searcher: PhotoSearch): super().__init__(); self.searcher = searcher
    def run(self): self.clean_complete.emit(self.searcher.clean_database())


class ThumbnailWorker(QThread):
    thumbnail_loaded = pyqtSignal(str, bytes, int, int, int)
    def __init__(self, files: List[str], searcher: Optional[PhotoSearch] = None):
        super().__init__(); self.files, self.searcher, self._cancelled = files, searcher, False
    def cancel(self) -> None: self._cancelled = True
    def run(self):
        for path in self.files:
            if self._cancelled: break
            try:
                thumb = self.searcher._get_cached_thumbnail(path) if self.searcher else None
                if not thumb:
                    with Image.open(path) as raw_img:
                        raw_img = ImageOps.exif_transpose(raw_img)
                        thumb = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img.copy()
                        thumb.thumbnail((200, 200), Image.Resampling.LANCZOS)
                data = thumb.tobytes("raw", "RGB"); w, h = thumb.width, thumb.height; thumb.close()
                self.thumbnail_loaded.emit(path, data, w, h, 3 * w)
            except Exception: pass


class DupesWorker(QThread):
    chunk_ready = pyqtSignal(list); scan_complete = pyqtSignal(float, int); error = pyqtSignal(str)
    cancelled = pyqtSignal(); status_update = pyqtSignal(str)
    def __init__(self, scan_dir: Optional[str] = None, indexed_set: Optional[Set[str]] = None):
        super().__init__(); self.scan_dir, self.indexed_set = scan_dir, indexed_set or set()
        self.process, self._stop_requested, self.temp_dir = None, False, None
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
            files_to_check = []
            if self.scan_dir:
                real_scan_dir = os.path.realpath(self.scan_dir)
                if not real_scan_dir.endswith(os.sep): real_scan_dir += os.sep
                files_to_check = [p for p in self.indexed_set if p.startswith(real_scan_dir)]
            if not files_to_check: self.scan_complete.emit(time.time() - start_time, 0); return
            self.temp_dir = tempfile.mkdtemp(prefix="photofind_scan_"); link_to_real = {}
            for i, fpath in enumerate(files_to_check):
                if self._stop_requested: self._cleanup_temp_dir(); self.cancelled.emit(); return
                if not os.path.exists(fpath): continue
                try:
                    link_path = os.path.join(self.temp_dir, f"{i:06d}_{os.path.basename(fpath)}")
                    os.symlink(fpath, link_path); link_to_real[link_path] = fpath
                except Exception: pass
            cmd = ['jdupes', '-r', '-s', self.temp_dir]; self.status_update.emit("Running jdupes...")
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_bytes, stderr_bytes = self.process.communicate()
            if self._stop_requested: self._cleanup_temp_dir(); self.cancelled.emit(); return
            if self.process.returncode == 2: self._cleanup_temp_dir(); self.error.emit(f"jdupes error (code {self.process.returncode})"); return
            stdout = stdout_bytes.decode('utf-8', errors='surrogateescape'); groups, current_group = [], []
            total_dupe_files = 0
            for line in stdout.splitlines():
                if line.strip() == "":
                    if len(current_group) > 1:
                        groups.append(current_group)
                        total_dupe_files += len(current_group)
                    if len(groups) >= 50: self.chunk_ready.emit(groups); groups = []
                    current_group = []
                else:
                    real_path = link_to_real.get(line.strip())
                    if real_path and real_path in self.indexed_set: current_group.append(real_path)
            if len(current_group) > 1:
                groups.append(current_group)
                total_dupe_files += len(current_group)
            if groups: self.chunk_ready.emit(groups)
            self.scan_complete.emit(time.time() - start_time, total_dupe_files)
        except FileNotFoundError: self.error.emit("'jdupes' is not installed.")
        except Exception as e:
            if not self._stop_requested: self.error.emit(f"Error: {str(e)}")
        finally: self._cleanup_temp_dir()


# --- GUI Classes ---

class DuplicateDialog(QDialog):
    _thumb_signal = pyqtSignal(str, bytes, int, int, int)
    def __init__(self, worker: DupesWorker, searcher: PhotoSearch, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.searcher = searcher; self.worker = worker; self.all_groups = []
        self.page_offset = 0; self.PAGE_SIZE = 50; self.is_streaming = True
        self.current_group_index = -1; self._path_to_item = {}
        self.setWindowTitle("Duplicate Manager (Scanning...)"); self.resize(1050, 700)
        self._thumb_signal.connect(self.update_thumbnail)
        self._thumb_stop = threading.Event(); self._thumb_threads = []; self._thumb_queue = queue.Queue()
        for _ in range(4):
            t = threading.Thread(target=self._thumb_worker_loop, daemon=True); t.start(); self._thumb_threads.append(t)
        main_layout = QVBoxLayout(self); content_layout = QHBoxLayout(); left_layout = QVBoxLayout()
        self.group_list = QListWidget(); self.group_list.setMaximumWidth(250)
        self.group_list.currentRowChanged.connect(self.load_group_images); left_layout.addWidget(self.group_list)
        self.next_page_btn = QPushButton("Load Next 50 Sets..."); self.next_page_btn.clicked.connect(self.load_next_page)
        self.next_page_btn.setEnabled(False); left_layout.addWidget(self.next_page_btn)
        content_layout.addLayout(left_layout)
        self.image_list = QListWidget(); self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(200, 200)); self.image_list.setResizeMode(QListWidget.ResizeMode.Fixed)
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.currentItemChanged.connect(self._handle_dupes_image_selection)
        self.image_list.itemDoubleClicked.connect(lambda item: self.open_in_explorer(item.data(Qt.ItemDataRole.UserRole)))
        self.image_list.itemSelectionChanged.connect(self.update_trash_btn_state)
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        content_layout.addWidget(self.image_list)
        btn_layout = QVBoxLayout()
        self.dup_folders_btn = QPushButton("Show Duplicate Folders"); self.dup_folders_btn.clicked.connect(self.show_duplicate_folders)
        self.trash_btn = QPushButton("Move Selected to Trash"); self.trash_btn.clicked.connect(self.trash_selected); self.trash_btn.setEnabled(False)
        self.trash_rest_btn = QPushButton("Keep Best, Trash Rest"); self.trash_rest_btn.clicked.connect(self.trash_rest); self.trash_rest_btn.setEnabled(False)
        btn_layout.addWidget(self.dup_folders_btn); btn_layout.addStretch()
        btn_layout.addWidget(self.trash_rest_btn); btn_layout.addWidget(self.trash_btn)
        content_layout.addLayout(btn_layout); main_layout.addLayout(content_layout)
        bottom_area = QHBoxLayout()
        self.info_label = QLabel("Waiting for jdupes to find duplicates...")
        self.info_label.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken); self.info_label.setMargin(5)
        self.stop_scan_btn = QPushButton("Stop Scan"); self.stop_scan_btn.clicked.connect(self.stop_scan); self.stop_scan_btn.setVisible(True)
        bottom_area.addWidget(self.info_label, 1); bottom_area.addWidget(self.stop_scan_btn, 0); main_layout.addLayout(bottom_area)

    def _thumb_worker_loop(self) -> None:
        while not self._thumb_stop.is_set():
            try: path = self._thumb_queue.get(timeout=0.5)
            except queue.Empty: continue
            if self._thumb_stop.is_set(): break
            try:
                thumb = self.searcher._get_cached_thumbnail(path)
                if not thumb:
                    with Image.open(path) as raw_img:
                        raw_img = ImageOps.exif_transpose(raw_img)
                        thumb = raw_img.convert("RGB") if raw_img.mode != "RGB" else raw_img.copy()
                        thumb.thumbnail((200, 200), Image.Resampling.LANCZOS)
                self._thumb_signal.emit(path, thumb.tobytes("raw", "RGB"), thumb.width, thumb.height, 3 * thumb.width)
                thumb.close()
            except Exception: pass

    def closeEvent(self, event):
        self._thumb_stop.set()
        for t in self._thumb_threads: t.join(timeout=1.0)
        super().closeEvent(event)

    def stop_scan(self):
        if self.worker: self.info_label.setText("Stopping..."); self.worker.stop()

    def add_groups_chunk(self, groups: List[List[str]]) -> None:
        ranked = [self.searcher.rank_duplicate_group(g) for g in groups]
        self.all_groups.extend(ranked)
        if self.page_offset >= len(self.all_groups) - self.PAGE_SIZE - len(ranked):
            self.page_offset = len(self.all_groups) - len(ranked); self.render_current_page()
        self.update_next_button_state()

    def stream_finished(self, duration: float, total_dupes: int) -> None:
        self.is_streaming = False; self.stop_scan_btn.setVisible(False); self.update_next_button_state()
        if not self.group_list.count() and self.all_groups: self.page_offset = 0; self.render_current_page()
        dur_str = format_duration(duration); count = len(self.all_groups)
        self.setWindowTitle(f"Duplicate Manager ({count} Total Sets Found)")
        self.info_label.setText(f"Scan complete ({dur_str}). Found {count} duplicate sets ({total_dupes} total files)." if count else f"Scan complete ({dur_str}). No duplicates found.")

    def update_status(self, msg: str) -> None: self.info_label.setText(msg)
    def update_next_button_state(self) -> None:
        has_more = (self.page_offset + self.PAGE_SIZE) < len(self.all_groups)
        self.next_page_btn.setEnabled(has_more or self.is_streaming); self.next_page_btn.setVisible(has_more or self.is_streaming)
    def load_next_page(self) -> None: self.page_offset += self.PAGE_SIZE; self.render_current_page()

    def render_current_page(self) -> None:
        self.group_list.clear(); self.image_list.clear(); self._path_to_item.clear()
        self.trash_btn.setEnabled(False); self.trash_rest_btn.setEnabled(False)
        self.info_label.setText("Select a set on the left")
        page_groups = self.all_groups[self.page_offset:self.page_offset + self.PAGE_SIZE]
        for i, group in enumerate(page_groups):
            self.group_list.addItem(QListWidgetItem(f"Set {self.page_offset + i+1}: {len(group)} files"))
        if page_groups: self.group_list.setCurrentRow(0); self.update_next_button_state()

    def update_trash_btn_state(self) -> None: self.trash_btn.setEnabled(len(self.image_list.selectedItems()) > 0)

    def load_group_images(self, list_row: int) -> None:
        self.image_list.clear(); self._path_to_item.clear()
        self.trash_btn.setEnabled(False); self.trash_rest_btn.setEnabled(False)
        self.info_label.setText("Select an image to see details")
        actual_index = self.page_offset + list_row
        if actual_index < 0 or actual_index >= len(self.all_groups): return
        self.current_group_index = actual_index; group = self.all_groups[actual_index]
        for i, path in enumerate(group):
            item = QListWidgetItem(); item.setData(Qt.ItemDataRole.UserRole, path); item.setSizeHint(QSize(200, 200))
            item.setToolTip(os.path.basename(path) + ("\n(Keep \u2014 best copy)" if i == 0 else ""))
            self.image_list.addItem(item); self._path_to_item[path] = item; self._thumb_queue.put(path)
        if len(group) > 1: self.trash_rest_btn.setEnabled(True)

    def _handle_dupes_image_selection(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current: self.show_image_info(current)

    def show_image_info(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            size_bytes = os.path.getsize(path)
            size_str = f"{size_bytes} B" if size_bytes < 1024 else (f"{size_bytes / 1024:.2f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB")
            self.info_label.setText(f"{path}  |  Size: {size_str}")
        except Exception as e: self.info_label.setText(f"{path}  |  Error: {e}")

    def update_thumbnail(self, path: str, data: bytes, w: int, h: int, bpl: int) -> None:
        try:
            item = self._path_to_item.get(path)
            if item is None: return
            try: _ = item.data(Qt.ItemDataRole.UserRole)
            except RuntimeError: return
            q_img = QImage(data, w, h, bpl, QImage.Format.Format_RGB888).copy()
            item.setIcon(QIcon(QPixmap.fromImage(q_img)))
        except RuntimeError: pass

    def open_in_explorer(self, path: str) -> None: open_in_file_browser(path)

    def copy_image_to_clipboard(self, path: str) -> None:
        copy_image_with_ui(self, path, self.info_label.setText)

    def show_image_context_menu(self, position: QPoint) -> None:
        item = self.image_list.itemAt(position)
        if not item: return
        if not item.isSelected(): self.image_list.clearSelection(); item.setSelected(True)
        selected_items = self.image_list.selectedItems()
        if not selected_items: return
        menu = QMenu()
        menu.addAction("Open in File Browser").triggered.connect(
            lambda: self.open_in_explorer(selected_items[0].data(Qt.ItemDataRole.UserRole)))
        menu.addAction("Copy Image to Clipboard").triggered.connect(
            lambda: self.copy_image_to_clipboard(selected_items[0].data(Qt.ItemDataRole.UserRole)))
        menu.addSeparator()
        delete_action = menu.addAction(f"Move {len(selected_items)} Photos to Trash")
        delete_action.triggered.connect(
            lambda: self._delete_paths([i.data(Qt.ItemDataRole.UserRole) for i in selected_items], selected_items))
        menu.exec(self.image_list.viewport().mapToGlobal(position))

    def trash_selected(self) -> None:
        sel = self.image_list.selectedItems()
        if sel: self._delete_paths([i.data(Qt.ItemDataRole.UserRole) for i in sel], sel)

    def trash_rest(self) -> None:
        if self.current_group_index < 0: return
        paths_to_delete = self.all_groups[self.current_group_index][1:]
        self._delete_paths(paths_to_delete,
            [i for i in [self.image_list.item(r) for r in range(self.image_list.count())]
             if i.data(Qt.ItemDataRole.UserRole) in paths_to_delete])

    def purge_file_from_all_groups(self, file_path: str) -> None:
        new_all_groups = []; changed = False
        for group in self.all_groups:
            new_group = [f for f in group if f != file_path]
            if len(new_group) < len(group): changed = True
            if len(new_group) > 1: new_all_groups.append(new_group)
        if changed:
            self.all_groups = new_all_groups
            self.page_offset = min(self.page_offset, max(0, len(self.all_groups) - self.PAGE_SIZE))
            self.render_current_page()
            self.setWindowTitle(f"Duplicate Manager ({len(self.all_groups)} Total Sets Found)")

    def _delete_paths(self, paths_to_delete: List[str], items_to_remove: List[QListWidgetItem]) -> None:
        if not paths_to_delete: return
        msg = (f"Move this file to Trash?\n{paths_to_delete[0]}" if len(paths_to_delete) == 1
               else f"Move {len(paths_to_delete)} files to Trash?")
        if QMessageBox.question(self, "Move to Trash", msg,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        success_paths = []; fail_paths = []
        for p in paths_to_delete:
            if QFile.moveToTrash(p): success_paths.append(p)
            else: fail_paths.append(p)
        if success_paths: self.searcher.mark_photos_deleted(success_paths)
        for p in success_paths: self.purge_file_from_all_groups(p)
        if fail_paths: QMessageBox.warning(self, "Error", f"Could not trash {len(fail_paths)} files.")

    def _open_in_explorer(self, path: str) -> None: open_in_file_browser(path)

    def show_duplicate_folders(self) -> None:
        stats = self.searcher.compute_folder_duplicate_stats(self.all_groups)
        if not stats:
            QMessageBox.information(self, "Duplicate Folders", "No overlapping duplicate folders found."); return

        dialog = QDialog(self); dialog.setWindowTitle("Duplicate Folders Analysis"); dialog.resize(860, 580)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "Folder pairs sharing duplicate files, largest potential waste first.\n"
            "mirror = both folders mostly duplicated; subset = one is contained in the other.\n"
            "Merge/drop touch only PhotoFind-indexed images; other files are left untouched."))
        list_widget = QListWidget()

        def fmt_bytes(n: int) -> str:
            return f"{n} B" if n < 1024 else (f"{n/1024:.1f} KB" if n < 1024*1024 else f"{n/(1024*1024):.1f} MB")

        def populate() -> None:
            list_widget.clear()
            for rec in self.searcher.compute_folder_duplicate_stats(self.all_groups)[:200]:
                line = (f"[{rec['kind'].upper()}] {rec['shared']} shared \u2022 {fmt_bytes(rec['bytes'])} potential waste\n"
                        f"  A ({rec['cov_a']*100:.0f}% dup of {rec['total_a']}) {rec['a']}\n"
                        f"  B ({rec['cov_b']*100:.0f}% dup of {rec['total_b']}) {rec['b']}")
                item = QListWidgetItem(line); item.setData(Qt.ItemDataRole.UserRole, rec); list_widget.addItem(item)
        populate()
        list_widget.itemDoubleClicked.connect(lambda item: self._open_in_explorer(item.data(Qt.ItemDataRole.UserRole)["a"]))
        layout.addWidget(list_widget)

        status = QLabel("Select a row, then choose an action.")
        status.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken); status.setMargin(5)
        layout.addWidget(status)

        btn_layout = QHBoxLayout()
        merge_ab = QPushButton("Merge A \u2192 B"); merge_ba = QPushButton("Merge B \u2192 A")
        drop_a = QPushButton("Drop Folder A"); drop_b = QPushButton("Drop Folder B")
        open_btn = QPushButton("Open\u2026"); close_btn = QPushButton("Close")
        for b in (merge_ab, merge_ba, drop_a, drop_b, open_btn): btn_layout.addWidget(b)
        btn_layout.addStretch(); btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        def current_rec() -> Optional[Dict[str, Any]]:
            item = list_widget.currentItem()
            return item.data(Qt.ItemDataRole.UserRole) if item else None

        def confirm_and_run(operation: str, src: str, dst: Optional[str], label: str) -> None:
            if operation == "merge":
                assert dst is not None
                unique, redundant = self.searcher.plan_merge(src, dst, self.all_groups)
                if not unique and not redundant:
                    QMessageBox.information(dialog, "Nothing to do", "No indexed images found to merge."); return
                msg = (f"Merge folder:\n  {src}\n\u2192 {dst}\n\n"
                       f"{len(unique)} unique file(s) will be MOVED to the destination.\n"
                       f"{len(redundant)} duplicate file(s) will be moved to Trash.\n\n"
                       f"Non-image files are left untouched. Continue?")
            else:
                targets = self.searcher._indexed_photos_in_dir(src)
                if not targets:
                    QMessageBox.information(dialog, "Nothing to do", "No indexed images found in that folder."); return
                msg = (f"Drop folder (move all indexed images to Trash):\n  {src}\n\n"
                       f"{len(targets)} indexed image(s) will be moved to Trash. Continue?")
            if QMessageBox.question(dialog, label, msg,
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            status.setText(f"{label}: working..."); QApplication.processEvents()
            try:
                if operation == "merge":
                    res = self.searcher.merge_directories(src, dst or "", self.all_groups)
                else:
                    res = self.searcher.trash_directory_indexed(src)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(dialog, "Error", f"Operation failed: {type(e).__name__}: {e}"); return
            QApplication.restoreOverrideCursor()
            for old, _new in res.get("moved", []): self.purge_file_from_all_groups(old)
            for p in res.get("trashed", []): self.purge_file_from_all_groups(p)
            populate()
            moved, trashed, failed = len(res.get("moved", [])), len(res.get("trashed", [])), len(res.get("failed", []))
            status.setText(f"Done: moved {moved}, trashed {trashed}, failed {failed}.")
            QMessageBox.information(dialog, label, f"Operation complete.\nMoved: {moved}\nTrashed: {trashed}\nFailed: {failed}")

        def wire(btn: QPushButton, operation: str, src_key: str, dst_key: Optional[str], label: str) -> None:
            def handler(_checked: bool = False) -> None:
                rec = current_rec()
                if not rec: return
                confirm_and_run(operation, rec[src_key], (rec[dst_key] if dst_key else None), label)
            btn.clicked.connect(handler)
        wire(merge_ab, "merge", "a", "b", "Merge A \u2192 B")
        wire(merge_ba, "merge", "b", "a", "Merge B \u2192 A")
        wire(drop_a, "drop", "a", None, "Drop Folder A")
        wire(drop_b, "drop", "b", None, "Drop Folder B")

        def on_open() -> None:
            rec = current_rec()
            if not rec: return
            menu = QMenu(open_btn)
            menu.addAction(f"Open A: {rec['a']}").triggered.connect(lambda _, p=rec["a"]: self._open_in_explorer(p))
            menu.addAction(f"Open B: {rec['b']}").triggered.connect(lambda _, p=rec["b"]: self._open_in_explorer(p))
            menu.exec(open_btn.mapToGlobal(QPoint(0, open_btn.height())))
        open_btn.clicked.connect(on_open)
        close_btn.clicked.connect(dialog.close)
        dialog.exec()


class SettingsDialog(QDialog):
    """Two-tab tuning dialog: search accuracy and photo-quality thresholds.

    Values are previewed live (debounced) against the loaded index/metadata and
    only persisted to ~/.cache/photofind/settings.json when Apply is pressed.
    """

    def __init__(self, searcher: PhotoSearch, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.searcher = searcher
        self.setWindowTitle("Search Tuning"); self.resize(580, 600)
        self._quality_worker = None
        self._last_counts: Dict[str, int] = {}
        self._last_preview: Optional[int] = None
        self._search_timer = QTimer(self); self._search_timer.setSingleShot(True); self._search_timer.setInterval(350)
        self._search_timer.timeout.connect(self._run_search_preview)
        self._quality_timer = QTimer(self); self._quality_timer.setSingleShot(True); self._quality_timer.setInterval(350)
        self._quality_timer.timeout.connect(self._run_quality_preview)
        # Live previews render in the main window, not just a count label.
        if parent is not None and hasattr(parent, "preview_results"):
            parent.preview_results.connect(self._on_search_preview)
        if parent is not None and hasattr(parent, "quality_preview_results"):
            parent.quality_preview_results.connect(self._on_quality_photos_preview)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs)
        self._build_search_tab(); self._build_quality_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        btns = QDialogButtonBox()
        reset_btn = btns.addButton("Reset to Defaults", QDialogButtonBox.ButtonRole.ResetRole)
        apply_btn = btns.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        cancel_btn = btns.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        reset_btn.clicked.connect(self._reset_defaults)
        apply_btn.clicked.connect(self._apply)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(btns)
        self._queue_search_preview(); self._queue_quality_preview()

    @staticmethod
    def _make_d(value: float, lo: float, hi: float, step: float, decimals: int) -> QDoubleSpinBox:
        sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(decimals)
        sb.setValue(value); return sb

    @staticmethod
    def _make_i(value: int, lo: int, hi: int) -> QSpinBox:
        sb = QSpinBox(); sb.setRange(lo, hi); sb.setValue(value); return sb

    def _build_search_tab(self) -> None:
        tab = QWidget(); form = QFormLayout(tab); s = self.searcher.search_settings
        parent = self.parent()
        initial_q = parent.search_input.text() if parent is not None and hasattr(parent, "search_input") else ""
        self.q_input = QLineEdit(initial_q); self.q_input.setPlaceholderText("Type a query to preview result counts")
        form.addRow("Preview query:", self.q_input)
        self.sp_top_k = self._make_i(int(s["top_k"]), 1, 500)
        self.sp_floor_min = self._make_d(float(s["floor_min"]), 0.0, 0.9, 0.01, 3)
        self.sp_floor_ratio = self._make_d(float(s["floor_ratio"]), 0.0, 1.0, 0.01, 3)
        self.sp_dedupe = self._make_d(float(s["dedupe_cos"]), 0.50, 1.0, 0.005, 3)
        form.addRow("Max results (top_k):", self.sp_top_k)
        form.addRow("Relevance floor (absolute):", self.sp_floor_min)
        form.addRow("Relevance floor (\u00d7 best hit):", self.sp_floor_ratio)
        form.addRow("Near-duplicate cutoff:", self.sp_dedupe)
        advanced = QGroupBox("Advanced"); aform = QFormLayout(advanced)
        self.sp_cand = self._make_i(int(s["candidate_mult"]), 1, 200)
        self.sp_black = self._make_d(float(s["black_lum_p95"]), 0.0, 255.0, 1.0, 1)
        aform.addRow("Candidate pool multiplier:", self.sp_cand)
        aform.addRow("Hide photos darker than (lum_p95):", self.sp_black)
        form.addRow(advanced)
        self.search_preview = QLabel("\u2014")
        self.search_preview.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken); self.search_preview.setMargin(5)
        form.addRow("Live preview:", self.search_preview)
        self.tabs.addTab(tab, "Search Accuracy")
        for w in (self.sp_top_k, self.sp_floor_min, self.sp_floor_ratio, self.sp_dedupe, self.sp_cand, self.sp_black):
            w.valueChanged.connect(self._queue_search_preview)
        self.q_input.textChanged.connect(self._queue_search_preview)

    def _build_quality_tab(self) -> None:
        tab = QWidget(); form = QFormLayout(tab); q = self.searcher.quality_settings
        dark = QGroupBox("Dark"); dform = QFormLayout(dark)
        self.sp_dark = self._make_d(float(q["dark_lum_p95"]), 0.0, 255.0, 1.0, 1)
        self.sp_dark_span = self._make_d(float(q["dark_span"]), 1.0, 255.0, 1.0, 1)
        dform.addRow("Darkness cutoff (lum_p95 below):", self.sp_dark)
        dform.addRow("Darkness span:", self.sp_dark_span)
        form.addRow(dark)
        blur = QGroupBox("Blurry"); bform = QFormLayout(blur)
        self.sp_blur = self._make_d(float(q["blur_lap_var"]), 0.0, 5000.0, 1.0, 1)
        self.sp_blur_span = self._make_d(float(q["blur_span"]), 0.5, 5000.0, 1.0, 1)
        self.sp_blur_min_edge = self._make_d(float(q["blur_min_edge"]), 0.0, 100.0, 0.5, 1)
        bform.addRow("Blur cutoff (lap_var below):", self.sp_blur)
        bform.addRow("Blur span:", self.sp_blur_span)
        bform.addRow("Ignore frames with edges below:", self.sp_blur_min_edge)
        form.addRow(blur)
        blank = QGroupBox("Blank / solid colour"); kform = QFormLayout(blank)
        self.sp_blank_std = self._make_d(float(q["blank_std_max"]), 0.0, 100.0, 0.5, 1)
        self.sp_blank_span = self._make_d(float(q["blank_span"]), 0.5, 100.0, 0.5, 1)
        kform.addRow("Blank cutoff (std_dev below):", self.sp_blank_std)
        kform.addRow("Blank span:", self.sp_blank_span)
        form.addRow(blank)
        over = QGroupBox("Overexposed"); oform = QFormLayout(over)
        self.sp_over_lum = self._make_d(float(q["over_lum_p95"]), 100.0, 255.0, 1.0, 1)
        self.sp_over_lum_span = self._make_d(float(q["over_lum_span"]), 1.0, 255.0, 1.0, 1)
        self.sp_over_mean = self._make_d(float(q["over_mean_lum"]), 0.0, 255.0, 1.0, 1)
        self.sp_over_mean_span = self._make_d(float(q["over_mean_span"]), 1.0, 255.0, 1.0, 1)
        oform.addRow("Highlights cutoff (lum_p95 above):", self.sp_over_lum)
        oform.addRow("Highlights span:", self.sp_over_lum_span)
        oform.addRow("Mean luminance cutoff (above):", self.sp_over_mean)
        oform.addRow("Mean luminance span:", self.sp_over_mean_span)
        form.addRow(over)
        folders = QGroupBox("Folders"); fform = QFormLayout(folders)
        self.sp_mirror = self._make_d(float(q["folder_mirror_coverage"]), 0.0, 1.0, 0.01, 2)
        fform.addRow("Folder mirror coverage:", self.sp_mirror)
        form.addRow(folders)
        self.preview_kind_combo = QComboBox()
        for value, label in (("all", "All Bad"), ("dark", "Dark"), ("blurry", "Blurry"),
                             ("overexposed", "Overexposed"), ("blank", "Blank / Solid")):
            self.preview_kind_combo.addItem(label, value)
        form.addRow("Preview in main window:", self.preview_kind_combo)
        self.quality_preview = QLabel("\u2014")
        self.quality_preview.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken); self.quality_preview.setMargin(5)
        form.addRow("Live preview:", self.quality_preview)
        self.tabs.addTab(tab, "Quality")
        self._quality_tab_index = self.tabs.count() - 1
        for w in (self.sp_dark, self.sp_dark_span, self.sp_blur, self.sp_blur_span,
                  self.sp_blur_min_edge, self.sp_blank_std, self.sp_blank_span,
                  self.sp_over_lum, self.sp_over_lum_span, self.sp_over_mean,
                  self.sp_over_mean_span, self.sp_mirror):
            w.valueChanged.connect(self._queue_quality_preview)
        self.preview_kind_combo.currentIndexChanged.connect(self._queue_quality_preview)

    def _collect_search(self) -> Dict[str, Any]:
        return {"top_k": int(self.sp_top_k.value()), "floor_min": float(self.sp_floor_min.value()),
                "floor_ratio": float(self.sp_floor_ratio.value()), "dedupe_cos": float(self.sp_dedupe.value()),
                "candidate_mult": int(self.sp_cand.value()), "black_lum_p95": float(self.sp_black.value())}

    def _collect_quality(self) -> Dict[str, Any]:
        return {"dark_lum_p95": float(self.sp_dark.value()), "dark_span": float(self.sp_dark_span.value()),
                "blur_lap_var": float(self.sp_blur.value()), "blur_span": float(self.sp_blur_span.value()),
                "blur_min_edge": float(self.sp_blur_min_edge.value()),
                "blank_std_max": float(self.sp_blank_std.value()), "blank_span": float(self.sp_blank_span.value()),
                "over_lum_p95": float(self.sp_over_lum.value()), "over_lum_span": float(self.sp_over_lum_span.value()),
                "over_mean_lum": float(self.sp_over_mean.value()), "over_mean_span": float(self.sp_over_mean_span.value()),
                "folder_mirror_coverage": float(self.sp_mirror.value())}

    def _queue_search_preview(self, *_args: Any) -> None: self._search_timer.start()
    def _queue_quality_preview(self, *_args: Any) -> None: self._quality_timer.start()

    def _on_tab_changed(self, index: int) -> None:
        # Preview the relevant tab in the main window when it becomes active, so
        # merely opening the dialog never clobbers the current results.
        if index == getattr(self, "_quality_tab_index", -1):
            self._queue_quality_preview()
        else:
            self._queue_search_preview()

    def _run_search_preview(self) -> None:
        query = self.q_input.text().strip()
        if self.searcher.image_embeddings is None:
            self.search_preview.setText("No index loaded."); return
        if not query:
            self.search_preview.setText("\u2014"); return
        parent = self.parent()
        if parent is not None and hasattr(parent, "preview_search"):
            self.search_preview.setText("Previewing in main window\u2026")
            parent.preview_search(query, self._collect_search())
        else:
            self.search_preview.setText("\u2014")

    def _on_search_preview(self, count: int, top: float) -> None:
        self.search_preview.setText(f"\u2248{count} photo(s) shown in main window  \u2022  top score {top:.3f}")

    def done(self, code: int) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "preview_results"):
            try: parent.preview_results.disconnect(self._on_search_preview)
            except (RuntimeError, TypeError): pass
        if parent is not None and hasattr(parent, "quality_preview_results"):
            try: parent.quality_preview_results.disconnect(self._on_quality_photos_preview)
            except (RuntimeError, TypeError): pass
        super().done(code)

    def _run_quality_preview(self) -> None:
        q = self._collect_quality()
        self.quality_preview.setText("Counting\u2026")
        w = QualityPreviewWorker(self.searcher, q)
        self._quality_worker = w; w.done.connect(self._on_quality_preview); w.finished.connect(w.deleteLater); w.start()
        # Only drive the main window from the Quality tab, so opening the dialog
        # to tune search does not replace the current results.
        parent = self.parent()
        if (parent is not None and hasattr(parent, "preview_problem_photos")
                and self.tabs.currentIndex() == getattr(self, "_quality_tab_index", -1)):
            self._last_preview = None
            parent.preview_problem_photos(str(self.preview_kind_combo.currentData()), q)

    def _on_quality_preview(self, counts: Dict[str, int]) -> None:
        self._last_counts = counts or {}
        self._refresh_quality_label()

    def _on_quality_photos_preview(self, count: int) -> None:
        self._last_preview = count
        self._refresh_quality_label()

    def _refresh_quality_label(self) -> None:
        c = self._last_counts
        if c:
            text = (f"dark {c.get('dark', 0)}  \u2022  blurry {c.get('blurry', 0)}  "
                    f"\u2022  overexposed {c.get('overexposed', 0)}  \u2022  blank {c.get('blank', 0)}")
        else:
            text = "No metadata loaded yet."
        if self._last_preview is not None:
            text += f"   \u2014   showing {self._last_preview} \u00d7 {self.preview_kind_combo.currentText()} in main window"
        self.quality_preview.setText(text)

    def _reset_defaults(self) -> None:
        s, q = DEFAULT_SEARCH_SETTINGS, DEFAULT_QUALITY_SETTINGS
        self.sp_top_k.setValue(int(s["top_k"])); self.sp_floor_min.setValue(float(s["floor_min"]))
        self.sp_floor_ratio.setValue(float(s["floor_ratio"])); self.sp_dedupe.setValue(float(s["dedupe_cos"]))
        self.sp_cand.setValue(int(s["candidate_mult"])); self.sp_black.setValue(float(s["black_lum_p95"]))
        self.sp_dark.setValue(float(q["dark_lum_p95"])); self.sp_dark_span.setValue(float(q["dark_span"]))
        self.sp_blur.setValue(float(q["blur_lap_var"])); self.sp_blur_span.setValue(float(q["blur_span"]))
        self.sp_blur_min_edge.setValue(float(q["blur_min_edge"]))
        self.sp_blank_std.setValue(float(q["blank_std_max"])); self.sp_blank_span.setValue(float(q["blank_span"]))
        self.sp_over_lum.setValue(float(q["over_lum_p95"])); self.sp_over_lum_span.setValue(float(q["over_lum_span"]))
        self.sp_over_mean.setValue(float(q["over_mean_lum"])); self.sp_over_mean_span.setValue(float(q["over_mean_span"]))
        self.sp_mirror.setValue(float(q["folder_mirror_coverage"]))
        idx = self.preview_kind_combo.findData("all")
        if idx >= 0: self.preview_kind_combo.setCurrentIndex(idx)
        self._queue_search_preview(); self._queue_quality_preview()

    def _apply(self) -> None:
        self.searcher.save_settings(search=self._collect_search(), quality=self._collect_quality())
        self.accept()


class FolderChoiceDialog(QDialog):
    """Pick one folder from a list, with "Open in File Manager" on right-click."""

    def __init__(self, title: str, prompt: str, labels: List[str], folders: List[str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(640, 360)
        layout = QVBoxLayout(self)
        prompt_label = QLabel(prompt); prompt_label.setWordWrap(True); layout.addWidget(prompt_label)
        self.list_widget = QListWidget()
        for text, folder in zip(labels, folders):
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            item.setToolTip(folder)
            self.list_widget.addItem(item)
        if self.list_widget.count(): self.list_widget.setCurrentRow(0)
        # Double-clicking a folder accepts it, like QInputDialog.getItem().
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.list_widget.itemAt(position)
        if not item: return
        if not item.isSelected(): self.list_widget.clearSelection(); item.setSelected(True)
        self.list_widget.setCurrentItem(item)
        menu = QMenu(self)
        menu.addAction("Open in File Manager").triggered.connect(
            lambda: open_in_file_browser(item.data(Qt.ItemDataRole.UserRole)))
        menu.exec(self.list_widget.viewport().mapToGlobal(position))

    def selected_folder(self) -> Optional[str]:
        item = self.list_widget.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None


class PhotoOrganizerWindow(QMainWindow):
    preview_results = pyqtSignal(int, float)
    quality_preview_results = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoFind"); self.resize(1200, 800)
        self.has_jdupes = shutil.which('jdupes') is not None
        if not self.has_jdupes: logging.info("'jdupes' not found. Duplicate detection disabled.")
        self.setup_ui(); self.setup_menu(); self.setup_shortcuts()
        self.statusBar().showMessage("Loading AI Model... (this may take a moment)"); QApplication.processEvents()
        self.searcher = PhotoSearch()
        self._sync_viewer_combo()
        self._viewer_reap_timer = QTimer(self)
        self._viewer_reap_timer.timeout.connect(reap_viewer_processes)
        self._viewer_reap_timer.start(5000)
        self.current_results, self.thumb_worker, self.dupes_worker, self.dupes_dialog = [], None, None, None
        self.index_worker, self.search_worker, self.garbage_worker, self.clean_worker = None, None, None, None
        self._preview_worker = None
        self._quality_preview_worker = None
        self._path_to_item = {}
        
        self._is_indexing = False
        self._pending_search_query = None
        self._garbage_kind = None
        self._garbage_page = 0

        if self.searcher.load_index(): self.statusBar().showMessage(self._get_idle_status_message())
        else: self.statusBar().showMessage("Ready. No index loaded.")

        sample_size = min(100, len(self.searcher.image_paths)); sample = random.sample(self.searcher.image_paths, sample_size) if self.searcher.image_paths else []
        missing = sum(1 for p in sample if not os.path.exists(p))
        if missing > 0: self.statusBar().showMessage(f"Warning: {missing}/{sample_size} sampled images missing. Consider 'Clean Database'.")
        else: self.statusBar().showMessage(self._get_idle_status_message())

    def _get_idle_status_message(self) -> str:
        parts = [f"Ready. {len(self.searcher.indexed_set)} images indexed."]; stats = self.searcher.stats
        if "last_index_count" in stats: parts.append(f"(last scan {stats['last_index_count']} images took {stats['last_index_duration']})")
        if "last_dedupe_count" in stats: parts.append(f"(last dedupe {stats['last_dedupe_count']} files took {stats['last_dedupe_duration']})")
        return " ".join(parts)

    def _show_temp_status(self, message: str, duration_ms: int = 5000) -> None:
        self.statusBar().showMessage(message); QTimer.singleShot(duration_ms, lambda: self.statusBar().showMessage(self._get_idle_status_message()))

    def closeEvent(self, event):
        try:
            if self._is_indexing and self.index_worker: self.index_worker.cancel()
            if self.dupes_worker and self.dupes_worker.isRunning(): self.dupes_worker.stop()
        except RuntimeError: pass
        if self.dupes_dialog and self.dupes_dialog.isVisible(): self.dupes_dialog.close()
        for w in [self.index_worker, self.search_worker, self.garbage_worker, self.clean_worker, self.dupes_worker]:
            if w is not None:
                try:
                    if w.isRunning(): w.wait(2000)
                except RuntimeError: pass
        try:
            if self.thumb_worker and self.thumb_worker.isRunning(): self.thumb_worker.cancel(); self.thumb_worker.wait(1000)
        except RuntimeError: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        event.accept()

    def setup_ui(self) -> None:
        central_widget = QWidget(); self.setCentralWidget(central_widget); layout = QVBoxLayout(central_widget); search_layout = QHBoxLayout()
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Type a description..."); self.search_input.returnPressed.connect(self.start_search)
        search_btn = QPushButton("Search"); search_btn.clicked.connect(self.start_search)
        self.garbage_btn = QToolButton(); self.garbage_btn.setText("Find Bad Photos")
        self.garbage_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.garbage_btn.setToolTip("Find the worst photos: Dark, Blurry, Overexposed, Blank, or All")
        gmenu = QMenu(self.garbage_btn)
        for kind, label in (("all", "All Bad"), ("dark", "Dark"), ("blurry", "Blurry"),
                            ("overexposed", "Overexposed"), ("blank", "Blank / Solid")):
            act = gmenu.addAction(f"Find {label} Photos")
            act.triggered.connect(lambda checked=False, k=kind: self.next_garbage_page(k))
        self.garbage_btn.setMenu(gmenu)
        search_layout.addWidget(self.search_input); search_layout.addWidget(search_btn); search_layout.addWidget(self.garbage_btn); layout.addLayout(search_layout)
        bottom_layout = QHBoxLayout(); self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setTextVisible(True)
        self.stop_index_btn = QPushButton("Stop Indexing"); self.stop_index_btn.setVisible(False); self.stop_index_btn.clicked.connect(self.cancel_current_operation)
        bottom_layout.addWidget(self.progress_bar, 1); bottom_layout.addWidget(self.stop_index_btn, 0); layout.addLayout(bottom_layout)
        self.list_widget = QListWidget(); self.list_widget.setViewMode(QListWidget.ViewMode.IconMode); self.list_widget.setIconSize(QSize(200, 200))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust); self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.itemDoubleClicked.connect(self.open_image); self.list_widget.currentItemChanged.connect(self.show_path_in_status)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.list_widget); self.setStatusBar(QStatusBar())

    def setup_menu(self) -> None:
        toolbar = QToolBar("Main"); self.addToolBar(toolbar)
        index_action = QAction("Index Folder", self); index_action.triggered.connect(self.select_folder_to_index); toolbar.addAction(index_action)
        self.dupes_action = QAction("Find Duplicates", self); self.dupes_action.setToolTip("Scan directories for exact byte-for-byte duplicates"); self.dupes_action.triggered.connect(self.start_global_dedupe)
        if not self.has_jdupes: self.dupes_action.setEnabled(False); self.dupes_action.setToolTip("jdupes is not installed")
        toolbar.addAction(self.dupes_action)
        clean_action = QAction("Clean Database", self); clean_action.setToolTip("Remove entries for files that no longer exist"); clean_action.triggered.connect(self.clean_database); toolbar.addAction(clean_action)
        tune_action = QAction("Search Tuning\u2026", self); tune_action.setToolTip("Adjust search-accuracy and photo-quality thresholds"); tune_action.triggered.connect(self.open_settings); toolbar.addAction(tune_action)
        clear_action = QAction("Clear Screen", self); clear_action.setToolTip("Clear search results and reset UI"); clear_action.triggered.connect(self.clear_screen); toolbar.addAction(clear_action)

        # Image-viewer selector: dynamically populated from installed desktop
        # apps. Selecting one persists it so opened photos use that viewer.
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Open with: "))
        self.viewer_combo = QComboBox(); self.viewer_combo.setToolTip("Application used to open a photo (double-click / Enter)")
        self.viewer_combo.addItem("System default", "system")
        for cmd, label in detect_image_viewers():
            self.viewer_combo.addItem(label, cmd)
        self.viewer_combo.setMinimumWidth(160)
        self.viewer_combo.currentIndexChanged.connect(self._on_viewer_changed)
        toolbar.addWidget(self.viewer_combo)

    def _on_viewer_changed(self, index: int) -> None:
        cmd = str(self.viewer_combo.itemData(index) or "system")
        if getattr(self, "searcher", None) is None: return
        self.searcher.save_settings(viewer={"image_viewer": cmd})
        self._show_temp_status(f"Photos will open with {self.viewer_combo.itemText(index)}.")

    def _sync_viewer_combo(self) -> None:
        """Select the persisted viewer in the toolbar combo (adding it if needed)."""
        cmd = str(self.searcher.viewer_settings.get("image_viewer", "system"))
        idx = self.viewer_combo.findData(cmd)
        if idx < 0 and cmd != "system":
            # Tolerate absolute paths: match an installed entry by executable name.
            base = os.path.basename(cmd)
            for i in range(self.viewer_combo.count()):
                if os.path.basename(str(self.viewer_combo.itemData(i) or "")) == base:
                    idx = i
                    break
        if idx < 0 and cmd != "system" and _find_executable(cmd):
            self.viewer_combo.addItem(f"Custom  ({cmd})", cmd); idx = self.viewer_combo.count() - 1
        if idx < 0:
            idx = max(0, self.viewer_combo.findData("system"))
        self.viewer_combo.blockSignals(True)
        self.viewer_combo.setCurrentIndex(idx)
        self.viewer_combo.blockSignals(False)

    def setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+F"), self, self.search_input.setFocus)
        QShortcut(QKeySequence("Ctrl+A"), self, self.list_widget.selectAll)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self.list_widget, self._copy_current_image, context=Qt.ShortcutContext.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence("Escape"), self, self.cancel_current_operation)
        # Bind Enter specifically to the list widget using WidgetShortcut 
        # so it doesn't intercept Enter from the search box
        QShortcut(QKeySequence(Qt.Key.Key_Return), self.list_widget, self._open_selected_image, context=Qt.ShortcutContext.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self.list_widget, self._open_selected_image, context=Qt.ShortcutContext.WidgetShortcut)

    def _open_selected_image(self) -> None:
        current = self.list_widget.currentItem()
        if current:
            self.open_image(current)

    def _format_file_size(self, path: str) -> str:
        try:
            size_bytes = os.path.getsize(path)
            return f"{size_bytes} B" if size_bytes < 1024 else (f"{size_bytes / 1024:.2f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB")
        except Exception: return "Unknown size"

    def clear_screen(self) -> None:
        self.list_widget.clear(); self._path_to_item.clear(); self.search_input.clear(); self.current_results = []; self.statusBar().showMessage(self._get_idle_status_message())

    def delete_selected(self) -> None:
        sel = self.list_widget.selectedItems()
        if sel: self.delete_photos_action([i.data(Qt.ItemDataRole.UserRole) for i in sel], sel)

    def cancel_current_operation(self) -> None:
        try:
            if self._is_indexing:
                self.index_worker.cancel(); self.statusBar().showMessage("Cancelling indexing...")
            elif self.dupes_worker and self.dupes_worker.isRunning():
                if self.dupes_dialog: self.dupes_dialog.stop_scan()
        except RuntimeError: pass

    def clean_database(self) -> None:
        if QMessageBox.question(self, 'Clean Database', 'Remove entries for files that no longer exist on disk?\n\nContinue?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Cleaning database..."); self.clean_worker = CleanWorker(self.searcher)
            self.clean_worker.clean_complete.connect(self.clean_finished); self.clean_worker.finished.connect(self.clean_worker.deleteLater); self.clean_worker.start()

    def clean_finished(self, removed_count: int) -> None: self._show_temp_status(f"Clean complete. Removed {removed_count} missing files."); self.searcher.load_index()

    def show_context_menu(self, position: QPoint) -> None:
        item = self.list_widget.itemAt(position)
        if not item: return
        if not item.isSelected(): self.list_widget.clearSelection(); item.setSelected(True)
        selected_items = self.list_widget.selectedItems(); selected_count = len(selected_items); menu = QMenu()
        if selected_count == 1:
            path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            menu.addAction("Open in Image Viewer").triggered.connect(lambda: self.open_image_viewer(path))
            copy_action = menu.addAction("Copy Image to Clipboard")
            copy_action.triggered.connect(lambda: self._copy_photo_to_clipboard(path))
            menu.addAction("Open in File Browser").triggered.connect(lambda: open_in_file_browser(path))
            menu.addAction("Exclude Folder from Index\u2026").triggered.connect(lambda: self.exclude_folder_dialog(path))
            menu.addSeparator()
        delete_action = menu.addAction(f"Move {selected_count} Photos to Trash" if selected_count > 1 else "Move to Trash")
        delete_action.triggered.connect(lambda: self.delete_photos_action([i.data(Qt.ItemDataRole.UserRole) for i in selected_items], selected_items))
        menu.exec(self.list_widget.viewport().mapToGlobal(position))

    def _copy_current_image(self) -> None:
        item = self.list_widget.currentItem()
        if item: self._copy_photo_to_clipboard(item.data(Qt.ItemDataRole.UserRole))

    def _copy_photo_to_clipboard(self, path: str) -> None:
        err = self.searcher._check_file_access(path)
        if err: self._show_temp_status(err); return
        copy_image_with_ui(self, path, self._show_temp_status)

    def delete_photos_action(self, paths: List[str], items: List[QListWidgetItem]) -> None:
        count = len(paths)
        if count == 0: return
        msg = f'Move this file to Trash?\n{paths[0]}' if count == 1 else f'Move {count} selected files to Trash?'
        if QMessageBox.question(self, 'Move to Trash', msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            success, fail = [], []
            for p in paths:
                if QFile.moveToTrash(p): success.append(p)
                else: fail.append(p)
            if success: self.searcher.mark_photos_deleted(success)
            for item in reversed([i for i in items if i.data(Qt.ItemDataRole.UserRole) in success]): self.list_widget.takeItem(self.list_widget.row(item))
            if fail: QMessageBox.warning(self, "Error", f"Could not move {len(fail)} files to trash.")
            else: self._show_temp_status(f"Moved {count} photos to Trash.")

    def _indexed_count_under(self, folder: str) -> int:
        prefix = folder.rstrip(os.sep) + os.sep
        return sum(1 for p in self.searcher.image_paths if p.startswith(prefix))

    def exclude_folder_dialog(self, path: str) -> None:
        """Let the user pick which ancestor folder to exclude (e.g. ~/music),
        not just the file's immediate parent."""
        start = Path(os.path.realpath(path)).parent
        ancestors = []
        node = start
        while True:
            ancestors.append(node)
            if node.parent == node: break
            node = node.parent
        labels, folders = [], []
        for anc in ancestors:
            folder = str(anc)
            count = self._indexed_count_under(folder)
            labels.append(f"{folder}    ({count} indexed photo{'' if count == 1 else 's'})")
            folders.append(folder)
        if not folders: return
        dialog = FolderChoiceDialog(
            "Exclude Folder from Index",
            "Choose the folder to exclude (its subfolders are excluded too):",
            labels, folders, self)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        folder = dialog.selected_folder()
        if not folder: return
        count = self._indexed_count_under(folder)
        msg = (f"Exclude this folder from the index?\n\n{folder}\n\n"
               f"{count} indexed photo(s) will be removed now. Future indexing will skip "
               f"this folder and all its subfolders.")
        if QMessageBox.question(self, 'Exclude Folder', msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            try:
                removed = self.searcher.exclude_directory(folder)
                self.list_widget.clear(); self._path_to_item.clear(); self.current_results = []
                self._show_temp_status(f"Excluded {folder}. Removed {removed} indexed photos.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to exclude folder: {e}")

    def _launch_image_viewer(self, command: str, path: str) -> bool:
        """Launch ``command`` on ``path`` detached from this process.

        ``command`` may be a full command line (e.g. ``flatpak run org.x.Y``);
        the photo path is appended. Returns True if the process was started.
        The viewer is put in its own session so it keeps running if PhotoFind
        is closed.
        """
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = [command]
        # Geeqie is single-instance: launched normally it hands the file to an
        # already-open window and waits on the hand-off, which can lag. Force a
        # fresh instance so the window appears immediately, and ask for
        # full-screen (it otherwise opens windowed behind the main window).
        env = None
        if "geeqie" in command.lower():
            env = {**os.environ, "GQ_NEW_INSTANCE": "yes"}
            argv = [*argv, *GEEQIE_EXTRA_ARGS]
        argv = [*argv, path]
        try:
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception as e:
            logging.warning(f"Failed to launch image viewer {command!r}: {type(e).__name__}: {e}")
            return False
        _VIEWER_PROCESSES.append(proc)
        reap_viewer_processes()
        return True

    def open_image_viewer(self, path: str) -> None:
        err = self.searcher._check_file_access(path)
        if err: self._show_temp_status(err); return

        # 1) User's explicitly chosen viewer (Settings -> Viewers).
        choice = str(self.searcher.viewer_settings.get("image_viewer", "system"))
        if choice and choice != "system":
            if self._launch_image_viewer(choice, path):
                return
            self._show_temp_status(f"Could not launch '{choice}'; trying system default.")

        # 2) System default.
        # On the Qt version whose KIO worker crashes, bypass the desktop
        # association and open a detected viewer directly.
        if QLibraryInfo.version().toString() == "6.10.3":
            for cmd, _label in detect_image_viewers():
                if self._launch_image_viewer(cmd, path):
                    return
            logging.warning("No image viewer found. Falling back to system default (may crash).")

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            # Desktop association refused; try a detected viewer directly.
            for cmd, _label in detect_image_viewers():
                if self._launch_image_viewer(cmd, path):
                    return
            self._show_temp_status("Could not open image viewer.")

    def start_global_dedupe(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Root Photo Folder to Scan")
        if folder:
            self.statusBar().showMessage("Running jdupes..."); self.dupes_action.setEnabled(False)
            self.dupes_worker = DupesWorker(scan_dir=folder, indexed_set=self.searcher.indexed_set)
            self.dupes_dialog = DuplicateDialog(self.dupes_worker, self.searcher, self); self.dupes_dialog.show()
            self.dupes_worker.chunk_ready.connect(self.dupes_dialog.add_groups_chunk); self.dupes_worker.scan_complete.connect(self.dupes_finished)
            self.dupes_worker.error.connect(self.dupes_error); self.dupes_worker.cancelled.connect(self.dupes_cancelled)
            self.dupes_worker.status_update.connect(self.dupes_dialog.update_status); self.dupes_worker.finished.connect(self.dupes_worker.deleteLater); self.dupes_worker.start()

    def dupes_finished(self, duration: float, total_dupes: int, message_override: Optional[str] = None) -> None:
        if self.dupes_dialog: self.dupes_dialog.stream_finished(duration, total_dupes)
        self.reset_dupes_ui_state(); self.searcher.save_dedupe_stats(duration, total_dupes)
        if message_override: self._show_temp_status(message_override)
        else: self.statusBar().showMessage(self._get_idle_status_message())

    def dupes_cancelled(self) -> None: self.dupes_finished(0.0, 0, message_override="Duplicate scan cancelled.")
    def dupes_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Deduplication Error", msg);
        if self.dupes_dialog: self.dupes_dialog.close()
        self.reset_dupes_ui_state(); self._show_temp_status("Duplicate scan failed.")
    def reset_dupes_ui_state(self) -> None: self.dupes_action.setEnabled(self.has_jdupes); self.dupes_dialog = None

    def select_folder_to_index(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.statusBar().showMessage(f"Indexing {folder}..."); self.progress_bar.setVisible(True); self.stop_index_btn.setVisible(True)
            self.progress_bar.setRange(0, 0); self.progress_bar.setFormat("Scanning...")
            self._is_indexing = True
            self.index_worker = IndexWorker(self.searcher, folder); self.index_worker.progress.connect(self.indexing_progress)
            self.index_worker.result_ready.connect(self.indexing_finished); self.index_worker.finished.connect(self.index_worker.deleteLater); self.index_worker.start()

    def indexing_progress(self, msg: str, current: int, total: int) -> None:
        if total > 0: self.progress_bar.setRange(0, total); self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{msg} (%p%)"); self.statusBar().showMessage(msg)

    def indexing_finished(self, count: int, duration: float) -> None:
        self.progress_bar.setVisible(False); self.stop_index_btn.setVisible(False)
        self.searcher.save_index_stats(duration, count)
        
        self._is_indexing = False
        
        if self._pending_search_query:
            query = self._pending_search_query
            self._pending_search_query = None
            self.search_input.setText(query); self.start_search()
        else:
            self.statusBar().showMessage(self._get_idle_status_message())

    def start_search(self) -> None:
        query = self.search_input.text().strip()
        if not query: return
        
        if self._is_indexing:
            self._pending_search_query = query
            self.statusBar().showMessage("Indexing in progress. Search will execute automatically when finished...")
            return

        self.list_widget.clear(); self._path_to_item.clear(); self.statusBar().showMessage("Searching...")
        if self.search_worker is not None:
            try: self.search_worker.result.disconnect(self.display_results)
            except (RuntimeError, TypeError): pass
        self.search_worker = SearchWorker(self.searcher, query); self.search_worker.result.connect(self.display_results)
        self.search_worker.finished.connect(self.search_worker.deleteLater); self.search_worker.start()

    def next_garbage_page(self, kind: str = "all") -> None:
        if getattr(self, "_garbage_kind", None) != kind:
            self._garbage_kind = kind; self._garbage_page = 0
        page = getattr(self, "_garbage_page", 0)
        self.searcher._garbage_offset = page * 50
        self._garbage_page = page + 1
        self.find_garbage(kind)

    def find_garbage(self, kind: str = "all") -> None:
        self.list_widget.clear(); self._path_to_item.clear()
        self.statusBar().showMessage(f"Scanning metadata for {kind} photos...")
        if self.garbage_worker is not None:
            try: self.garbage_worker.result.disconnect(self.display_results)
            except (RuntimeError, TypeError): pass
        self.garbage_worker = GarbageWorker(self.searcher, kind); self.garbage_worker.result.connect(self.display_results)
        self.garbage_worker.finished.connect(self.garbage_worker.deleteLater); self.garbage_worker.start()

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.searcher, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted: self._show_temp_status("Thresholds updated.")
        # Re-run with the (saved) settings so the main window reflects reality
        # whether the user applied or cancelled a live preview.
        if self.search_input.text().strip():
            self.start_search()
        elif getattr(self, "_garbage_kind", None):
            self.find_garbage(self._garbage_kind)

    def preview_search(self, query: str, settings: Dict[str, Any]) -> None:
        """Run a query with candidate search thresholds and show it live in the
        main window, without persisting the settings."""
        query = (query or "").strip()
        if not query or self.searcher.image_embeddings is None or self._is_indexing: return
        self.list_widget.clear(); self._path_to_item.clear()
        self.statusBar().showMessage("Previewing search with tuning\u2026")
        if self._preview_worker is not None:
            try: self._preview_worker.result.disconnect()
            except (RuntimeError, TypeError): pass
        self._preview_worker = SearchWorker(self.searcher, query, settings=settings)
        self._preview_worker.result.connect(self.display_results)
        self._preview_worker.finished.connect(self._preview_worker.deleteLater)
        self._preview_worker.start()

    def preview_problem_photos(self, kind: str, settings: Dict[str, Any]) -> None:
        """Show the worst photos for `kind` under candidate quality thresholds in
        the main window, without persisting the settings."""
        if self._is_indexing or not self.searcher.image_metadata: return
        if self._quality_preview_worker is not None:
            try: self._quality_preview_worker.result.disconnect()
            except (RuntimeError, TypeError): pass
        # Remember the kind so the view is restored to real data when the
        # tuning dialog closes.
        self._garbage_kind = kind; self._garbage_page = 0; self.searcher._garbage_offset = 0
        self.statusBar().showMessage(f"Previewing {kind} photos with tuning\u2026")
        self._quality_preview_worker = QualityPhotoPreviewWorker(self.searcher, kind, settings)
        self._quality_preview_worker.result.connect(self._on_quality_photos)
        self._quality_preview_worker.finished.connect(self._quality_preview_worker.deleteLater)
        self._quality_preview_worker.start()

    def _on_quality_photos(self, hits: List[Dict[str, Any]]) -> None:
        self.display_results(hits, emit_preview=False)
        self.quality_preview_results.emit(len(hits))

    def display_results(self, hits: List[Dict[str, Any]], warnings: Optional[List[str]] = None,
                        emit_preview: bool = True) -> None:
        self.list_widget.clear(); self._path_to_item.clear(); self.current_results = hits
        warn_suffix = f" Warning: {'; '.join(warnings)}" if warnings else ""
        top_score = hits[0].get('score', 0.0) if hits else 0.0
        if emit_preview: self.preview_results.emit(len(hits), float(top_score))
        if not hits:
            self._show_temp_status(f"No results found.{warn_suffix}" if warnings else "No results found."); return
        self.statusBar().showMessage(f"Found {len(hits)} results. Loading thumbnails...{warn_suffix}"); paths_to_load = []
        for hit in hits:
            path, score = hit['file'], hit.get('score', 0.0); item = QListWidgetItem(); item.setData(Qt.ItemDataRole.UserRole, path); item.setSizeHint(QSize(200, 200))

            if hit.get('is_garbage'):
                meta = self.searcher.image_metadata.get(path, {})
                kind = hit.get('kind', 'all')
                note = "  (below cutoff \u2014 among the worst available)" if hit.get('below_threshold') else ""
                tooltip = (
                    f"{os.path.basename(path)}\n"
                    f"Size: {self._format_file_size(path)}\n"
                    f"[{kind.upper()} {score:.2f}] lum_p95={meta.get('lum_p95', -1):.1f}, "
                    f"mean_lum={meta.get('mean_lum', -1):.1f}, "
                    f"lap_var={meta.get('lap_var', -1):.1f}, std_dev={meta.get('std_dev', -1):.1f}{note}"
                )
            else:
                tooltip = f"{os.path.basename(path)}\nSize: {self._format_file_size(path)}\nScore: {score:.3f}"

            item.setToolTip(tooltip)
            self.list_widget.addItem(item)
            self._path_to_item[path] = item
            paths_to_load.append(path)

        try:
            if self.thumb_worker and self.thumb_worker.isRunning(): self.thumb_worker.cancel(); self.thumb_worker.wait(500)
        except RuntimeError: pass
        self.thumb_worker = ThumbnailWorker(paths_to_load, self.searcher); self.thumb_worker.thumbnail_loaded.connect(self.update_thumbnail)
        self.thumb_worker.finished.connect(lambda: self.statusBar().showMessage(self._get_idle_status_message() + warn_suffix))
        self.thumb_worker.finished.connect(self.thumb_worker.deleteLater); self.thumb_worker.start()

    def update_thumbnail(self, path: str, data: bytes, w: int, h: int, bpl: int) -> None:
        try:
            item = self._path_to_item.get(path)
            if item is None: return
            try: _ = item.data(Qt.ItemDataRole.UserRole)
            except RuntimeError: return
            q_img = QImage(data, w, h, bpl, QImage.Format.Format_RGB888).copy()
            item.setIcon(QIcon(QPixmap.fromImage(q_img)))
        except RuntimeError: pass

    def show_path_in_status(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if not current: self.statusBar().showMessage("Ready"); return
        selected_count = len(self.list_widget.selectedItems())
        if selected_count > 1: self.statusBar().showMessage(f"{selected_count} items selected")
        else:
            path = current.data(Qt.ItemDataRole.UserRole)
            err = self.searcher._check_file_access(path)
            if err: self.statusBar().showMessage(f"{path}  |  {err}")
            else: self.statusBar().showMessage(f"{path}  |  Size: {self._format_file_size(path)}")

    def open_image(self, item: QListWidgetItem) -> None: self.open_image_viewer(item.data(Qt.ItemDataRole.UserRole))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Photo Search")
    parser.add_argument("--index", type=str, help="Directory to index"); parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Clear index"); parser.add_argument("--top", type=int, default=5, help="Results count")
    parser.add_argument("--find-garbage", action="store_true", help="Find low quality images")
    parser.add_argument("--find-dark", action="store_true", help="Find dark images")
    parser.add_argument("--find-blurry", action="store_true", help="Find blurry images")
    parser.add_argument("--find-overexposed", action="store_true", help="Find overexposed images")
    parser.add_argument("--floor", type=float, help="Absolute cosine-similarity floor for search")
    parser.add_argument("--floor-ratio", type=float, dest="floor_ratio", help="Relevance floor as a fraction of the best hit")
    parser.add_argument("--dedupe-cos", type=float, dest="dedupe_cos", help="Near-duplicate suppression cutoff for search")
    parser.add_argument("--purge-junk", action="store_true", help="Remove non-photo directory entries from index")
    args, _ = parser.parse_known_args()
    cache_path = Path.home() / ".cache" / "photofind"
    find_kinds = [k for k, flag in (("dark", args.find_dark), ("blurry", args.find_blurry),
                                    ("overexposed", args.find_overexposed)) if flag]
    find_garbage = args.find_garbage or bool(find_kinds)

    if args.index or args.search or find_garbage or args.purge_junk:
        # CLI Mode
        searcher = PhotoSearch()
        if args.reindex:
            for p in [cache_path / "photo_index.json", cache_path / "photo_embeddings.pt", cache_path / "photo_metadata.json"]:
                if p.exists(): p.unlink()
        searcher.load_index()
        overrides = {k: v for k, v in (("floor_min", args.floor), ("floor_ratio", args.floor_ratio),
                                       ("dedupe_cos", args.dedupe_cos)) if v is not None}
        if overrides: searcher.save_settings(search=overrides)
        if args.index: searcher.index_photos(args.index)
        elif args.purge_junk:
            removed = searcher.purge_fragment_entries()
            print(f"Removed {removed} entries matching {INDEX_EXCLUDE_FRAGMENTS}. New index size: {len(searcher.image_paths)}")
        elif find_garbage:
            kinds = find_kinds or ["all"]
            for kind in kinds:
                hits = searcher.get_problem_photos(kind, 0, 50)
                print(f"\nFound {len(hits)} {kind} images:")
                for h in hits: print(f"[{kind} {h['score']:.2f}] {h['file']}")
        elif args.search:
            hits = searcher.search(args.search, top_k=args.top); print(f"\nTop {args.top} results for '{args.search}':")
            for h in hits: print(f"[Score: {h['score']:.3f}] {h['file']}")
    else:
        # GUI Mode
        app = QApplication(sys.argv)
        # Silence benign KDE/KIO "No node found for item that was just removed"
        # spam from the native (KDE) folder chooser.
        try: QLoggingCategory.setFilterRules("kf.kio.widgets.kdirmodel=false")
        except Exception: pass
        if args.reindex:
            for p in [cache_path / "photo_index.json", cache_path / "photo_embeddings.pt", cache_path / "photo_metadata.json"]:
                if p.exists(): p.unlink()
        # Show splash screen while loading
        splash = None
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photofind.png")
            splash_pixmap = QPixmap(icon_path) if os.path.exists(icon_path) else QPixmap(512, 512)
            if splash_pixmap.isNull():
                splash_pixmap = QPixmap(512, 512)
            splash_pixmap = splash_pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            splash = QSplashScreen(splash_pixmap)
            splash.show()
            splash.showMessage("PhotoFind - Starting up\nLoading AI Model...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
            for _ in range(10):
                app.processEvents()
        except Exception as e:
            print(f"Splash error: {e}")
            splash = None

        window_ref = []
        def _create_main():
            try:
                window_ref.append(PhotoOrganizerWindow())
                if splash is not None and window_ref:
                    try:
                        splash.finish(window_ref[0])
                    except Exception:
                        pass
                if window_ref:
                    window_ref[0].show()
            except Exception as e:
                print(f"Error creating window: {e}")
                if splash is not None:
                    try:
                        splash.finish(None)
                    except Exception:
                        pass
                raise

        QTimer.singleShot(50, _create_main)
        sys.exit(app.exec())
