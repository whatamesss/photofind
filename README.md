# photofind

**photofind** is a local, AI-powered photo search tool written in Python. It provides a graphical user interface (Qt6) to help you organize, search, and clean up massive photo collections using natural language.

The goal of photofind is to tame a massive, disorganized photo collection taken over the course of many years (or decades!). It provides a Google Photos-style visual search interface without relying on external cloud services or proprietary algorithms.

## Features
*   **Semantic Visual Search:** Search your photos using natural language (e.g., "sunset at the beach," "cat sleeping on a sofa," "blue car with white stripe").
*   **100% Local & Private:** No cloud uploads. All AI processing happens on your own CPU or GPU hardware.
*   **Self-Optimizing Engine:** Automatically calibrates to your specific GPU VRAM and adjusts batch sizes dynamically to prevent crashes and maximize speed.
*   **Hardware Acceleration:** Automatically detects and utilizes NVIDIA CUDA-capable GPUs.
*   **Duplicate Management:** If `jdupes` is installed, a dedicated "Duplicate Manager" helps you find and remove redundant files.
*   **Quality Control:** Includes a "Find Bad Photos" feature to detect blurry, dark, or low-quality images.

## What is CLIP?

CLIP stands for **C**ontrastive **L**anguage-**I**mage **P**re-training. It is a neural network model created by OpenAI (released in 2021) that fundamentally changed how computers "see" and understand images.

Unlike traditional image recognition that requires specific training for every object, CLIP was trained on hundreds of millions of image-text pairs from the internet. This allows it to understand the connection between visual concepts and written descriptions. Because of this, photofind can search for concepts it has never seen before in your specific library—it understands the *idea* of a "sunset" or a "birthday party" just as well as it understands a specific face.

## Installation (Gentoo Linux)

I've included an .ebuild so that you may add my repo to a local portage overlay.

Don't forget to run 'ebuild photofind-9999.ebuild manifest' in the overlay dir.
and echo "media-gfx/photofind **" > /etc/portage/pacakage.accept_keywords/photofind

You must explicitly enable the `torch` backend for the Hugging Face libraries.

echo "sci-ml/transformers torch" >> /etc/portage/package.use/photofind
echo "sci-ml/huggingface_hub torch" >> /etc/portage/package.use/photofind


If you have an NVIDIA graphics card with CUDA capability, enable the `cuda` flag to the backend.

echo "sci-libs/pytorch cuda" >> /etc/portage/package.use/photofind
echo "sci-ml/caffe2 cuda" >> /etc/portage/package.use/photofind
echo "dev-cpp/cutlass -clang-cuda" >> /etc/portage/package.use/photofind (i needed this one, ymmv)

**Note:** This package requires `xdg-utils` to open images in your default external viewer.

## Usage

Upon launching the application for the first time:
1.  **Click Index Folder:** In the toolbar, click "Index Folder" and select the base directory containing your photos.
2.  **Scan & Index:** The tool will recursively scan and index your photos. This process uses your GPU if available and is optimized for cards with lower VRAM (like the 4GB GTX 1050Ti).
    *   The engine automatically calibrates to your hardware. If an Out-Of-Memory error occurs, it will automatically reduce the batch size and recover.
    *   You can press "Stop Indexing" at any time to cancel the operation.
3.  **Search:** Type a description into the search bar to find matching images instantly.
4.  **Manage:** Use the context menu (right-click) to open files in your file manager, open them in your default image viewer, move to trash, or find duplicates.
