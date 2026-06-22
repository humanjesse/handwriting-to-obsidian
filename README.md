# Handwritten Notes → Obsidian (local VLM transcriber)

A small Python script that turns photos/scans of handwritten pages into
searchable Obsidian notes, using a vision model running **entirely locally**.

## Why?

I am always looking for use cases of small LLM's, and transcribing my personal handwritten notes has worked out better than expected.
Thus I am sharing the script.

I love writing on a physical medium, but I also want to be able to search notes,
have a cool graph, backups, etc. so this is my solution.

## Requirements

- **Python 3**
- **ImageMagick** (used for cropping/cleanup)
- **Obsidian**
- **llama.cpp** — specifically the `llama-server` binary. It must be **recent
  enough to support Qwen 3.6's `qwen35moe` architecture** (older builds, and
  Ollama, fail with `unknown model architecture`). Any backend works
  (Vulkan / ROCm / CUDA / Metal / CPU). Grab a build from the
  [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
- **A Qwen 3.6 GGUF + its vision projector** — I used the
  [unsloth Q6_K](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF). You need
  **two** files: the model GGUF **and** the matching `mmproj` (without the mmproj
  the model can't see images).

## Setup

**1. llama.cpp** — put `llama-server` on your `PATH`, or point the script at it
with the `LLAMA_SERVER` env var. (If you use a prebuilt release tarball, the
script finds the bundled libraries automatically.)

**2. The model** — download both files into `~/models/qwen3.6/` (or anywhere, and
set `MODEL` / `MMPROJ`):

```bash
mkdir -p ~/models/qwen3.6 && cd ~/models/qwen3.6
base="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main"
curl -L -O "$base/Qwen3.6-35B-A3B-UD-Q6_K.gguf"   # ~27 GB
curl -L -O "$base/mmproj-F16.gguf"                # vision projector
```

## Project layout

```
Notes/                  (a folder in your Obsidian vault)
├── transcribe.py       the script
├── images/             put scans / photos of your notes here
├── transcriptions/     created by the script — the output notes
└── _preprocessed/      created by the script — the cropped images it sends
```

## How to use

`cd` into the Notes folder and run `python3 transcribe.py`.

- The script will make additional sub-folders (`_preprocessed` and
  `transcriptions`).
- For each new image it will crop the image to page size (it finds the page's
  corners and flattens it), transcribe the contents, add tags, title the note
  with the date written on the page, and give it a short descriptive name.
- It's **idempotent** — already-processed images are skipped, so just keep
  dropping new photos into `images/` and re-running.

Handy options:

```bash
python3 transcribe.py --force          # re-do everything
KEEP_SERVER=1 python3 transcribe.py    # leave the model loaded between runs
PREP=0 python3 transcribe.py           # skip image pre-processing
CROP=0 python3 transcribe.py           # enhance but don't crop
```

## Configuration (environment variables)

All optional — sensible defaults are used if unset.

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL` | `~/models/qwen3.6/Qwen3.6-35B-A3B-UD-Q6_K.gguf` | model GGUF |
| `MMPROJ` | `~/models/qwen3.6/mmproj-F16.gguf` | vision projector |
| `LLAMA_SERVER` | (PATH) | path to the `llama-server` binary |
| `PORT` | `8080` | port the server listens on |
| `NGL` | `99` | layers to offload to GPU |
| `CTX` | `16384` | context size |

## A note on accuracy

Overall transcriptions have been remarkably accurate although accuracy will
depend on legibility of handwriting. The model uses surrounding context to infer
hard-to-read words, and marks anything it truly can't read with `[?]`. Dates are
the field most worth a quick double-check; pages with no readable date are saved
with an `undated` prefix so they're easy to find and fix by hand.

> Built and tested on an AMD Strix Halo (Radeon 8060S) using the Vulkan backend,
> but nothing in the script is tied to that hardware.
