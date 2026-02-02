# Developer Log: Rijksmuseum Data Downloader Implementation

**Branch:** `feature-data-crawler`

---

## Overview

This branch implements a **data downloader tool** for fetching artwork images and metadata from the Rijksmuseum collection. It uses the new Rijksmuseum Linked Art APIs which require no API key, downloading high-resolution images via IIIF and metadata in JSON-LD format.

---

## Features Implemented

### Download Modes

**Diverse Mode (Default)** - Downloads objects from multiple types for dataset variety
- Iterates through 20 configured object types (painting, sculpture, furniture, etc.)
- Downloads a configurable number of objects per type
- Provides variety in the training/evaluation dataset

**Single Type Mode** - Downloads objects of a specific type only
- Activated with `--type <type>` flag
- Useful for targeted data collection

### Core Capabilities

- **Resume Capability** - Saves downloaded object IDs and pagination tokens between runs
- **Progress Tracking** - Rich terminal output with progress bars
- **Rate Limiting** - Configurable delays between requests to respect API limits
- **Smart Filtering** - Skips objects without downloadable images (no partial data)
- **Atomic State Writes** - Uses temp file + rename to prevent state corruption

---

## Architecture

### Chain Fetching

The Rijksmuseum Linked Art API requires following a chain of resources to get image URLs:

```
Object → VisualItem → DigitalObject → IIIF Image URL
```

Each step requires a separate API call with rate limiting between requests.

### Application Structure

```
tools/downloader/
├── __init__.py
├── config.yaml          # All configuration values
├── download_data.py     # Main implementation
└── README.md            # Usage documentation
```

### Output Structure

```
data/downloads/
├── metadata/              # JSON files with Linked Art metadata
│   ├── RP-P-1906-2550.json
│   └── ...
├── images/                # Max-resolution JPEG images
│   ├── RP-P-1906-2550.jpg
│   └── ...
└── .download_state.json   # Resume state (auto-generated)
```

---

## Design Decisions

### 1. New Linked Art APIs Over Legacy API

Used the new `data.rijksmuseum.nl` APIs instead of the legacy `www.rijksmuseum.nl/api`.

**Why:** No API key required. Returns richer Linked Art (JSON-LD) metadata. Better long-term support from the museum.

### 2. Diverse Mode as Default

Default behavior downloads from multiple object types instead of sequential downloading.

**Why:** Creates a more varied dataset for training and evaluation. Prevents bias toward any single object type.

### 3. Object Number as File ID

Uses the human-readable object number (e.g., `RP-P-1906-2550`) for filenames instead of numeric IDs.

**Why:** More meaningful filenames. Easier to correlate with museum catalog. Stable across API changes.

### 4. Skip Non-Downloadable Objects Entirely

Objects without downloadable images are skipped completely (no metadata saved).

**Why:** Avoids partial data that would cause issues in downstream processing. Keeps dataset consistent.

### 5. All Configuration from config.yaml

All values (rate limits, timeouts, types, batch sizes, output directory) come from `config.yaml` with no defaults in code.

**Why:** Fail-fast at startup. No hidden assumptions. Aligns with project's CLAUDE.md guidelines.

### 6. Atomic State Persistence

State is written to a temp file first, then atomically renamed.

**Why:** Prevents corruption if interrupted during write. Ensures resume capability is reliable.

### 7. Corrupted State Recovery

If state file is corrupted, backs it up and starts fresh instead of crashing.

**Why:** Graceful degradation. User doesn't lose ability to run the tool. Backup preserved for debugging.

### 8. Metadata Saved Only After Image Success

Metadata JSON is only written after the image downloads successfully.

**Why:** Prevents orphaned metadata without corresponding images. Maintains data consistency.

---

## Issues Encountered & Fixes

### Chain Fetching Complexity

**Problem:** Getting an image URL requires 4 API calls (Object → VisualItem → DigitalObject → access_point).

**Fix:** Implemented `resolve_image_url()` method that handles the full chain with proper error handling at each step.

### Downloadability Detection

**Problem:** Not all objects with images have downloadable images (some are restricted).

**Fix:** Check `referred_to_by` field in DigitalObject for "downloadbaar" status before attempting download.

### PR Review: CLAUDE.md Compliance

**Problem:** Initial implementation used default values in dataclass and `.get()` fallbacks.

**Fix:** Removed all defaults. All configuration must be explicit in `config.yaml`. Added `output` section for `download_dir`.

### PR Review: Silent Failures

**Problem:** Config loading, state loading/saving, and metadata saving had no error handling.

**Fix:** Added comprehensive error handling:
- Config: Fail-fast with clear error messages
- State load: Recover from corruption, backup bad file
- State save: Atomic writes, warn on failure
- Metadata: Clean up orphaned images on failure

---

## APIs Used

| API | URL | Description |
|-----|-----|-------------|
| Search | data.rijksmuseum.nl/search/collection | Find objects with images |
| Object Resolver | data.rijksmuseum.nl/{id} | Get Linked Art metadata |
| IIIF Image | iiif.micr.io/{id}/full/max/0/default.jpg | Download max-resolution images |

All APIs are public (no authentication required) and data is CC0 (public domain).

---

## Justfile Commands

### From `tools/` directory

| Command | Description |
|---------|-------------|
| `just download` | Run downloader with custom args |
| `just download-batch` | Download diverse batch (default settings) |
| `just download-test` | Test download (5 objects) |
| `just download-info` | Show configuration and API info |

### From project root

| Command | Description |
|---------|-------------|
| `just download` | Delegates to tools/download |
| `just download-batch` | Delegates to tools/download-batch |
