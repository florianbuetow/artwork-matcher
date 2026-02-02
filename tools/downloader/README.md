# Rijksmuseum Data Downloader

Downloads collection metadata and images from the Rijksmuseum APIs for the artwork-matcher project.

## Quick Start

```bash
# From the tools/ directory:

# 1. Initialize the environment
just init

# 2. Download some objects (default: diverse mode from multiple types)
just download

# 3. Check the output (from project root)
ls ../data/downloads/images/
ls ../data/downloads/metadata/
```

## Usage

```bash
# From the tools/ directory:

# Default: diverse download (10 objects from each configured type)
just download

# Download specific type only
just download --type painting --limit 10

# Force re-download (reset state, existing image files still skipped)
just download --force

# Custom download directory
just download --download-dir /path/to/data --limit 100

# Show API info
just download-info

# Test download (5 objects)
just download-test
```

## How It Works

The downloader uses the Rijksmuseum's Linked Art APIs (no API key required):

1. **Search API** - Finds objects with available images
2. **Object Resolver** - Gets detailed metadata in Linked Art format
3. **IIIF Images** - Downloads max-resolution images

For each object, the downloader:
1. Searches for objects with `imageAvailable=true`
2. Fetches object metadata and follows the chain: Object → VisualItem → DigitalObject
3. Checks if the image is downloadable (some are restricted)
4. Downloads the image at full resolution via IIIF
5. Saves metadata as JSON

Objects without downloadable images are skipped entirely (no partial data).

## Output Structure

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

## Features

- **Resume capability**: Interrupted downloads continue where they left off
- **Progress tracking**: Rich terminal output with progress bars
- **Rate limiting**: Respects API limits (configurable delay between requests)
- **No API key required**: Uses the public Linked Art APIs
- **Full resolution images**: Downloads at maximum available quality

## APIs Used

| API | URL | Description |
|-----|-----|-------------|
| Search | data.rijksmuseum.nl/search/collection | Find objects with images |
| Object Resolver | data.rijksmuseum.nl/{id} | Get Linked Art metadata |
| IIIF Image | iiif.micr.io/{id}/full/max/0/default.jpg | Download images |

## Licensing

All Rijksmuseum data is available under **Creative Commons Zero (CC0)** - public domain dedication. You can freely use, modify, and distribute the data and images, including for commercial purposes, without permission.

**Attribution** (appreciated but not required): "Rijksmuseum Amsterdam"

## References

- [Rijksmuseum Data Services](https://data.rijksmuseum.nl/)
- [API Documentation](https://data.rijksmuseum.nl/docs/)
- [Data Policy](https://data.rijksmuseum.nl/policy/)
