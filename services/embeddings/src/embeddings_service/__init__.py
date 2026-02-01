"""
Embeddings Service - DINOv2 embedding extraction.

A microservice that extracts L2-normalized visual embeddings from artwork images
using Meta's DINOv2 foundation model.

API Endpoints:
    GET /health - Service health check
    GET /info - Service configuration and metadata
    POST /embed - Extract embedding from base64-encoded image
"""

__version__ = "0.1.0"
