"""
Service entry point.

This module provides the main() function for running the service
with production-grade configuration.
"""

from __future__ import annotations

import sys

import uvicorn

from gateway.app import create_app
from gateway.config import ConfigurationError, get_settings


def main() -> int:
    """
    Run the service with uvicorn.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate configuration before starting
        settings = get_settings()
    except ConfigurationError as e:
        # Print to stderr - logging isn't configured yet
        print(f"FATAL: Configuration error\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL: Unexpected error during configuration\n{e}", file=sys.stderr)
        return 1

    # Create application
    app = create_app()

    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        # Production settings
        log_level="warning",  # Uvicorn logs (our JSON logger handles app logs)
        access_log=False,  # Disable uvicorn access log (use middleware if needed)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
