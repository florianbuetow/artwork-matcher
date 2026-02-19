"""
Service entry point.
"""

from __future__ import annotations

import sys

import uvicorn

from storage_service.app import create_app
from storage_service.config import ConfigurationError, get_settings


def main() -> int:
    """Run the service with uvicorn."""
    try:
        settings = get_settings()
    except ConfigurationError as e:
        print(f"FATAL: Configuration error\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL: Unexpected error during configuration\n{e}", file=sys.stderr)
        return 1

    app = create_app()

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="warning",
        access_log=False,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
