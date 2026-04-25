"""Nexus Trader packaged runtime.

This package is an isolated production runtime that does not depend on
`src.*` imports at runtime. The package is designed to be imported safely:

    python -c "import nexus_packaged"
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "v27.1"

