"""An optional browser playground for running SQL against prism.

The server is a thin HTTP layer over :class:`~prism.engine.Database`: it parses,
optimizes, and executes queries and streams the results (and query plans) to a
single-page UI. It lives behind the ``server`` extra so the core engine keeps
its only runtime dependency as NumPy.
"""

from prism.server.app import create_app, serve

__all__ = ["create_app", "serve"]
