"""Persistent singleton store that survives app.py hot-reloads.

dev.py keeps ONE Python process alive and calls importlib.reload(app) on save.
A reload re-runs app.py's top-level code — which would otherwise re-load the
~90 MB models, the 10k-chunk index, and re-extract every PDF (~13-20 s). Anything
stashed here lives in THIS module, which dev.py never reloads, so a reload just
re-fetches the already-built objects instantly. A normal `python app.py` run uses
it too (builds each once); behavior is identical either way.
"""
import threading

_STORE: dict = {}
_LOCK = threading.Lock()


def cache(key, factory):
    """Return _STORE[key], building it via factory() exactly once (thread-safe)."""
    if key in _STORE:
        return _STORE[key]
    with _LOCK:
        if key not in _STORE:
            _STORE[key] = factory()
        return _STORE[key]
