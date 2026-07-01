"""Hot-reload dev server: edit app.py, save, changes are live — no restart.

The models (~90 MB embedder + reranker), the 10k-chunk index, and the extracted
PDFs load ONCE here and stay in memory (they live in _hot.py, which is never
reloaded). On every save to app.py this process re-runs only app.py's top-level
code — routes + pipeline functions rebind instantly because the heavy objects come
straight from the _hot cache. Reload cost ≈ 0.1 s vs a ~15-20 s cold restart.

Run:  python dev.py     (instead of `python app.py`)
Serves the same app on http://127.0.0.1:5001 — point the frontend at it as usual.

Note: only app.py is hot-reloaded. Editing indexer.py / chunk_legal.py / _hot.py
still needs a manual restart (Ctrl-C then rerun) — those hold the loaded state.
"""
import importlib
import os
import threading
import time

from werkzeug.serving import run_simple

import app as app_module

_APP_PATH = os.path.abspath(app_module.__file__)
_mtime = os.path.getmtime(_APP_PATH)
_lock = threading.Lock()


def _maybe_reload():
    """Reload app.py if it changed on disk since the last request (mtime check)."""
    global _mtime
    m = os.path.getmtime(_APP_PATH)
    if m == _mtime:
        return
    with _lock:
        if m == _mtime:            # another thread already reloaded
            return
        t = time.time()
        try:
            importlib.reload(app_module)
            print(f"♻  reloaded app.py in {time.time() - t:.2f}s")
        except Exception as e:     # keep serving the old app if the edit is broken
            print(f"⚠  reload failed ({e!r}) — keeping previous app.py")
        _mtime = m


def application(environ, start_response):
    _maybe_reload()
    return app_module.app(environ, start_response)   # always the freshest Flask app


if __name__ == "__main__":
    print("Hot-reload dev server on http://127.0.0.1:5001  (edit app.py, just save)")
    # use_reloader=False: WE own reloading (in-process, keeping models warm). The
    # Werkzeug reloader would re-exec the whole interpreter and re-load the models.
    run_simple("127.0.0.1", 5001, application, threaded=True, use_reloader=False)
