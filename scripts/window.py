#!/usr/bin/python3
"""Native Lemur window via GTK + WebKit (system Python)."""
from __future__ import annotations

import sys

import gi

gi.require_version("Gtk", "3.0")
try:
    gi.require_version("WebKit2", "4.1")
except ValueError:
    gi.require_version("WebKit2", "4.0")
from gi.repository import Gtk, WebKit2  # noqa: E402


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9000"

    win = Gtk.Window(title="Lemur")
    win.set_default_size(1800, 1300)
    win.set_icon_name("utilities-terminal")
    win.connect("destroy", Gtk.main_quit)

    context = WebKit2.WebContext.get_default()
    context.set_cache_model(WebKit2.CacheModel.DOCUMENT_VIEWER)

    view = WebKit2.WebView()
    view.load_uri(url)

    # No GTK scroll chrome — the web UI owns viewport layout.
    win.add(view)
    win.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
