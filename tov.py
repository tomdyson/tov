#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0",
#     "httpx",
# ]
# ///
"""tov — rewrite text in a given tone of voice using Gemini.

Usage:
    tov TONE INPUT [MODEL]
    tov INPUT [MODEL]              # with TOV_TONE set
    echo "text" | tov TONE [MODEL] # INPUT omitted or "-" reads stdin

    tov --ui [--port N] [--no-open]  # web UI for tuning a tone across models
    tov --version

TONE and INPUT may each be literal text or a path to a file.

Environment:
    GEMINI_API_KEY  required
    TOV_TONE        default tone (text or a path to a file)
    TOV_MODEL       default model (gemini-3.8-flash)
    TOV_TIMEOUT     seconds to wait for Gemini before giving up (60)
    TOV_THINKING    thinking level (low): default, minimal, low, medium, high, xhigh
"""

__version__ = "0.2.0"

import datetime as dt
import json
import os
import re
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, NoReturn

import httpx
from google import genai
from google.genai import errors, types

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING = "low"
DEFAULT_TIMEOUT = 60.0
DEFAULT_PORT = 8686

SYSTEM_PROMPT = """You are an editor who rewrites text to match a tone-of-voice guide.

Rules:
- Rewrite the user's text so it follows the tone guidance below.
- Preserve the meaning, facts, names, numbers, links and overall structure \
(paragraphs, lists, headings) unless the guidance says otherwise.
- Do not add new claims or remove substantive content.
- Keep the input's format. Don't add Markdown or other formatting the input \
doesn't have: no **bold**, *italics*, # headings or `code`, and don't turn \
prose into bulleted or numbered lists. If the input uses Markdown, keep it.
- Output ONLY the rewritten text: no preamble, no explanation, no quotation marks \
around it, no code fences.

<tone_guidance>
{tone}
</tone_guidance>"""

# USD per 1M tokens: Gemini API paid tier, standard (non-batch), text input and
# output including thinking. Checked against Google's pricing page 2026-11-19.
# Each model has (effective_from, input, output) entries; the latest entry
# that has taken effect applies. Long-prompt (>200k token) tiers are ignored.
PRICES_CHECKED = "2026-11-19"
PRICES: dict[str, list[tuple[str | None, float, float]]] = {
    "gemini-4-neon": [(None, 3.00, 15.00)],
    "gemini-4-argon": [(None, 2.00, 10.00), ("2026-12-01", 4.00, 20.00)],
    "gemini-4-carbon": [(None, 2.00, 10.00), ("2026-12-01", 4.00, 20.00)],
    "gemini-4-hydrogen": [(None, 1.50, 7.50)],
    "gemini-3.8-flash": [(None, 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3.6-flash": [(None, 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3.5-flash-lite": [(None, 0.30, 2.50)],
    "gemini-3.1-pro-preview": [(None, 2.00, 12.00)],
    "gemini-3.1-flash-lite": [(None, 0.25, 1.50)],
    "gemini-3-flash-preview": [(None, 0.50, 3.00)],
    "gemini-2.5-pro": [(None, 1.25, 10.00)],
    "gemini-2.5-flash": [(None, 0.30, 2.50)],
    "gemini-2.5-flash-lite": [(None, 0.10, 0.40)],
}

# Models the API lists that aren't useful for rewriting text.
NON_TEXT_MODELS = re.compile(
    r"tts|image|banana|live|embedding|audio|robotics|computer-use|transcribe|"
    r"lyria|aqa|deep-research|antigravity|decision|customtools|imagen|veo|omni"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def usage(msg: str) -> NoReturn:
    print(f"tov: {msg}\n", file=sys.stderr)
    print(__doc__.strip(), file=sys.stderr)
    sys.exit(2)


def fail(msg: str) -> NoReturn:
    print(f"tov: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve(value: str) -> str:
    """Return the file's contents if value is a path to an existing file, else value."""
    if value == "-":
        return sys.stdin.read()
    path = tone_file(value)
    return path.read_text() if path else value


def tone_file(value: str) -> Path | None:
    if value and len(value) < 1024 and "\n" not in value:
        p = Path(value).expanduser()
        if p.is_file():
            return p
    return None


def looks_like_model(value: str) -> bool:
    """Model IDs are single lowercase hyphenated tokens, e.g. gemini-4-neon."""
    return bool(re.fullmatch(r"(models/)?[a-z][a-z0-9.]*(-[a-z0-9.]+)+", value))


def price_for(model: str, today: dt.date | None = None) -> dict | None:
    entries = PRICES.get(model.removeprefix("models/"))
    if not entries:
        return None
    today_s = (today or dt.date.today()).isoformat()
    current = None
    for since, pin, pout in entries:
        if since is None or since <= today_s:
            current = {"in": pin, "out": pout}
    return current


def get_timeout() -> float:
    raw = os.environ.get("TOV_TIMEOUT", "")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        usage(f"TOV_TIMEOUT must be a number of seconds, not {raw!r}")
    if value <= 0:
        usage("TOV_TIMEOUT must be greater than 0")
    return value


class StallTimeoutTransport(httpx.HTTPTransport):
    """Give up when Gemini sends nothing for `seconds`.

    google-genai passes timeout=None on every request (overriding any client
    default), and its own `timeout` option also becomes a server-side deadline
    on the whole generation. Setting the timeout here instead makes it an
    inactivity limit: a stalled connection fails, a long streaming rewrite
    doesn't.
    """

    def __init__(self, seconds: float):
        super().__init__()
        self._timeout = httpx.Timeout(seconds, connect=min(seconds, 15.0)).as_dict()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.extensions["timeout"] = self._timeout
        return super().handle_request(request)


def make_client(timeout: float) -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fail("GEMINI_API_KEY is not set")
    # Use GEMINI_API_KEY only; stop the SDK preferring GOOGLE_API_KEY if set.
    os.environ.pop("GOOGLE_API_KEY", None)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            client_args={"transport": StallTimeoutTransport(timeout)}
        ),
    )


def is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, TimeoutError)) or (
        "timeout" in type(exc).__name__.lower()
    )


THINKING_LEVELS = ("default", "minimal", "low", "medium", "high", "xhigh")

# Models support different subsets of thinking levels. When a model rejects
# the requested level, try the nearest alternatives in this order, then fall
# back to the model's own default.
THINKING_FALLBACKS = {
    "minimal": ["minimal", "low", "medium", "high"],
    "low": ["low", "minimal", "medium", "high"],
    "medium": ["medium", "low", "high", "minimal"],
    "high": ["high", "medium", "xhigh", "low"],
    "xhigh": ["xhigh", "high", "medium"],
}
_thinking_learned: dict[tuple[str, str], str] = {}  # (model, requested) -> used
_thinking_lock = threading.Lock()


def get_thinking(value: str | None = None) -> str:
    level = (value if value is not None else os.environ.get("TOV_THINKING", "")).strip().lower()
    level = level or DEFAULT_THINKING
    if level not in THINKING_LEVELS:
        usage(f"TOV_THINKING must be one of {', '.join(THINKING_LEVELS)}, not {level!r}")
    return level


def _is_thinking_rejection(e: errors.APIError) -> bool:
    return e.code == 400 and "thinking" in str(e.message or e).lower()


def stream_rewrite(
    client: genai.Client, model: str, tone: str, text: str, thinking: str = "default"
) -> Iterator[tuple[str, object]]:
    """Yield ("text", chunk) pairs, then ("done", {"usage": ..., "thinking": used})."""
    with _thinking_lock:
        learned = _thinking_learned.get((model, thinking))
    if thinking == "default":
        candidates = ["default"]
    elif learned:
        candidates = [learned]
    else:
        candidates = [*THINKING_FALLBACKS[thinking], "default"]

    for i, level in enumerate(candidates):
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(tone=tone),
            thinking_config=(
                None if level == "default"
                else types.ThinkingConfig(thinking_level=level.upper())
            ),
        )
        usage_metadata, started = None, False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=text, config=config
            ):
                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
                if chunk.text:
                    started = True
                    yield "text", chunk.text
        except errors.APIError as e:
            # A rejected thinking level fails before any output: try the next.
            if not started and _is_thinking_rejection(e) and i + 1 < len(candidates):
                continue
            raise
        if thinking != "default":
            with _thinking_lock:
                _thinking_learned[(model, thinking)] = level
        yield "done", {"usage": usage_metadata, "thinking": level}
        return


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> tuple[str, str, str]:
    env_tone = os.environ.get("TOV_TONE")
    args = list(argv)

    # With TOV_TONE set, positionals are INPUT [MODEL] — unless there are two
    # and the second doesn't look like a model name, in which case the user
    # passed TONE INPUT explicitly and that overrides the environment.
    if env_tone is not None and (
        len(args) <= 1 or (len(args) == 2 and looks_like_model(args[1]))
    ):
        tone = env_tone
    elif args:
        tone = args.pop(0)
    else:
        usage("no tone given (pass it as the first argument or set TOV_TONE)")

    if len(args) > 2:
        usage("too many arguments")

    # `echo text | tov TONE MODEL`: a lone model-like argument with piped
    # stdin is the model, and the input comes from stdin.
    if len(args) == 1 and looks_like_model(args[0]) and not sys.stdin.isatty():
        args.insert(0, "-")

    input_arg = args[0] if args else "-"
    model = args[1] if len(args) > 1 else os.environ.get("TOV_MODEL", DEFAULT_MODEL)

    if input_arg == "-" and sys.stdin.isatty():
        usage("no input given (pass it as an argument or pipe it on stdin)")

    tone_text = resolve(tone).strip()
    input_text = resolve(input_arg).strip()
    if not tone_text:
        usage("tone is empty")
    if not input_text:
        usage("input is empty")
    return tone_text, input_text, model


def main() -> None:
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip())
        return
    if argv[:1] in (["-V"], ["--version"]):
        print(f"tov {__version__}")
        return
    if "--ui" in argv:
        serve_ui(argv)
        return

    tone, text, model = parse_args(argv)
    timeout = get_timeout()
    thinking = get_thinking()
    client = make_client(timeout)
    ended_with_newline = True
    note = ""

    def bail(msg: str) -> NoReturn:
        if not ended_with_newline:
            print()
        fail(msg)

    try:
        for kind, value in stream_rewrite(client, model, tone, text, thinking):
            if kind == "text":
                sys.stdout.write(value)
                sys.stdout.flush()
                ended_with_newline = value.endswith("\n")
            elif value["thinking"] != thinking:
                used = value["thinking"]
                note = (f"tov: {model} doesn't support thinking level {thinking}; "
                        f"used {'its default' if used == 'default' else used}")
        if not ended_with_newline:
            sys.stdout.write("\n")
        if note:
            sys.stdout.flush()
            print(note, file=sys.stderr)
    except errors.APIError as e:
        bail(f"Gemini API error ({model}): {e}")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        if is_timeout(e):
            bail(
                f"no response from {model} after {timeout:g}s "
                "(set TOV_TIMEOUT to wait longer)"
            )
        if isinstance(e, httpx.HTTPError):
            bail(f"could not reach Gemini: {e}")
        raise


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, client: genai.Client, tone: str, tone_path: Path | None,
                 default_model: str, timeout: float, thinking: str):
        super().__init__(addr, UIHandler)
        self.thinking = thinking
        self.client = client
        self.tone = tone
        self.tone_path = tone_path
        self.default_model = default_model
        self.timeout = timeout
        self.token = secrets.token_urlsafe(24)
        port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}
        self._models: list[dict] | None = None
        self._models_lock = threading.Lock()

    def models(self) -> list[dict]:
        with self._models_lock:
            if self._models is None:
                self._models = self._load_models()
            return self._models

    def _load_models(self) -> list[dict]:
        found: dict[str, str] = {}
        try:
            for m in self.client.models.list():
                mid = (m.name or "").removeprefix("models/")
                actions = m.supported_actions or []
                if mid and "generateContent" in actions and not NON_TEXT_MODELS.search(mid):
                    found[mid] = m.display_name or mid
        except Exception as e:  # fall back to the models we know prices for
            print(f"tov: could not list models ({e}); using built-in list", file=sys.stderr)
        for mid in [*PRICES, self.default_model]:
            found.setdefault(mid, mid)
        return [
            {"id": mid, "name": found[mid], "price": price_for(mid)}
            for mid in sorted(found, key=model_sort_key)
        ]


def model_sort_key(mid: str):
    # Newest Gemini generations first, then everything else alphabetically.
    m = re.match(r"gemini-(\d+(?:\.\d+)?)", mid)
    return (0, -float(m.group(1)), mid) if m else (1, 0.0, mid)


class UIHandler(BaseHTTPRequestHandler):
    server: UIServer
    server_version = f"tov/{__version__}"

    def log_message(self, format, *args):  # keep the terminal quiet
        pass

    # -- guards -------------------------------------------------------------

    def _allowed(self, needs_token: bool) -> bool:
        # Host check defeats DNS rebinding; the per-run token (only readable
        # by pages served from this origin) and Origin check defeat CSRF.
        if self.headers.get("Host") not in self.server.allowed_hosts:
            self._json(403, {"error": "forbidden host"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.allowed_origins:
            self._json(403, {"error": "forbidden origin"})
            return False
        if needs_token and not secrets.compare_digest(
            self.headers.get("X-TOV-Token", ""), self.server.token
        ):
            self._json(403, {"error": "missing or wrong token"})
            return False
        return True

    def _json(self, code: int, obj: object) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 5_000_000:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return None

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            if not self._allowed(needs_token=False):
                return
            page = (
                UI_HTML.replace("__TOV_TOKEN__", self.server.token)
                .replace("__TOV_VERSION__", __version__)
                .encode()
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
        elif path == "/api/config":
            if not self._allowed(needs_token=True):
                return
            s = self.server
            tone_path = str(s.tone_path) if s.tone_path else None
            if tone_path and tone_path.startswith(str(Path.home())):
                tone_path = "~" + tone_path[len(str(Path.home())):]
            self._json(200, {
                "version": __version__,
                "tone": s.tone,
                "tone_path": tone_path,
                "default_model": s.default_model,
                "timeout": s.timeout,
                "thinking": s.thinking,
                "thinking_levels": THINKING_LEVELS,
                "models": s.models(),
                "prices_checked": PRICES_CHECKED,
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._allowed(needs_token=True):
            return
        path = self.path.split("?", 1)[0]
        data = self._body()
        if data is None:
            return
        if path == "/api/rewrite":
            self._rewrite(data)
        elif path == "/api/tone":
            self._save_tone(data)
        else:
            self._json(404, {"error": "not found"})

    def _save_tone(self, data: dict) -> None:
        tone = data.get("tone")
        if not isinstance(tone, str) or not tone.strip():
            return self._json(400, {"error": "tone is empty"})
        if not self.server.tone_path:
            return self._json(400, {"error": "TOV_TONE is not a file, so there is nowhere to save"})
        self.server.tone_path.write_text(tone.rstrip("\n") + "\n")
        self.server.tone = tone
        self._json(200, {"ok": True})

    def _rewrite(self, data: dict) -> None:
        tone, text, model = data.get("tone"), data.get("input"), data.get("model")
        thinking = data.get("thinking") or "default"
        if not all(isinstance(v, str) and v.strip() for v in (tone, text, model)):
            return self._json(400, {"error": "tone, input and model are all required"})
        if thinking not in THINKING_LEVELS:
            return self._json(400, {"error": f"unknown thinking level {thinking!r}"})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(obj: dict) -> None:
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()

        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000)  # noqa: E731
        first = None
        try:
            for kind, value in stream_rewrite(
                self.server.client, model, tone.strip(), text.strip(), thinking
            ):
                if kind == "text":
                    if first is None:
                        first = ms()
                    emit({"t": "chunk", "text": value})
                else:
                    u = value["usage"]
                    emit({
                        "t": "done",
                        "ms_first": first,
                        "ms_total": ms(),
                        "thinking": {"requested": thinking, "used": value["thinking"]},
                        "usage": {
                            "prompt": getattr(u, "prompt_token_count", None) or 0,
                            "output": getattr(u, "candidates_token_count", None) or 0,
                            "thinking": getattr(u, "thoughts_token_count", None) or 0,
                        },
                        "price": price_for(model),
                    })
        except (BrokenPipeError, ConnectionResetError):
            return  # the browser stopped the request
        except errors.APIError as e:
            self._emit_error(emit, f"Gemini API error: {e.message or e}")
        except Exception as e:
            if is_timeout(e):
                msg = (f"No response after {self.server.timeout:g}s "
                       "(set TOV_TIMEOUT to wait longer)")
            elif isinstance(e, httpx.HTTPError):
                msg = f"Could not reach Gemini: {e}"
            else:
                msg = f"{type(e).__name__}: {e}"
            self._emit_error(emit, msg)

    @staticmethod
    def _emit_error(emit, message: str) -> None:
        try:
            emit({"t": "error", "message": message})
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve_ui(argv: list[str]) -> None:
    port, open_browser = DEFAULT_PORT, True
    args = [a for a in argv if a != "--ui"]
    while args:
        a = args.pop(0)
        if a == "--no-open":
            open_browser = False
        elif a == "--port" and args:
            try:
                port = int(args.pop(0))
            except ValueError:
                usage("--port needs a number")
        elif a.startswith("--port="):
            try:
                port = int(a.split("=", 1)[1])
            except ValueError:
                usage("--port needs a number")
        else:
            usage(f"unexpected argument for --ui: {a}")

    timeout = get_timeout()
    client = make_client(timeout)
    env_tone = os.environ.get("TOV_TONE", "")
    tone_path = tone_file(env_tone)
    tone = (tone_path.read_text() if tone_path else env_tone).strip()
    default_model = os.environ.get("TOV_MODEL", DEFAULT_MODEL)
    thinking = get_thinking()

    def make(p: int) -> UIServer:
        return UIServer(("127.0.0.1", p), client, tone, tone_path, default_model, timeout, thinking)

    try:
        httpd = make(port)
    except OSError:
        httpd = make(0)  # port busy: let the OS pick a free one
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"tov {__version__} UI running at {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ntov: stopped", file=sys.stderr)
    finally:
        httpd.server_close()


UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tov</title>
<style>
:root {
  --bg: #f6f5f1; --panel: #ffffff; --text: #1d1c19; --muted: #6d6b63; --faint: #a3a198;
  --border: #e4e2da; --field: #fbfaf7; --chip: #efede6;
  --accent: #4338ca; --accent-hover: #3730a3; --accent-text: #ffffff; --accent-soft: #e0e7ff;
  --ins-bg: #d9f5e1; --ins-text: #14532d; --del-bg: #fde2e2; --del-text: #8f1d1d;
  --error: #b42318; --ok: #15803d;
  --shadow: 0 1px 2px rgb(0 0 0 / .04), 0 2px 8px rgb(0 0 0 / .04);
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #131312; --panel: #1c1c1a; --text: #ecebe5; --muted: #a09e95; --faint: #6c6a63;
    --border: #2f2e2a; --field: #171716; --chip: #282724;
    --accent: #8b93f8; --accent-hover: #a5abfa; --accent-text: #12121a; --accent-soft: #2a2c52;
    --ins-bg: #173a24; --ins-text: #b7f0c8; --del-bg: #45201f; --del-text: #f7c3c0;
    --error: #f97066; --ok: #4ade80;
    --shadow: none;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased; }
button, select, input, textarea { font: inherit; color: inherit; }

header.top { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 16px; padding: 10px 20px; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }
.brand { font-weight: 700; font-size: 16px; letter-spacing: -.01em; }
.brand small { font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 6px; }
.session { margin-left: auto; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.session b { color: var(--text); font-weight: 600; }
@media (max-width: 600px) { .brand small { display: none; } }

main { display: grid; grid-template-columns: minmax(340px, 440px) minmax(0, 1fr); gap: 20px; padding: 20px; align-items: start; }
@media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); padding: 16px; } header.top { padding: 10px 16px; } }
.left { display: flex; flex-direction: column; gap: 16px; }

.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; box-shadow: var(--shadow); }
.panel-head { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
.panel-head label { font-weight: 600; }
.src { color: var(--muted); font-size: 12px; font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px; }
.spacer { flex: 1; }
textarea { width: 100%; resize: vertical; border: 1px solid var(--border); border-radius: 8px; background: var(--field); padding: 10px 11px; line-height: 1.55; outline: none; }
textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
#tone { min-height: 200px; }
#input { min-height: 150px; }
.meta { display: flex; gap: 10px; color: var(--faint); font-size: 12px; margin-top: 6px; font-variant-numeric: tabular-nums; }
.meta .edited { color: var(--accent); }

.btn { border: 1px solid var(--border); background: var(--panel); border-radius: 7px; padding: 3px 9px; font-size: 12.5px; cursor: pointer; color: var(--muted); }
.btn:hover { color: var(--text); border-color: var(--faint); }
.btn:disabled { opacity: .45; cursor: default; }
.btn.primary { background: var(--accent); color: var(--accent-text); border-color: var(--accent); font-weight: 600; padding: 8px 16px; font-size: 14px; }
.btn.primary:hover { background: var(--accent-hover); }
.btn.on { background: var(--accent-soft); color: var(--text); border-color: transparent; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.chip { display: inline-flex; align-items: center; gap: 6px; background: var(--chip); border-radius: 999px; padding: 3px 4px 3px 10px; font-family: var(--mono); font-size: 12px; }
.chip .price { color: var(--muted); cursor: pointer; border-bottom: 1px dotted var(--faint); }
.chip .x { border: 0; background: transparent; cursor: pointer; color: var(--muted); width: 20px; height: 20px; border-radius: 50%; line-height: 1; }
.chip .x:hover { background: var(--border); color: var(--text); }
.chip select.think { width: auto; padding: 1px 6px; border-radius: 999px; font-size: 11.5px; background: var(--panel); cursor: pointer; }
.tag { font-size: 11.5px; color: var(--muted); background: var(--chip); border-radius: 999px; padding: 1px 8px; }
select { width: 100%; border: 1px solid var(--border); border-radius: 8px; background: var(--field); padding: 7px 9px; font-size: 13px; }
.hint { color: var(--faint); font-size: 12px; margin-top: 6px; }

.run-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
kbd { font-family: var(--mono); font-size: 11px; opacity: .75; margin-left: 6px; }

.outputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; align-items: start; }
.empty { border: 1px dashed var(--border); border-radius: 10px; padding: 48px 24px; text-align: center; color: var(--muted); }
.empty h2 { margin: 0 0 6px; font-size: 16px; color: var(--text); }
.empty p { margin: 4px auto; max-width: 460px; }

.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; box-shadow: var(--shadow); display: flex; flex-direction: column; min-width: 0; }
.card > header { display: flex; align-items: center; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
.card .name { font-family: var(--mono); font-size: 13px; font-weight: 600; }
.status { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; display: inline-flex; align-items: center; gap: 6px; }
.status.err { color: var(--error); }
.status.ok { color: var(--ok); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .25; } }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 7px; overflow: hidden; }
.seg button { border: 0; background: transparent; padding: 3px 8px; font-size: 12px; color: var(--muted); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button.on { background: var(--accent-soft); color: var(--text); }
.seg button:disabled { opacity: .4; cursor: default; }
.out { padding: 12px 14px; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; min-height: 80px; font-size: 14.5px; }
.out .placeholder { color: var(--faint); }
.out .errmsg { color: var(--error); display: block; margin-top: 8px; font-size: 13px; }
ins { background: var(--ins-bg); color: var(--ins-text); text-decoration: none; border-radius: 3px; }
del { background: var(--del-bg); color: var(--del-text); border-radius: 3px; }
.stats { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 9px 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.stats b { color: var(--text); font-weight: 600; }
.stats .cost b { color: var(--accent); }
.linkish { background: none; border: 0; padding: 0; color: var(--accent); cursor: pointer; text-decoration: underline; font-size: 12px; }

.popover { position: absolute; z-index: 20; width: min(420px, calc(100vw - 32px)); max-height: 360px; overflow: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 8px 30px rgb(0 0 0 / .18); padding: 6px; }
.popover button { display: block; width: 100%; text-align: left; border: 0; background: transparent; padding: 8px 10px; border-radius: 7px; cursor: pointer; }
.popover button:hover { background: var(--chip); }
.popover .when { color: var(--faint); font-size: 11.5px; display: block; }
.popover .preview { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.popover .none { padding: 10px; color: var(--muted); }

#toast { position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%) translateY(20px); background: var(--text); color: var(--bg); padding: 8px 14px; border-radius: 8px; font-size: 13px; opacity: 0; pointer-events: none; transition: .2s; z-index: 30; }
#toast.show { opacity: 1; transform: translateX(-50%); }
</style>
</head>
<body>
<header class="top">
  <div class="brand">tov<small>tone-of-voice tuner · v__TOV_VERSION__</small></div>
  <div class="session" id="session">No runs yet</div>
</header>

<main>
  <section class="left">
    <div class="panel">
      <div class="panel-head">
        <label for="tone">Tone</label>
        <span class="src" id="tone-src"></span>
        <span class="spacer"></span>
        <button class="btn" id="history-btn" title="Tones you've run this browser">History</button>
        <button class="btn" id="export-btn" title="Copy a shell command that sets TOV_TONE">Copy export</button>
        <button class="btn" id="save-btn" hidden>Save</button>
      </div>
      <textarea id="tone" spellcheck="true" placeholder="Describe the voice: who it's for, how it should sound, words to use and avoid…"></textarea>
      <div class="meta"><span id="tone-count"></span><span class="edited" id="tone-edited" hidden>● edited</span></div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <label for="input">Input</label>
        <span class="spacer"></span>
        <button class="btn" id="clear-input">Clear</button>
      </div>
      <textarea id="input" placeholder="Paste the text to rewrite…"></textarea>
      <div class="meta"><span id="input-count"></span></div>
    </div>

    <div class="panel">
      <div class="panel-head"><label for="add-model">Models</label></div>
      <div class="chips" id="chips"></div>
      <select id="add-model"></select>
      <div class="hint">Prices are USD per 1M tokens, input / output, from Google's price list on <span id="price-date"></span>. Click a price to change it.</div>
    </div>

    <div class="run-row">
      <button class="btn primary" id="run">Rewrite<kbd id="kbd">⌘↵</kbd></button>
      <button class="btn" id="stop" hidden>Stop</button>
      <span class="hint" id="run-hint"></span>
    </div>
  </section>

  <section>
    <div class="outputs" id="outputs">
      <div class="empty" id="empty">
        <h2>Tune a tone against real models</h2>
        <p>Write or paste a tone and some input, pick one or more models, then press Rewrite.</p>
        <p>Each model's output streams into its own card with timing, token counts and cost. Edit the tone and run again to see what changed.</p>
      </div>
    </div>
  </section>
</main>
<div id="toast" role="status" aria-live="polite"></div>

<script>
"use strict";
const TOKEN = "__TOV_TOKEN__";
const $ = (s, el = document) => el.querySelector(s);
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem("tov." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("tov." + k, JSON.stringify(v)); } catch {} },
};
const api = (path, opts = {}) => fetch(path, { ...opts, headers: { "X-TOV-Token": TOKEN, "Content-Type": "application/json" } });

let config = null;
let models = [];
let selected = [];        // [{ id, thinking }]
let unavailable = store.get("unavailable", {});
const keyOf = e => `${e.id}|${e.thinking}`;
const saveSelected = () => store.set("selection", selected.map(({ id, thinking }) => ({ id, thinking })));
let originalTone = "";
let priceOverrides = store.get("prices", {});
const runs = [];           // completed runs, for the session total
const cards = new Map();   // model id -> card state

// ---------- formatting ----------
const n = x => x.toLocaleString();
const secs = ms => ms == null ? "—" : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + "s";
function usd(x) {
  if (x == null) return "—";
  if (x === 0) return "$0";
  if (x < 0.00001) return "<$0.00001";
  return "$" + (x < 1 ? x.toPrecision(3) : x.toFixed(2));
}
function ago(t) {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago";
  return new Date(t).toLocaleDateString();
}
const words = s => (s.match(/[\p{L}\p{N}][\p{L}\p{N}'’_-]*/gu) || []);
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("show"), 2200);
}

// ---------- prices ----------
function priceOf(id) { return priceOverrides[id] || (models.find(m => m.id === id) || {}).price || null; }
const priceLabel = p => p ? `$${p.in} / $${p.out}` : "price ?";
function costOf(usage, price) {
  if (!usage || !price) return null;
  return (usage.prompt * price.in + (usage.output + usage.thinking) * price.out) / 1e6;
}
function editPrice(id) {
  const cur = priceOf(id);
  const ans = prompt(`USD per 1M tokens for ${id}, as "input, output" (e.g. 0.5, 3). Leave empty to reset.`, cur ? `${cur.in}, ${cur.out}` : "");
  if (ans === null) return;
  if (!ans.trim()) { delete priceOverrides[id]; }
  else {
    const [pin, pout] = ans.split(/[,\s/]+/).map(Number);
    if (!(pin >= 0 && pout >= 0)) { toast("Couldn't read that price"); return; }
    priceOverrides[id] = { in: pin, out: pout };
  }
  store.set("prices", priceOverrides);
  renderModels(); cards.forEach(renderStats); renderSession();
}

// ---------- tone ----------
const toneEl = $("#tone"), inputEl = $("#input");
function renderToneMeta() {
  const t = toneEl.value;
  $("#tone-count").textContent = `${n(t.length)} chars · ~${n(Math.ceil(t.length / 4))} tokens`;
  $("#tone-edited").hidden = t.trim() === originalTone.trim();
}
function renderInputMeta() {
  const t = inputEl.value;
  $("#input-count").textContent = `${n(words(t).length)} words · ${n(t.length)} chars`;
}
function pushToneHistory(tone) {
  const h = store.get("toneHistory", []).filter(e => e.tone !== tone);
  h.unshift({ tone, at: Date.now() });
  store.set("toneHistory", h.slice(0, 30));
}
function showHistory(anchor) {
  closePopover();
  const pop = document.createElement("div");
  pop.className = "popover"; pop.id = "popover";
  const items = [];
  if (originalTone) items.push({ label: config.tone_path ? `Saved in ${config.tone_path.split("/").pop()}` : "Original TOV_TONE", tone: originalTone });
  for (const e of store.get("toneHistory", [])) items.push({ label: ago(e.at), tone: e.tone });
  if (!items.length) pop.innerHTML = `<div class="none">Tones you run will appear here.</div>`;
  for (const it of items) {
    const b = document.createElement("button");
    b.innerHTML = `<span class="when">${esc(it.label)}</span><span class="preview">${esc(it.tone.replace(/\s+/g, " ").slice(0, 140))}</span>`;
    b.onclick = () => { toneEl.value = it.tone; renderToneMeta(); closePopover(); toneEl.focus(); };
    pop.append(b);
  }
  document.body.append(pop);
  const r = anchor.getBoundingClientRect();
  pop.style.top = (r.bottom + window.scrollY + 6) + "px";
  pop.style.left = Math.max(16, Math.min(r.left + window.scrollX, window.innerWidth - pop.offsetWidth - 16)) + "px";
  setTimeout(() => document.addEventListener("click", closePopover, { once: true }), 0);
}
function closePopover() { $("#popover")?.remove(); }

$("#history-btn").onclick = e => { e.stopPropagation(); $("#popover") ? closePopover() : showHistory(e.currentTarget); };
$("#export-btn").onclick = async () => {
  const cmd = `export TOV_TONE='${toneEl.value.trim().replace(/'/g, `'\\''`)}'`;
  try { await navigator.clipboard.writeText(cmd); toast("Copied an export command for TOV_TONE"); }
  catch { toast("Couldn't copy to the clipboard"); }
};
$("#save-btn").onclick = async () => {
  const tone = toneEl.value.trim();
  const r = await api("/api/tone", { method: "POST", body: JSON.stringify({ tone }) });
  const j = await r.json().catch(() => ({}));
  if (r.ok) { originalTone = tone; renderToneMeta(); toast(`Saved to ${config.tone_path}`); }
  else toast(j.error || "Save failed");
};
toneEl.addEventListener("input", renderToneMeta);
inputEl.addEventListener("input", () => { renderInputMeta(); store.set("input", inputEl.value); });
$("#clear-input").onclick = () => { inputEl.value = ""; renderInputMeta(); store.set("input", ""); inputEl.focus(); };

// ---------- models ----------
function renderModels() {
  const chips = $("#chips"); chips.innerHTML = "";
  for (const e of selected) {
    const c = document.createElement("span"); c.className = "chip";
    const opts = config.thinking_levels.map(l => `<option value="${l}"${l === e.thinking ? " selected" : ""}>thinking: ${l}</option>`).join("");
    c.innerHTML = `${esc(e.id)} <select class="think" title="Thinking level" aria-label="Thinking level for ${esc(e.id)}">${opts}</select> <span class="price" title="Edit price">${priceLabel(priceOf(e.id))}</span><button class="x" title="Remove" aria-label="Remove ${esc(e.id)}">×</button>`;
    $(".think", c).onchange = ev => {
      const t = ev.target.value;
      if (selected.some(o => o !== e && o.id === e.id && o.thinking === t)) { toast(`Already comparing ${e.id} at ${t}`); ev.target.value = e.thinking; return; }
      e.thinking = t; saveSelected();
    };
    $(".price", c).onclick = () => editPrice(e.id);
    $(".x", c).onclick = () => { selected = selected.filter(o => o !== e); saveSelected(); renderModels(); };
    chips.append(c);
  }
  const sel = $("#add-model");
  sel.innerHTML = `<option value="">${selected.length ? "+ Compare with another model or thinking level…" : "+ Add a model…"}</option>` +
    models.map(m => `<option value="${esc(m.id)}">${esc(m.id)} — ${unavailable[m.id] ? "unavailable to this key" : priceLabel(priceOf(m.id))}</option>`).join("");
  $("#run-hint").textContent = selected.length > 1 ? `${selected.length} runs in parallel` : "";
}
$("#add-model").onchange = ev => {
  const id = ev.target.value;
  if (!id) return;
  const used = new Set(selected.filter(o => o.id === id).map(o => o.thinking));
  const thinking = [config.thinking, ...config.thinking_levels].find(l => !used.has(l));
  if (thinking) { selected.push({ id, thinking }); saveSelected(); }
  else toast(`Already comparing ${id} at every thinking level`);
  renderModels();
};

// ---------- diff ----------
const diffTokens = s => s.match(/\s+|[\p{L}\p{N}_'’]+|[^\s\p{L}\p{N}_]/gu) || [];
function lcsTable(A, B) {
  const W = B.length + 1, T = new Uint32Array((A.length + 1) * W);
  for (let i = A.length - 1; i >= 0; i--)
    for (let j = B.length - 1; j >= 0; j--)
      T[i * W + j] = A[i] === B[j] ? T[(i + 1) * W + j + 1] + 1 : Math.max(T[(i + 1) * W + j], T[i * W + j + 1]);
  return T;
}
function diffHTML(a, b) {
  const A = diffTokens(a), B = diffTokens(b);
  if (A.length * B.length > 6e6) return null;
  const T = lcsTable(A, B), W = B.length + 1, parts = [];
  const push = (cls, s) => { const l = parts[parts.length - 1]; if (l && l.cls === cls) l.s += s; else parts.push({ cls, s }); };
  let i = 0, j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { push("", A[i]); i++; j++; }
    else if (T[(i + 1) * W + j] >= T[i * W + j + 1]) push("del", A[i++]);
    else push("ins", B[j++]);
  }
  while (i < A.length) push("del", A[i++]);
  while (j < B.length) push("ins", B[j++]);
  return groupChanges(parts).map(p => p.cls ? `<${p.cls}>${esc(p.s)}</${p.cls}>` : esc(p.s)).join("");
}
// Merge changes separated only by whitespace into one deletion + one insertion,
// so a reworded phrase reads as "old phrase → new phrase", not word soup.
function groupChanges(parts) {
  const out = [];
  let d = "", a = "", ws = "", open = false;
  const flush = () => {
    if (d.trim()) out.push({ cls: "del", s: d });
    if (a.trim()) out.push({ cls: "ins", s: a });
    d = a = ""; open = false;
  };
  for (const p of parts) {
    if (p.cls) {
      if (open) { d += ws; a += ws; } else if (ws) out.push({ cls: "", s: ws });
      ws = ""; open = true;
      if (p.cls === "del") d += p.s; else a += p.s;
    } else if (open && /^\s+$/.test(p.s)) ws += p.s;
    else { flush(); if (ws) { out.push({ cls: "", s: ws }); ws = ""; } out.push(p); }
  }
  flush();
  if (ws) out.push({ cls: "", s: ws });
  return out;
}
function changedShare(a, b) {
  const A = words(a.toLowerCase()), B = words(b.toLowerCase());
  if (!A.length || !B.length || A.length * B.length > 6e6) return null;
  const same = lcsTable(A, B)[0];
  return 1 - same / Math.max(A.length, B.length);
}

// ---------- cards ----------
function ensureCard(entry) {
  const id = entry.id, key = keyOf(entry);
  let c = cards.get(key);
  if (c) return c;
  const el = document.createElement("article"); el.className = "card";
  el.innerHTML = `
    <header>
      <span class="name">${esc(id)}</span>${entry.thinking === "default" ? "" : `<span class="tag">thinking ${esc(entry.thinking)}</span>`}
      <span class="status"></span>
      <span class="spacer"></span>
      <span class="seg" role="group" aria-label="View">
        <button data-mode="text" class="on" title="Plain output">Output</button>
        <button data-mode="input" title="Changes compared with the input">vs input</button>
        <button data-mode="prev" title="Changes compared with this model's previous run">vs last run</button>
      </span>
      <button class="btn" data-act="copy">Copy</button>
      <button class="btn" data-act="rerun" title="Run this model again with the current tone and input">Rerun</button>
    </header>
    <div class="out"></div>
    <div class="stats"></div>`;
  c = { id, key, thinking: entry.thinking, thinkingUsed: null, el, text: "", prev: null, input: "", mode: "text", usage: null, timing: null, error: null, ctrl: null, startedAt: 0, firstAt: 0, tick: null };
  el.querySelectorAll(".seg button").forEach(b => b.onclick = () => { c.mode = b.dataset.mode; renderCard(c); });
  $('[data-act="copy"]', el).onclick = async () => {
    try { await navigator.clipboard.writeText(c.text); toast("Copied output"); } catch { toast("Couldn't copy"); }
  };
  $('[data-act="rerun"]', el).onclick = () => startRun([entry]);
  cards.set(key, c);
  $("#empty")?.remove();
  $("#outputs").append(el);
  return c;
}
function renderStatus(c) {
  const st = $(".status", c.el);
  const now = performance.now();
  if (c.ctrl) {
    const label = c.firstAt ? "writing" : "waiting";
    st.className = "status"; st.innerHTML = `<span class="dot"></span>${label} ${secs(now - c.startedAt)}`;
  } else if (c.error) {
    st.className = "status err"; st.textContent = c.error === "Stopped" ? "stopped" : "error";
  } else if (c.timing) {
    st.className = "status ok"; st.textContent = "done";
  } else st.textContent = "";
}
function renderOut(c) {
  const out = $(".out", c.el);
  let html;
  const base = c.mode === "input" ? c.input : c.mode === "prev" ? c.prev : null;
  if (!c.text && c.ctrl) html = `<span class="placeholder">Waiting for ${esc(c.id)}…</span>`;
  else if (base != null && c.text) html = diffHTML(base, c.text) ?? esc(c.text) + `<span class="errmsg">Too long to diff.</span>`;
  else html = esc(c.text);
  if (c.error && c.error !== "Stopped") html += `<span class="errmsg">${esc(c.error)}</span>`;
  else if (c.error === "Stopped" && c.text) html += `<span class="errmsg">Stopped before the end.</span>`;
  out.innerHTML = html;
  c.el.querySelectorAll(".seg button").forEach(b => {
    b.classList.toggle("on", b.dataset.mode === c.mode);
    b.disabled = b.dataset.mode === "prev" && c.prev == null;
  });
}
function renderStats(c) {
  const s = $(".stats", c.el);
  if (!c.timing) { s.innerHTML = c.ctrl ? `<span>timeout ${config.timeout}s</span>` : ""; s.hidden = !c.ctrl; return; }
  s.hidden = false;
  const u = c.usage, price = priceOf(c.id), cost = costOf(u, price);
  const share = changedShare(c.input, c.text);
  const inW = words(c.input).length, outW = words(c.text).length;
  s.innerHTML = [
    `<span title="Time to first token">first <b>${secs(c.timing.first)}</b></span>`,
    `<span>total <b>${secs(c.timing.total)}</b></span>`,
    `<span title="Prompt tokens (tone + instructions + input)">in <b>${n(u.prompt)}</b></span>`,
    `<span>out <b>${n(u.output)}</b>${u.thinking ? ` + ${n(u.thinking)} thinking` : ""}</span>`,
    c.thinkingUsed ? `<span title="Thinking level sent to the model">level <b>${esc(c.thinkingUsed.used)}</b>${c.thinkingUsed.used !== c.thinkingUsed.requested ? ` (${esc(c.thinkingUsed.requested)} unsupported)` : ""}</span>` : "",
    `<span class="cost">${cost == null ? `<button class="linkish">set price</button>` : `cost <b>${usd(cost)}</b>`}</span>`,
    `<span title="Words in the input → words in the output">words ${n(inW)} → ${n(outW)}</span>`,
    share == null ? "" : `<span title="Share of words changed from the input">${Math.round(share * 100)}% reworded</span>`,
  ].join("");
  const link = $(".linkish", s); if (link) link.onclick = () => editPrice(c.id);
}
function renderCard(c) { renderStatus(c); renderOut(c); renderStats(c); }
function renderSession() {
  if (!runs.length) { $("#session").textContent = "No runs yet"; return; }
  let total = 0, unknown = 0;
  for (const r of runs) { const x = costOf(r.usage, priceOf(r.id)); if (x == null) unknown++; else total += x; }
  $("#session").innerHTML = `<b>${runs.length}</b> run${runs.length > 1 ? "s" : ""} · <b>${usd(total)}</b> this session` + (unknown ? ` · ${unknown} unpriced` : "");
}
function renderRunning() {
  const busy = [...cards.values()].some(c => c.ctrl);
  $("#stop").hidden = !busy;
}

// ---------- running ----------
const UNAVAILABLE = /not found for API version|no longer available|only supports Interactions/i;
function noteAvailability(id, message) {
  const was = !!unavailable[id], now = !!(message && UNAVAILABLE.test(message));
  if (was === now) return;
  if (now) unavailable[id] = true; else delete unavailable[id];
  store.set("unavailable", unavailable); renderModels();
}
async function runOne(entry, tone, input) {
  const id = entry.id, c = ensureCard(entry);
  c.ctrl?.abort();
  if (c.text && !c.error) c.prev = c.text;
  Object.assign(c, { text: "", input, usage: null, timing: null, thinkingUsed: null, error: null, ctrl: new AbortController(), startedAt: performance.now(), firstAt: 0 });
  c.tick = setInterval(() => renderStatus(c), 100);
  renderCard(c); renderRunning();
  const ctrl = c.ctrl;
  let pending = false;
  const paint = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; renderOut(c); }); } };
  try {
    const res = await api("/api/rewrite", { method: "POST", body: JSON.stringify({ tone, input, model: id, thinking: entry.thinking }), signal: ctrl.signal });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.t === "chunk") { if (!c.firstAt) c.firstAt = performance.now(); c.text += ev.text; paint(); }
        else if (ev.t === "done") {
          c.usage = ev.usage; c.timing = { first: ev.ms_first, total: ev.ms_total }; c.thinkingUsed = ev.thinking;
          runs.push({ id, usage: ev.usage }); renderSession(); noteAvailability(id, null);
        } else if (ev.t === "error") c.error = ev.message;
      }
    }
    if (!c.timing && !c.error) c.error = "The connection closed before the response finished.";
  } catch (e) {
    c.error = e.name === "AbortError" ? "Stopped" : (e.message || String(e));
  } finally {
    if (c.error) noteAvailability(id, c.error);
    if (c.ctrl === ctrl) { c.ctrl = null; clearInterval(c.tick); renderCard(c); renderRunning(); }
  }
}
function startRun(entries) {
  const tone = toneEl.value.trim(), input = inputEl.value.trim();
  if (!tone) { toast("Add a tone first"); toneEl.focus(); return; }
  if (!input) { toast("Add some input text first"); inputEl.focus(); return; }
  if (!entries.length) { toast("Pick at least one model"); $("#add-model").focus(); return; }
  pushToneHistory(tone);
  for (const e of entries) runOne(e, tone, input);
}
function runAll() {
  // Drop cards for models no longer selected, and keep cards in selection order.
  const keys = new Set(selected.map(keyOf));
  for (const [key, c] of cards) if (!keys.has(key)) { c.ctrl?.abort(); c.el.remove(); cards.delete(key); }
  startRun(selected);
  for (const e of selected) { const c = cards.get(keyOf(e)); if (c) $("#outputs").append(c.el); }
}
$("#run").onclick = runAll;
$("#stop").onclick = () => cards.forEach(c => c.ctrl?.abort());
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); runAll(); }
  if (e.key === "Escape") closePopover();
});
if (!/Mac|iPhone|iPad/.test(navigator.platform)) $("#kbd").textContent = "Ctrl+↵";

// ---------- start ----------
(async function init() {
  try {
    const r = await api("/api/config");
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    config = await r.json();
  } catch (e) {
    $("#empty").innerHTML = `<h2>Can't reach the tov server</h2><p>${esc(e.message)}</p><p>Is <code>tov --ui</code> still running? Restart it and reload this page.</p>`;
    return;
  }
  models = config.models;
  originalTone = config.tone || "";
  toneEl.value = originalTone;
  $("#tone-src").textContent = config.tone_path ? config.tone_path : originalTone ? "from TOV_TONE" : "";
  $("#tone-src").title = config.tone_path ? `TOV_TONE file: ${config.tone_path}` : "";
  $("#save-btn").hidden = !config.tone_path;
  $("#save-btn").title = config.tone_path ? `Write this tone back to ${config.tone_path}` : "";
  $("#price-date").textContent = config.prices_checked;
  inputEl.value = store.get("input", "");
  const known = new Set(models.map(m => m.id));
  const levels = new Set(config.thinking_levels);
  const saved = store.get("selection", null) || (store.get("models", null) || []).map(id => ({ id, thinking: config.thinking }));
  const seen = new Set();
  selected = saved.filter(e => e && known.has(e.id) && levels.has(e.thinking) && !seen.has(keyOf(e)) && seen.add(keyOf(e)));
  if (!selected.length) selected = [{ id: config.default_model, thinking: config.thinking }];
  renderModels(); renderToneMeta(); renderInputMeta();
  (originalTone ? inputEl : toneEl).focus();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
