# tov

Rewrite text in a given **t**one **o**f **v**oice using Google Gemini, from the command line,
and tune that tone across models in a local web UI.

## Install

```bash
uv tool install tov
```

or `pipx install tov`, or `pip install tov`. You'll need a Gemini API key in `GEMINI_API_KEY`.

To try it without installing:

```bash
uvx tov "Plain, friendly British English" "Please be advised that the office is closed."
```

### As a single-file script

`tov.py` is self-contained: its dependencies are declared inline
([PEP 723](https://peps.python.org/pep-0723/)), so [uv](https://docs.astral.sh/uv/)
installs them automatically when you run it directly.

```bash
curl -fsSL https://raw.githubusercontent.com/tomdyson/tov/main/tov.py -o ~/.local/bin/tov
chmod +x ~/.local/bin/tov
```

## Usage

```
tov TONE INPUT [MODEL]
tov INPUT [MODEL]              # with TOV_TONE set
echo "text" | tov TONE [MODEL] # INPUT omitted or "-" reads stdin
tov --ui [--port N] [--no-open]
tov --version
```

- `TONE` and `INPUT` can each be literal text or a path to a file.
- `MODEL` defaults to `gemini-3.8-flash` (or `TOV_MODEL` if set), with thinking set to
  `low` (or `TOV_THINKING`).
- The rewrite streams to stdout with nothing else added; errors go to stderr.

### Examples

```bash
tov "Warm, plain British English. No jargon." "Per our previous correspondence, your application has been actioned."

pbpaste | tov ~/tone-guide.md | pbcopy

export TOV_TONE="Formal incident update for enterprise customers"
tov "server's down again lol" gemini-3.5-flash
```

A model name is recognised by its shape: a single lowercase hyphenated token such as
`gemini-3.5-flash`. That lets `tov` work out what you meant:

- When input is piped in, `tov TONE MODEL` reads the input from stdin.
- When `TOV_TONE` is set, `tov INPUT MODEL` uses the tone from the environment, while
  `tov TONE INPUT` (second argument not model-shaped) overrides it.

If your input text happens to look like a model name, pass it explicitly with
`tov TONE - MODEL` and pipe it in.

## Tuning a tone in the web UI

```bash
TOV_TONE=~/tone-guide.md tov --ui
```

This starts a small web server on your machine (port 8686, or a free one if that's
taken) and opens it in your browser. Use it to iterate on a tone and check how
different models handle it:

- **Compare models side by side.** Pick any number of Gemini models; each rewrite
  streams into its own card at the same time.
- **Control thinking.** Each model has its own thinking level, and you can add the
  same model twice to compare, say, `gemini-4-neon` at low against high. Thinking is
  usually most of the time and cost of a rewrite, so this is where the trade-offs are.
- **Timing and cost for every run.** Time to first token, total time, input, output
  and thinking tokens, and the cost of the run, plus a running total for the session.
  Prices come from a built-in table (USD per 1M tokens); click a price to set your own,
  for models without a published price or when prices change.
- **See what changed.** Switch a card between the plain output, a word diff against
  the input, and a diff against that model's previous run, so you can see exactly what
  a tone edit did. Each card also shows the share of words reworded.
- **Keep your tone.** If `TOV_TONE` points to a file, **Save** writes your edits back
  to it. **Copy export** copies an `export TOV_TONE='…'` command, and **History** lists
  every tone you've run in this browser.

The server listens only on `127.0.0.1` and rejects requests from other websites.
Stop it with Ctrl-C.

## Environment

| Variable         | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `GEMINI_API_KEY` | Required. Your Gemini API key.                                 |
| `TOV_TONE`       | Default tone guidance (text or path).                          |
| `TOV_MODEL`      | Default model. Defaults to `gemini-3.8-flash`.                 |
| `TOV_TIMEOUT`    | Seconds to wait without hearing from Gemini before giving up. Defaults to 60. |
| `TOV_THINKING`   | Thinking level: `default`, `minimal`, `low`, `medium`, `high` or `xhigh`. Defaults to `low`. |

The timeout is for a stalled connection: a long rewrite that keeps streaming is never
cut off.

Models support different thinking levels (for example, `gemini-4-neon` has no
`minimal`). If a model rejects the level you asked for, `tov` uses the nearest one it
accepts and says so: on stderr from the command line, and on the card in the web UI.
`default` leaves the choice to the model.

The defaults, `gemini-3.8-flash` at `low`, are a good balance of quality, speed and
cost for rewriting. (`gemini-3.8-flash` doesn't accept `minimal`.)

## Development

```bash
uv build                 # builds dist/tov-*.whl and .tar.gz
uv publish               # uploads to PyPI (needs a PyPI token)
```

Bump `__version__` in `tov.py` before each release.

`evals/markdown_eval.py` checks that rewrites keep the input's format: no Markdown
added to plain text, and none lost from Markdown input. Rerun it after changing the
system prompt:

```bash
uv run evals/markdown_eval.py            # 5 runs per input with the default model
uv run evals/markdown_eval.py 20 gemini-3.8-flash low todo,md_link
```

## Licence

MIT — see [LICENSE](LICENSE).
