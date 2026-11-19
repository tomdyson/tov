# tov

Rewrite text in a given **t**one **o**f **v**oice using Google Gemini, from the command line.

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
tov --version
```

- `TONE` and `INPUT` can each be literal text or a path to a file.
- `MODEL` defaults to `gemini-4-neon` (or `TOV_MODEL` if set).
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

## Environment

| Variable         | Purpose                               |
| ---------------- | ------------------------------------- |
| `GEMINI_API_KEY` | Required. Your Gemini API key.        |
| `TOV_TONE`       | Default tone guidance (text or path). |
| `TOV_MODEL`      | Default model.                        |

## Development

```bash
uv build                 # builds dist/tov-*.whl and .tar.gz
uv publish               # uploads to PyPI (needs a PyPI token)
```

Bump `__version__` in `tov.py` before each release.

## Licence

MIT — see [LICENSE](LICENSE).
