# tov

Rewrite text in a given **t**one **o**f **v**oice using Google Gemini, from the command line.

`tov` is a single self-contained Python script. Its dependencies are declared inline
([PEP 723](https://peps.python.org/pep-0723/)), so [uv](https://docs.astral.sh/uv/)
installs them automatically on first run.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/tomdyson/tov/main/tov -o ~/.local/bin/tov
chmod +x ~/.local/bin/tov
```

Requires `uv` and a Gemini API key in `GEMINI_API_KEY`.

## Usage

```
tov TONE INPUT [MODEL]
tov INPUT [MODEL]              # with TOV_TONE set
echo "text" | tov TONE         # INPUT omitted or "-" reads stdin
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

When `TOV_TONE` is set and you pass two arguments, the second is treated as a model
name if it contains no spaces (`tov INPUT MODEL`); otherwise the two are read as
`TONE INPUT` and the tone you passed overrides the environment.

## Environment

| Variable         | Purpose                               |
| ---------------- | ------------------------------------- |
| `GEMINI_API_KEY` | Required. Your Gemini API key.        |
| `TOV_TONE`       | Default tone guidance (text or path). |
| `TOV_MODEL`      | Default model.                        |
