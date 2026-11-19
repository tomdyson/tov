---
name: tov
description: Rewrite or draft text in the user's tone of voice with the tov command-line tool, which uses Gemini. Use whenever the user asks for text "with tov", "using tov", "through tov", "tov it" or "in my tone of voice", for example "output the message using tov" or "draft me an email, with tov, to accept the request". Needs tov installed (uv tool install tov) and GEMINI_API_KEY set.
allowed-tools: Bash(tov *)
---

# tov

`tov` rewrites text so it matches a tone-of-voice guide, using a Gemini model.
The user's usual tone is in the `TOV_TONE` environment variable, either as text
or as a path to a file. Your job is to get the content right; tov's job is the
voice.

## Steps

1. **Write the content first.** Work out what the text needs to say.
   - For "draft me an email, with tov, to accept the request", write the complete
     email yourself: the right facts, names, dates, greeting, sign-off and any
     subject line. Don't try to imitate the user's voice; tov will do that.
   - For "output the message using tov", the message is usually the draft you just
     wrote or text the user gave you. If it's unclear which, ask.
   - Write it in the format the user will use. tov keeps the input's format: plain
     text stays plain text, and Markdown stays Markdown. Use plain text for emails
     and chat messages.

2. **Run it through tov.** Pass the text on stdin in a quoted heredoc, so the shell
   leaves `$`, backticks and `!` alone:

   ```bash
   tov <<'TOV_EOF'
   Hi Sam,

   Thanks for the invitation to speak on 12 March. I'd be glad to accept.
   ...
   TOV_EOF
   ```

   If the text has a line that is exactly `TOV_EOF`, choose another delimiter.
   Rewrites usually take a few seconds; allow up to two minutes.

3. **Reply with tov's output, exactly as it came back.** The user can't see command
   output, so put the rewritten text in your reply. Don't edit it afterwards, or you
   will undo the tone. Check it against your draft, though: if tov dropped or changed
   a fact, name, number, date or link, say so in a short note after the text and
   offer to fix it.

Don't send, post or save the text anywhere unless the user asks for that as well.

## Options

**A different tone for one request**, such as "with tov, in a formal tone": pass
the tone as the first argument and `-` to read the text from stdin. This replaces
`TOV_TONE` for that command.

```bash
tov 'Formal, warm British English. No contractions.' - <<'TOV_EOF'
...
TOV_EOF
```

**The usual tone plus an extra instruction**, such as "with tov, but shorter":

```bash
tov "$( [ -f "$TOV_TONE" ] && cat "$TOV_TONE" || printf '%s' "$TOV_TONE" )
For this text: keep it under 50 words." - <<'TOV_EOF'
...
TOV_EOF
```

**A particular model or thinking level**, such as "with tov using gemini-4-neon":
set them for that command only.

```bash
TOV_MODEL=gemini-4-neon TOV_THINKING=high tov <<'TOV_EOF'
...
TOV_EOF
```

Thinking levels are `default`, `minimal`, `low`, `medium`, `high` and `xhigh`. If a
model doesn't support the level, tov uses the nearest one and prints a note on
stderr; mention it only if the user asked for that level.

**Comparing versions**: run tov once per variant; the commands are independent. To
tune the tone itself across models, suggest the user runs `tov --ui`, which opens a
web page for comparing models, thinking levels, timing and cost.

## Problems

- `command not found: tov`: tov isn't installed. Tell the user to run
  `uv tool install tov` (or `pipx install tov`), then try again.
- `no tone given`: `TOV_TONE` isn't set in this session. Ask the user what tone to
  use and pass it as the first argument. To set a default, they can save the tone in
  a file such as `~/.config/tov/tone.md` and add
  `export TOV_TONE="$HOME/.config/tov/tone.md"` to their shell profile (for example
  `~/.zshrc`). Claude only sees the new variable in a new session.
- `GEMINI_API_KEY is not set`: the user needs a Gemini API key from
  https://aistudio.google.com/apikey, exported in their shell profile.
- `no response from ... after 60s`: try once more; if it fails again, tell the user.
- `Gemini API error`: tell the user what it says. If the model isn't available,
  try again without `TOV_MODEL` to use the default.

tov sends the text to Google's Gemini API. Never include passwords, keys or other
secrets in the text.
