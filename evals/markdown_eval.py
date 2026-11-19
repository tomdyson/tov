#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["google-genai>=1.0", "httpx"]
# ///
"""Does tov add Markdown to plain-text input, or drop it from Markdown input?

Runs each input several times under three setups and counts Markdown
features that appear in the output but not the input (for plain text), or
that the input had but the output lost (for Markdown).

    uv run evals/markdown_eval.py [RUNS] [MODEL] [THINKING] [INPUTS]

Defaults: 5 runs, tov's default model and thinking level, every input.
INPUTS is a comma-separated list of input names, e.g. todo,md_list.
Needs GEMINI_API_KEY.
"""

import concurrent.futures as cf
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("GOOGLE_API_KEY", None)

import tov  # noqa: E402
from google.genai import types  # noqa: E402

# The system prompt before the format-matching rule, for comparison.
OLD_PROMPT = """You are an editor who rewrites text to match a tone-of-voice guide.

Rules:
- Rewrite the user's text so it follows the tone guidance below.
- Preserve the meaning, facts, names, numbers, links and overall structure \
(paragraphs, lists, headings) unless the guidance says otherwise.
- Do not add new claims or remove substantive content.
- Output ONLY the rewritten text: no preamble, no explanation, no quotation marks \
around it, no code fences.

<tone_guidance>
{tone}
</tone_guidance>"""

TONE = (
    'Natural, neutral, concise, British English. "I have" -> "I\'ve". Write like a '
    "scientist - favour preciseness and clarity, reduce emotion. But don't compensate "
    "by increasing formality. Use plain language, no jargon."
)
MARKDOWN_SENTENCE = " Don't output markdown unless the input is markdown."

SETUPS = {
    "old prompt, tone without md sentence": (OLD_PROMPT, TONE),
    "old prompt, tone with md sentence": (OLD_PROMPT, TONE + MARKDOWN_SENTENCE),
    "new prompt, tone without md sentence": (tov.SYSTEM_PROMPT, TONE),
}

# Plain-text inputs chosen to tempt a model into bullets, bold or headings.
PLAIN = {
    "shopping": "For the party we need three things really: loads of crisps, some decent "
                "wine (red AND white!) and those little sausage rolls everyone loves.",
    "steps": "To reset the router first unplug it, wait about thirty seconds, plug it back "
             "in, then wait for the light to go green before you try connecting again.",
    "important": "This is REALLY important. Please make sure you lock the back door every "
                 "single night, even if you think someone else has already done it.",
    "notes": "Meeting notes 12 March\n\nAttendees were Sam, Priya and Tom.\n\nWe agreed "
             "the budget is too tight. Priya will talk to finance. Tom will draft a new "
             "plan by Friday. Next meeting is on the 19th.",
    "email": "Hi Jo,\n\nI have looked at the three quotes. The first is cheapest but they "
             "can't start until June. The second is mid-priced and can start next week. "
             "The third is the most expensive but includes the scaffolding.\n\nI think "
             "the second one is our best bet honestly.\n\nCheers,\nAlex",
    "comparison": "Option A is faster but costs more and needs a specialist. Option B is "
                  "slower, cheaper and anyone on the team could do it. Option C is "
                  "somewhere in the middle on everything.",
    "update": "Quick update on the website: the new homepage is live, the contact form is "
              "fixed, and we still need to sort out the broken links on the about page and "
              "the slow image loading on mobile.",
    "pros_cons": "Pros of moving offices: more space, better transport links, cheaper "
                 "rent. Cons: the move itself will be disruptive and some people will have "
                 "a longer commute.",
    "todo": "Things to do this weekend\nbuy milk\ncall mum about Sunday lunch\nfix the "
            "shed door, it's hanging off again\nbook the MOT before it runs out!!",
    "sections": "Background: the old server is five years old and keeps crashing.\n"
                "Problem: backups are failing about once a week and nobody notices.\n"
                "Proposal: move everything to a managed service and set up alerts.\n"
                "Cost: roughly £40 a month, which I think is a total bargain.",
    "faq": "Q: when is the shop open? A: weekdays 8 till 6 and Saturday mornings.\n"
           "Q: do you deliver? A: yes, anywhere within 5 miles, it's free over £30.\n"
           "Q: can I pay by card? A: of course, we take everything except Amex.",
    "specs": "The new laptop is amazing - 16GB RAM, a 1TB SSD, a gorgeous 14 inch screen "
             "and it only weighs 1.3kg. Battery life is about 12 hours which is incredible.",
    "recipe": "For the sauce you'll need 2 tins of tomatoes, a garlic clove, a handful of "
              "basil, olive oil and salt. Fry the garlic gently, add the tomatoes, simmer "
              "for 20 minutes, then stir in the torn basil right at the end.",
    "long": "I have been thinking about how we run the Monday meetings. At the moment they "
            "go on for over an hour and most people only need to be there for ten minutes "
            "of it. The updates from each team are useful but the discussion that follows "
            "often wanders.\n\nWhat I'd suggest is that each team posts a short written "
            "update by 9am, and the meeting itself is only for decisions that need "
            "everyone. We could try it for a month and see how people feel.\n\nThe other "
            "thing is the notes. Nobody reads them because they're too long. A list of "
            "decisions and who is doing what would be far more useful.",
}

# Markdown inputs whose formatting should survive.
MARKDOWN = {
    "md_list": "## Things to fix\n\n- The **login page** crashes on Safari\n- Emails "
               "are going to spam\n- The logo is blurry on retina screens",
    "md_link": "I have written up the results in [the report](https://example.com/report). "
               "The key finding is that response times improved by *roughly 40%*.",
    "md_steps": "### Setup\n\n1. Install `uv`\n2. Run `uv tool install tov`\n3. Set "
                "`GEMINI_API_KEY`",
}

FEATURES = {
    "heading": re.compile(r"^\s{0,3}#{1,6}\s", re.M),
    "bullet": re.compile(r"^\s*[-*+]\s+\S", re.M),
    "numbered": re.compile(r"^\s*\d+[.)]\s+\S", re.M),
    "bold": re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__"),
    "italic": re.compile(r"(?<![*\w])\*[^*\s][^*\n]*\*(?!\*)|(?<![_\w])_[^_\s][^_\n]*_(?!\w)"),
    "code": re.compile(r"`[^`\n]+`"),
    "link": re.compile(r"\[[^\]]+\]\([^)]+\)"),
}


def features(text: str) -> set[str]:
    return {name for name, rx in FEATURES.items() if rx.search(text)}


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    model = sys.argv[2] if len(sys.argv) > 2 else tov.DEFAULT_MODEL
    thinking = sys.argv[3] if len(sys.argv) > 3 else tov.DEFAULT_THINKING
    client = tov.make_client(120)

    def rewrite(setup: str, text: str) -> str:
        prompt, tone = SETUPS[setup]
        config = types.GenerateContentConfig(
            system_instruction=prompt.format(tone=tone),
            thinking_config=None if thinking == "default"
            else types.ThinkingConfig(thinking_level=thinking.upper()),
        )
        for attempt in range(3):
            try:
                return client.models.generate_content(model=model, contents=text, config=config).text or ""
            except Exception as e:  # retry transient errors
                if attempt == 2:
                    return f"<<error: {e}>>"
        return ""

    inputs = {**PLAIN, **MARKDOWN}
    if len(sys.argv) > 4:
        wanted = sys.argv[4].split(",")
        unknown = [w for w in wanted if w not in inputs]
        if unknown:
            sys.exit(f"unknown inputs: {', '.join(unknown)}")
        inputs = {k: v for k, v in inputs.items() if k in wanted}
    jobs = [(s, name, i) for s in SETUPS for name in inputs for i in range(runs)]
    print(f"{len(jobs)} rewrites with {model} (thinking {thinking}), {runs} runs per input\n",
          file=sys.stderr)
    with cf.ThreadPoolExecutor(12) as ex:
        outputs = dict(zip(jobs, ex.map(lambda j: rewrite(j[0], inputs[j[1]]), jobs)))

    errors = sum(o.startswith("<<error") for o in outputs.values())
    for setup in SETUPS:
        added, lost, added_kinds, lost_kinds = 0, 0, Counter(), Counter()
        for name, text in inputs.items():
            before = features(text)
            for i in range(runs):
                after = features(outputs[(setup, name, i)])
                if name in PLAIN:
                    extra = after - before
                    added += bool(extra)
                    added_kinds.update(extra)
                else:
                    missing = before - after
                    lost += bool(missing)
                    lost_kinds.update(missing)
        p_total = sum(k in PLAIN for k in inputs) * runs
        m_total = sum(k in MARKDOWN for k in inputs) * runs
        print(f"{setup}")
        if p_total:
            print(f"  plain text with Markdown added:  {added:>2}/{p_total}  {dict(added_kinds) or ''}")
        if m_total:
            print(f"  Markdown input that lost some:   {lost:>2}/{m_total}  {dict(lost_kinds) or ''}")
    if errors:
        print(f"\n{errors} rewrites failed", file=sys.stderr)

    out = Path(__file__).with_suffix(".last.txt")
    with out.open("w") as f:
        for (setup, name, i), o in outputs.items():
            f.write(f"=== {setup} | {name} | run {i + 1} | {sorted(features(o))}\n{o}\n\n")
    print(f"\nAll outputs: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
