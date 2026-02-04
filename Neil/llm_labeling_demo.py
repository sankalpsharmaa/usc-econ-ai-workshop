"""
LLM Labeling Demo (API-only)

This script calls the OpenAI API directly to:
- Label each review with structured JSON (is_political, reasoning)
- Run a second-pass rewrite to enforce a word limit
"""

import re
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

# Load API key from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "fake_movie_reviews.csv"
OUTPUT_PATH = SCRIPT_DIR / "labeled_reviews_output.csv"
MODEL = "gpt-4o-mini"  # or "gpt-4o"

RUBRIC = (
    "Label is_political=1 ONLY if the review explicitly uses political / ideology / culture-war framing "
    "(e.g., woke, SJW, propaganda, left/right, party politics, identity politics, agenda). "
    "If it's just 'preachy' or 'message' without explicit politics, label 0."
)


# --- Data Generation ---
def maybe_make_fake_dataset(path: Path, n: int = 30, seed: int = 7):
    """Generate fake movie reviews dataset if it doesn't exist."""
    random.seed(seed)
    political = [
        "This is woke propaganda dressed as a movie.",
        "Another SJW agenda push. Hard pass.",
        "Left/right culture-war nonsense ruined the plot.",
        "Pure identity politics. Not cinema.",
        "Party politics in superhero form. Obvious agenda.",
        "This felt like partisan messaging, not storytelling.",
    ]
    nonpolitical = [
        "Bad pacing and weak dialogue.",
        "The acting was fine but the plot was messy.",
        "Great visuals, mediocre script.",
        "Too long; the third act dragged.",
        "Sound mixing was awful in my theater.",
        "Fun movie, not perfect, but enjoyable.",
    ]
    rows = []
    for i in range(1, n + 1):
        if random.random() < 0.35:
            txt = random.choice(political)
            y = 1
        else:
            txt = random.choice(nonpolitical)
            y = 0
        if random.random() < 0.25:
            txt += " " + random.choice(["Seriously.", "LOL.", "Just my opinion."])
        rows.append({"review_id": i, "review_text": txt.strip(), "true_is_political": y})
    pd.DataFrame(rows).to_csv(path, index=False)


# --- Pydantic Model for Structured Output ---
class PoliticalLabel(BaseModel):
    is_political: int = Field(..., description="0 or 1")
    reasoning: str = Field(..., description="brief explanation")


# --- Helper Functions ---
def build_prompt(review_text: str) -> str:
    return (
        f"Rubric: {RUBRIC}\n\n"
        "Task: Read the review and output JSON with fields {is_political, reasoning}.\n"
        "- is_political must be 0 or 1\n"
        "- reasoning should be <= 50 words (soft)\n\n"
        f"Review: {review_text}"
    )


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def label_one(client: OpenAI, review_text: str) -> dict:
    """Label a single review using the OpenAI API."""
    prompt = build_prompt(review_text)
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format=PoliticalLabel,
    )
    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(msg.refusal)
    parsed = msg.parsed
    return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()


def enforce_reasoning_limit(client: OpenAI, result: dict, max_words: int = 50) -> dict:
    """Rewrite reasoning to enforce word limit."""
    prompt_2 = f"Rewrite reasoning to <= {max_words} words. Keep is_political unchanged. JSON only."
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "assistant", "content": str(result)},
            {"role": "user", "content": prompt_2},
        ],
        response_format=PoliticalLabel,
    )
    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(msg.refusal)
    parsed = msg.parsed
    out = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
    # Hard guard
    if word_count(out["reasoning"]) > max_words:
        out["reasoning"] = " ".join(out["reasoning"].split()[:max_words])
    return out


def main():
    # Initialize OpenAI client (reads OPENAI_API_KEY from environment)
    client = OpenAI()

    # Generate fake dataset if needed
    if not DATA_PATH.exists():
        maybe_make_fake_dataset(DATA_PATH)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews")
    print(df.head())

    # Label all reviews
    rows = []
    for _, r in df.iterrows():
        out = label_one(client, r["review_text"])
        out = enforce_reasoning_limit(client, out, max_words=50)
        rows.append({
            "review_id": int(r["review_id"]),
            "review_text": r["review_text"],
            "true_is_political": int(r["true_is_political"]),
            "pred_is_political": int(out["is_political"]),
            "reasoning": out["reasoning"],
            "reasoning_words": word_count(out["reasoning"]),
        })

    out_df = pd.DataFrame(rows)
    print("\nLabeled reviews:")
    print(out_df.head())

    # Calculate accuracy
    acc = (out_df.true_is_political == out_df.pred_is_political).mean()
    print(f"\nAccuracy: {acc:.2%}")

    # Save output
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
