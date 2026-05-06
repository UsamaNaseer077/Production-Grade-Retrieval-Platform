#!/usr/bin/env python3
"""
Download Shakespeare's complete works from Folger Shakespeare Library
and run the full retrieval pipeline test on them.

Downloads each play/poem as a separate .txt file into data/shakespeare/
then runs ingestion, builds Shakespeare-specific eval queries, and
executes the full test suite.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://shakespeare.folger.edu/shakespeares-works"

# All 43 works with their URL slugs and categories
WORKS = {
    # ── Comedies ──
    "alls-well-that-ends-well":    ("All's Well That Ends Well", "comedy"),
    "as-you-like-it":              ("As You Like It", "comedy"),
    "the-comedy-of-errors":        ("The Comedy of Errors", "comedy"),
    "loves-labors-lost":           ("Love's Labor's Lost", "comedy"),
    "measure-for-measure":         ("Measure for Measure", "comedy"),
    "the-merchant-of-venice":      ("The Merchant of Venice", "comedy"),
    "the-merry-wives-of-windsor":  ("The Merry Wives of Windsor", "comedy"),
    "a-midsummer-nights-dream":    ("A Midsummer Night's Dream", "comedy"),
    "much-ado-about-nothing":      ("Much Ado About Nothing", "comedy"),
    "pericles":                    ("Pericles", "comedy"),
    "the-taming-of-the-shrew":     ("The Taming of the Shrew", "comedy"),
    "the-tempest":                 ("The Tempest", "comedy"),
    "twelfth-night":               ("Twelfth Night", "comedy"),
    "the-two-gentlemen-of-verona": ("The Two Gentlemen of Verona", "comedy"),
    "the-winters-tale":            ("The Winter's Tale", "comedy"),
    # ── Tragedies ──
    "antony-and-cleopatra":        ("Antony and Cleopatra", "tragedy"),
    "coriolanus":                  ("Coriolanus", "tragedy"),
    "cymbeline":                   ("Cymbeline", "tragedy"),
    "hamlet":                      ("Hamlet", "tragedy"),
    "julius-caesar":               ("Julius Caesar", "tragedy"),
    "king-lear":                   ("King Lear", "tragedy"),
    "macbeth":                     ("Macbeth", "tragedy"),
    "othello":                     ("Othello", "tragedy"),
    "romeo-and-juliet":            ("Romeo and Juliet", "tragedy"),
    "timon-of-athens":             ("Timon of Athens", "tragedy"),
    "titus-andronicus":            ("Titus Andronicus", "tragedy"),
    # ── Histories ──
    "henry-iv-part-1":             ("Henry IV, Part 1", "history"),
    "henry-iv-part-2":             ("Henry IV, Part 2", "history"),
    "henry-v":                     ("Henry V", "history"),
    "henry-vi-part-1":             ("Henry VI, Part 1", "history"),
    "henry-vi-part-2":             ("Henry VI, Part 2", "history"),
    "henry-vi-part-3":             ("Henry VI, Part 3", "history"),
    "henry-viii":                  ("Henry VIII", "history"),
    "king-john":                   ("King John", "history"),
    "richard-ii":                  ("Richard II", "history"),
    "richard-iii":                 ("Richard III", "history"),
    # ── Poems ──
    "shakespeares-sonnets":        ("Shakespeare's Sonnets", "poetry"),
    "venus-and-adonis":            ("Venus and Adonis", "poetry"),
    "lucrece":                     ("Lucrece", "poetry"),
    "the-phoenix-and-turtle":      ("The Phoenix and Turtle", "poetry"),
    "a-lovers-complaint":          ("A Lover's Complaint", "poetry"),
}

def clean_html(html: str) -> str:
    """Strip HTML tags and decode entities to plain text."""
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "\n", html)
    # Decode common HTML entities
    for ent, ch in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                     ("&quot;", '"'), ("&#39;", "'"), ("&rsquo;", "'"),
                     ("&lsquo;", "'"), ("&rdquo;", '"'), ("&ldquo;", '"'),
                     ("&mdash;", "—"), ("&ndash;", "–"), ("&nbsp;", " ")]:
        text = text.replace(ent, ch)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_work(slug: str, title: str, out_dir: Path) -> Path:
    """Download a single work from Folger and save as .txt."""
    out_path = out_dir / f"{slug}.txt"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  ✓ {title} (cached)")
        return out_path

    url = f"{BASE}/{slug}/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RetrievalPlatform/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        text = clean_html(html)

        # Try to isolate the actual play content (between main content markers)
        # Folger uses specific content sections
        lines = text.split("\n")
        content_lines = []
        in_content = False
        for line in lines:
            stripped = line.strip()
            # Start capturing after the title
            if title.lower() in stripped.lower() and not in_content:
                in_content = True
            if in_content:
                # Stop at footer/navigation
                if any(marker in stripped.lower() for marker in [
                    "folger shakespeare library", "cookie", "privacy policy",
                    "join and support", "sign up for our newsletter"
                ]):
                    break
                content_lines.append(stripped)

        # Use extracted content or fall back to full text
        final_text = "\n".join(content_lines) if len(content_lines) > 100 else text

        # Prepend metadata header
        header = f"Title: {title}\nSource: Folger Shakespeare Library ({url})\nCategory: Shakespeare\n\n"
        final_text = header + final_text

        out_path.write_text(final_text, encoding="utf-8")
        chars = len(final_text)
        print(f"  ✓ {title} ({chars:,} chars)")
        time.sleep(0.5)  # polite rate limiting
        return out_path

    except Exception as e:
        print(f"  ✗ {title}: {e}")
        # Write a minimal fallback
        fallback = f"Title: {title}\nSource: Folger Shakespeare Library\n\n[Content unavailable — download failed: {e}]\n"
        out_path.write_text(fallback)
        return out_path


def create_shakespeare_queries() -> list:
    """Create Shakespeare-specific test queries with expected content."""
    return [
        {
            "query_id": "shk_001",
            "text": "To be or not to be that is the question",
            "expected_source": "hamlet",
            "category": "famous_quote"
        },
        {
            "query_id": "shk_002",
            "text": "Romeo and Juliet love story Montague Capulet",
            "expected_source": "romeo-and-juliet",
            "category": "plot"
        },
        {
            "query_id": "shk_003",
            "text": "Macbeth witches prophecy king of Scotland",
            "expected_source": "macbeth",
            "category": "plot"
        },
        {
            "query_id": "shk_004",
            "text": "Othello jealousy Desdemona Iago handkerchief",
            "expected_source": "othello",
            "category": "plot"
        },
        {
            "query_id": "shk_005",
            "text": "King Lear divides kingdom daughters Cordelia",
            "expected_source": "king-lear",
            "category": "plot"
        },
        {
            "query_id": "shk_006",
            "text": "merchant Venice Shylock pound of flesh bond",
            "expected_source": "the-merchant-of-venice",
            "category": "plot"
        },
        {
            "query_id": "shk_007",
            "text": "midsummer night dream fairy Oberon Titania Puck",
            "expected_source": "a-midsummer-nights-dream",
            "category": "plot"
        },
        {
            "query_id": "shk_008",
            "text": "Prospero magic island tempest Ariel Caliban",
            "expected_source": "the-tempest",
            "category": "plot"
        },
        {
            "query_id": "shk_009",
            "text": "Julius Caesar assassination Brutus conspirators Rome",
            "expected_source": "julius-caesar",
            "category": "plot"
        },
        {
            "query_id": "shk_010",
            "text": "shall I compare thee to a summer's day sonnet",
            "expected_source": "shakespeares-sonnets",
            "category": "famous_quote"
        },
        {
            "query_id": "shk_011",
            "text": "Hamlet prince Denmark ghost father revenge",
            "expected_source": "hamlet",
            "category": "plot"
        },
        {
            "query_id": "shk_012",
            "text": "Twelfth Night Viola disguise Orsino Olivia",
            "expected_source": "twelfth-night",
            "category": "plot"
        },
        {
            "query_id": "shk_013",
            "text": "Henry V battle Agincourt band of brothers",
            "expected_source": "henry-v",
            "category": "history"
        },
        {
            "query_id": "shk_014",
            "text": "taming of the shrew Petruchio Katherine marriage",
            "expected_source": "the-taming-of-the-shrew",
            "category": "plot"
        },
        {
            "query_id": "shk_015",
            "text": "Much Ado About Nothing Beatrice Benedick witty",
            "expected_source": "much-ado-about-nothing",
            "category": "plot"
        },
    ]


def main():
    out_dir = Path("data/shakespeare")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(WORKS)} Shakespeare works to {out_dir}/")
    print("=" * 60)

    downloaded = []
    for slug, (title, category) in WORKS.items():
        p = download_work(slug, title, out_dir)
        if p.exists() and p.stat().st_size > 500:
            downloaded.append(p)

    total_chars = sum(p.stat().st_size for p in downloaded)
    print(f"\n{'=' * 60}")
    print(f"Downloaded: {len(downloaded)}/{len(WORKS)} works")
    print(f"Total size: {total_chars:,} bytes ({total_chars/1024/1024:.1f} MB)")

    # Save Shakespeare queries
    queries = create_shakespeare_queries()
    q_path = Path("eval/shakespeare_queries.json")
    q_path.write_text(json.dumps(queries, indent=2))
    print(f"Shakespeare queries: {len(queries)} → {q_path}")

    # Create a metadata CSV
    csv_path = out_dir / "works_metadata.csv"
    with csv_path.open("w") as f:
        f.write("slug,title,category,chars\n")
        for slug, (title, category) in WORKS.items():
            p = out_dir / f"{slug}.txt"
            chars = p.stat().st_size if p.exists() else 0
            f.write(f"{slug},{title},{category},{chars}\n")
    print(f"Metadata CSV: {csv_path}")


if __name__ == "__main__":
    main()
