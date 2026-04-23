from __future__ import annotations

import html
import re


def highlight_query_terms(text: str, query: str) -> str:
    highlighted = html.escape(text)
    terms = [term.strip() for term in re.findall(r"\w+", query) if len(term.strip()) > 2]
    unique_terms = sorted(set(terms), key=len, reverse=True)

    for term in unique_terms:
        pattern = re.compile(rf"(?i)\b({re.escape(term)})\b")
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)
    return highlighted


def keyword_snippets(text: str, keyword: str, window: int = 90, limit: int = 5) -> list[str]:
    if not keyword.strip():
        return []
    matches = []
    for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = text[start:end].replace("\n", " ").strip()
        matches.append(f"...{snippet}...")
        if len(matches) >= limit:
            break
    return matches


def apply_theme(dark_mode: bool) -> str:
    if dark_mode:
        return """
        <style>
        html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stButton, .stSelectbox {
            font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important;
        }
        .stApp {
            background: radial-gradient(circle at top, #171717 0%, #090909 55%, #040404 100%);
            color: #e5e7eb;
        }
        .block-container { padding-top: 1.25rem; max-width: 1200px; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f10 0%, #09090b 100%);
            border-right: 1px solid #232327;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.6rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }
        .nav-item {
            background: #121216;
            border: 1px solid #2b2b31;
            color: #f4f4f5;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.92rem;
        }
        .nav-sub {
            color: #9ca3af;
            margin: 8px 6px;
            font-size: 0.78rem;
        }
        h1, h2, h3 { color: #f8fafc !important; }
        .card {
            background: rgba(24, 24, 27, 0.85);
            border: 1px solid #2a2a2f;
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(6px);
        }
        .hero-title {
            text-align: center;
            font-size: 2.9rem;
            letter-spacing: -0.04em;
            font-weight: 700;
            margin-top: 0.3rem;
            margin-bottom: 0.25rem;
            color: #f9fafb;
        }
        .hero-sub {
            text-align: center;
            color: #a1a1aa;
            margin-bottom: 1.2rem;
        }
        .search-shell {
            background: rgba(20, 20, 23, 0.9);
            border: 1px solid #2a2a30;
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 0.6rem;
        }
        .search-shell-strong {
            background: linear-gradient(180deg, rgba(22,22,26,0.95) 0%, rgba(16,16,20,0.95) 100%);
            border: 1px solid #32323a;
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 0.8rem;
        }
        .search-title {
            color: #71717a;
            font-size: 0.95rem;
            margin-bottom: 6px;
        }
        .muted {
            color: #9ca3af;
            font-size: 0.86rem;
        }
        .answer-text {
            font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important;
            font-size: 1rem;
            line-height: 1.65;
            color: #e5e7eb;
            letter-spacing: 0.01em;
            word-spacing: 0.02em;
        }
        .chip-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        .chip {
            background: #17171b;
            border: 1px solid #2f2f36;
            color: #d4d4d8;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.82rem;
        }
        .auth-panel {
            background: rgba(24, 24, 27, 0.9);
            border: 1px solid #2a2a2f;
            border-radius: 14px;
            padding: 14px;
            margin-top: 6px;
        }
        .auth-heading {
            font-size: 0.95rem;
            color: #f4f4f5;
            margin-bottom: 0.2rem;
        }
        .auth-sub {
            color: #a1a1aa;
            font-size: 0.82rem;
            margin-bottom: 0.65rem;
        }
        .auth-btn {
            width: 100%;
            border: 1px solid #34343b;
            border-radius: 10px;
            background: #111113;
            color: #fafafa;
            padding: 9px 10px;
            text-align: center;
            margin-top: 8px;
            font-size: 0.84rem;
        }
        </style>
        """
    return """
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stButton, .stSelectbox {
        font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important;
    }
    .stApp { background: #f4f6fb; color: #0f172a; }
    .block-container { padding-top: 1.25rem; max-width: 1200px; }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem;
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
    h1, h2, h3 { color: #0f172a !important; }
    .card {
        background: #ffffff;
        border: 1px solid #dbe2ea;
        padding: 1rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(2, 6, 23, 0.06);
    }
    .hero-title {
        text-align: center;
        font-size: 2.9rem;
        letter-spacing: -0.04em;
        font-weight: 700;
        margin-top: 0.3rem;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    .hero-sub {
        text-align: center;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .search-shell {
        background: #ffffff;
        border: 1px solid #dbe2ea;
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 0.6rem;
    }
    .search-shell-strong {
        background: #ffffff;
        border: 1px solid #cfd8e3;
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 0.8rem;
    }
    .search-title {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .muted {
        color: #64748b;
        font-size: 0.86rem;
    }
    .answer-text {
        font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important;
        font-size: 1rem;
        line-height: 1.65;
        color: #0f172a;
        letter-spacing: 0.01em;
        word-spacing: 0.02em;
    }
    .chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .chip {
        background: #f8fafc;
        border: 1px solid #d7dee8;
        color: #334155;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.82rem;
    }
    .auth-panel {
        background: #ffffff;
        border: 1px solid #dbe2ea;
        border-radius: 14px;
        padding: 14px;
        margin-top: 6px;
    }
    .auth-heading {
        font-size: 0.95rem;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .auth-sub {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 0.65rem;
    }
    .auth-btn {
        width: 100%;
        border: 1px solid #d7dee8;
        border-radius: 10px;
        background: #f8fafc;
        color: #0f172a;
        padding: 9px 10px;
        text-align: center;
        margin-top: 8px;
        font-size: 0.84rem;
    }
    .nav-item {
        background: #f8fafc;
        border: 1px solid #d7dee8;
        color: #0f172a;
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .nav-sub {
        color: #64748b;
        margin: 8px 6px;
        font-size: 0.78rem;
    }
    </style>
    """
