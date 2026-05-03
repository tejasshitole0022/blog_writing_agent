from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=8)
    target_words: int = Field(..., description="Target words (300-500).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)



class PlacedImageSpec(BaseModel):
    """One image: where to place it + what to generate."""
    placeholder: Literal["[[IMAGE_1]]", "[[IMAGE_2]]", "[[IMAGE_3]]"]
    after_heading: str = Field(..., description="Exact markdown heading after which the placeholder is inserted.")
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    images: List[PlacedImageSpec] = Field(default_factory=list, description="Up to 3 images, in document order.")

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    import json as _json
    # Use plain JSON output to avoid schema type coercion issues with small models
    prompt = (
        f"Topic: {state['topic']}\nAs-of date: {state['as_of']}\n\n"
        "Reply with ONLY a JSON object with keys: mode (string), needs_research (boolean), "
        "queries (array of strings). Example:\n"
        '{"mode": "closed_book", "needs_research": false, "queries": []}'
    )
    raw = llm.invoke([SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=prompt)]).content.strip()
    # Extract JSON from response
    start, end = raw.find("{"), raw.rfind("}") + 1
    data = _json.loads(raw[start:end]) if start != -1 else {}

    mode = data.get("mode", "closed_book")
    needs_research = bool(data.get("needs_research", False))
    queries = data.get("queries", [])
    if isinstance(queries, str):
        queries = []

    if mode == "open_book":
        recency_days = 7
    elif mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": needs_research,
        "mode": mode,
        "queries": queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]  # max 5 queries
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=3))  # max 3 results per query

    if not raw:
        return {"evidence": []}

    # Truncate snippets to keep token count low (Groq free tier limit)
    trimmed = []
    for r in raw[:12]:  # max 12 total results
        trimmed.append({
            "title": (r.get("title") or "")[:80],
            "url": r.get("url") or "",
            "snippet": (r.get("snippet") or "")[:200],
            "published_at": r.get("published_at"),
            "source": r.get("source"),
        })

    import json as _json
    prompt = (
        f"As-of date: {state['as_of']}\nRecency days: {state['recency_days']}\n\n"
        f"Raw results:\n{trimmed}\n\n"
        "Reply with ONLY a JSON object: {\"evidence\": [{\"title\": ..., \"url\": ..., \"snippet\": ..., \"published_at\": null}]}"
    )
    raw_resp = llm.invoke([SystemMessage(content=RESEARCH_SYSTEM), HumanMessage(content=prompt)]).content.strip()
    start, end = raw_resp.find("{"), raw_resp.rfind("}") + 1
    try:
        data = _json.loads(raw_resp[start:end]) if start != -1 else {}
        evidence = [EvidenceItem(**e) for e in data.get("evidence", []) if e.get("url")]
    except Exception:
        evidence = []

    dedup = {}
    for e in evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 4-6 tasks, each with goal + 4-6 bullets + target_words (300-500 words each).
- The blog should be comprehensive, detailed, and genuinely useful to the reader.
- Tags are flexible; do not force a fixed taxonomy.

ACCURACY RULES (critical):
- Only include claims you are highly confident are correct.
- Do NOT hallucinate library names, API names, version numbers, or features.
- If unsure about a specific fact, keep the bullet general rather than specific.
- Stick to well-known, verifiable facts only.

Grounding:
- closed_book: evergreen, no evidence dependence. Stick to verified facts only.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    import json as _json
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    evidence_text = _json.dumps([e.model_dump() for e in evidence][:8])
    prompt = (
        f"Topic: {state['topic']}\nMode: {mode}\nAs-of: {state['as_of']}\n"
        f"{'Force blog_kind=news_roundup.' if forced_kind else ''}\n"
        f"Evidence: {evidence_text}\n\n"
        "Reply with ONLY a JSON object matching this structure exactly:\n"
        '{"blog_title": "...", "audience": "...", "tone": "...", "blog_kind": "explainer", '
        '"constraints": [], "tasks": [{"id": 1, "title": "...", "goal": "...", '
        '"bullets": ["...", "...", "...", "...", "..."], "target_words": 400, "tags": [], '
        '"requires_research": false, "requires_citations": false, "requires_code": false}]}'
    )
    raw = llm.invoke([SystemMessage(content=ORCH_SYSTEM), HumanMessage(content=prompt)]).content.strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    data = _json.loads(raw[start:end])
    if forced_kind:
        data["blog_kind"] = "news_roundup"
    plan = Plan(**data)
    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order, expanding each into full paragraphs.
- Target words +-15%. Write rich, detailed, genuinely useful content — do NOT pad with filler.
- Output ONLY the section markdown starting with "## <Section Title>". No preamble, no meta-commentary.
- Use subheadings (###), bullet lists, and code blocks where they add clarity.

ACCURACY RULES (critical):
- Write only factually accurate content you are highly confident about.
- Do NOT invent statistics, version numbers, benchmarks, or specific claims.
- Do NOT hallucinate library names, function signatures, or API details.
- If a bullet asks for something you are not sure about, write a general accurate statement instead.
- Prefer well-known, verifiable facts over specific claims.

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal, correct, runnable snippet.
"""

def worker_node(payload: dict) -> dict:
    import time
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:8]
    )

    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n"
                f"Tone: {plan.tone}\n"
                f"Blog kind: {plan.blog_kind}\n"
                f"Topic: {payload['topic']}\n"
                f"Mode: {payload.get('mode')}\n\n"
                f"Section title: {task.title}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"requires_code: {task.requires_code}\n"
                f"Bullets:{bullets_text}\n\n"
                f"Evidence:\n{evidence_text}\n"
            )
        ),
    ]

    # Retry up to 4 times on rate limit with backoff
    for attempt in range(4):
        try:
            section_md = llm.invoke(messages).content.strip()
            return {"sections": [(task.id, section_md)]}
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 8 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    section_md = llm.invoke(messages).content.strip()
    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- If no images needed: return images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan (NO full markdown).
"""

def _insert_placeholders(md: str, images: list) -> str:
    lines = md.splitlines()
    for img in reversed(images):
        for i, line in enumerate(lines):
            if line.strip() == img.after_heading.strip():
                lines.insert(i + 1, f"\n{img.placeholder}\n")
                break
    return "\n".join(lines)

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Propose image placements and prompts for the blog below.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    md_with_placeholders = (
        _insert_placeholders(merged_md, image_plan.images)
        if image_plan.images
        else merged_md
    )

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    # Try models in order until one works
    models = [
        "models/gemini-2.5-flash-image",
        "models/gemini-3.1-flash-image-preview",
        "models/gemini-3-pro-image-preview",
    ]
    last_exc = None
    for model in models:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_ONLY_HIGH",
                        )
                    ],
                ),
            )
            parts = getattr(resp, "parts", None)
            if not parts and getattr(resp, "candidates", None):
                try:
                    parts = resp.candidates[0].content.parts
                except Exception:
                    parts = None
            if parts:
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return inline.data
        except Exception as e:
            last_exc = e
            print(f"[image] {model} failed: {e}")
            continue

    raise RuntimeError(f"All image models failed. Last error: {last_exc}")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        blogs_dir = Path("blogs")
        blogs_dir.mkdir(exist_ok=True)
        filename = f"{_safe_slug(plan.blog_title)}.md"
        (blogs_dir / filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _generate_one(spec):
        out_path = images_dir / spec["filename"]
        if not out_path.exists():
            img_bytes = _gemini_generate_image_bytes(spec["prompt"])
            out_path.write_bytes(img_bytes)
        return spec

    failed = set()
    with ThreadPoolExecutor(max_workers=len(image_specs)) as ex:
        futures = {ex.submit(_generate_one, spec): spec for spec in image_specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            exc = fut.exception()
            if exc:
                print(f"[image] FAILED {spec['filename']}: {exc}")
                failed.add(spec["placeholder"])

    for spec in image_specs:
        placeholder = spec["placeholder"]
        if placeholder in failed:
            md = md.replace(placeholder, "")
            continue
        img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    blogs_dir = Path("blogs")
    blogs_dir.mkdir(exist_ok=True)
    filename = f"{_safe_slug(plan.blog_title)}.md"
    (blogs_dir / filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

