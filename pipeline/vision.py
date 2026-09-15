"""
pipeline/vision.py — image description via Groq vision model (qwen).

For each extracted image, the VLM:
    * classifies it (image / diagram / flowchart)
    * produces a DETAILED retrieval-oriented description (never "an image of...")
    * if flowchart -> also extracts the algorithmic representation (steps/conditions)

This turns a raw image into an embeddable TEXT representation, per the
"store original modality, embed a semantic representation" principle.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from mimetypes import guess_type

from groq import Groq

import config
from knowledge_unit import KnowledgeUnit, KnowledgeUnitType

CLASSIFY_PROMPT = """You are analyzing an image extracted from an academic/technical document.

Return a JSON object ONLY (no markdown, no commentary):
{{
  "category": "flowchart" | "diagram" | "table_snapshot" | "other",
  "description": "detailed retrieval-oriented description"
}}

Category guide:
- flowchart: contains sequence of steps/decisions with arrows and branches.
- diagram: architecture/relationship/process diagram (boxes, arrows, layers).
- table_snapshot: a table rendered as an image.
- other: any other photo or illustration.

Description rules:
- Describe what the figure represents and its purpose.
- Include EVERY visible label, term, acronym and number.
- Describe entities, relationships, processes, arrows and their direction.
- Use precise technical terminology.
- If the figure depicts a process, describe the ordering and the logic.
- Do NOT begin with "the image shows" - state the content directly.
- Be detailed enough that someone who cannot see it could reconstruct it.

Document context of this image:
Section: {section_path}
Surrounding text: {context}
"""

FLOWCHART_PROMPT = """Return a JSON object ONLY describing the flowchart/pseudo-code algorithm in the image.

{
  "title": "short title",
  "steps": ["ordered steps as plain text"],
  "conditions": [
     {"condition": "the decision text", "if": "what happens when true", "else": "what happens when false"}
  ],
  "inputs": ["inputs if visible"],
  "outputs": ["outputs if visible"]
}

Write the steps as if transcribing an algorithm. Preserve labels and branch
logic exactly. Everything must come from what is visible in the image.
"""


def _encode_image(image_path: str) -> str:
    mime, _ = guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _complete(client: Groq, prompt: str, image_path: str, max_tokens: int = 512) -> str | None:
    if not config.GROQ_API_KEY:
        return None
    url = _encode_image(image_path)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_VISION_MODEL,
                temperature=0,
                max_tokens=max_tokens,
                reasoning_effort="none",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
            )
            return resp.choices[0].message.content
        except Exception as exc:
            # rate limits (e.g. free-tier OTPM) and 5xx are transient; retry with backoff
            if attempt < 2 and getattr(exc, "status_code", None) in (429, 500, 502, 503):
                time.sleep(5 * (attempt + 1))
                continue
            print(f"[VISION] call failed for {image_path}: {exc}")
            return None


def _extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _surrounding_context(unit: KnowledgeUnit, max_chars: int = 400) -> str:
    # The unit itself carries its embedding_text (ref name). Neighbouring text is
    # not passed here; section_path gives the location context instead.
    return " (figure in page %d)" % unit.page


def process_image_unit(unit: KnowledgeUnit) -> KnowledgeUnit:
    """Classify + describe + (for flowcharts) extract algorithm. Mutates nothing
    except returning a new/updated unit with type/embedding_text/description set."""
    if unit.image_path is None:
        unit.embedding_text = "[image not extracted]"
        return unit

    client = Groq(api_key=config.GROQ_API_KEY)
    section = " / ".join(unit.section_path) if unit.section_path else "(none)"
    context = _surrounding_context(unit)

    classify_prompt = CLASSIFY_PROMPT.format(section_path=section, context=context)
    result = _extract_json(_complete(client, classify_prompt, unit.image_path))

    category = "other"
    description = ""
    if result:
        category = result.get("category", "other")
        description = result.get("description") or ""

    unit.type = KnowledgeUnitType.FLOWCHART if category == "flowchart" else KnowledgeUnitType.DIAGRAM if category == "diagram" else KnowledgeUnitType.IMAGE

    alg_text = ""
    if category == "flowchart":
        alg = _extract_json(_complete(client, FLOWCHART_PROMPT, unit.image_path))
        if alg:
            unit.structured_data = alg
            title = alg.get("title") or section
            steps = "\n".join(f"- {s}" for s in alg.get("steps", []))
            conds = "\n".join(
                f"- IF {c.get('condition')}: {c.get('if')} / ELSE: {c.get('else')}"
                for c in alg.get("conditions", [])
            )
            alg_text = f"\nAlgorithm ({title}):\n{steps}\n{conds}".strip()

    if description:
        unit.description = description
        base = description + (f"\n\n{alg_text}" if alg_text else "")
        unit.embedding_text = base
    else:
        unit.embedding_text = f"[image on page {unit.page}]"

    unit.content = unit.image_path or unit.content
    return unit


def describe_all_images(units: list[KnowledgeUnit], enabled: bool = True) -> list[KnowledgeUnit]:
    if not enabled or not config.GROQ_API_KEY:
        for u in units:
            if u.type in {KnowledgeUnitType.IMAGE, KnowledgeUnitType.DIAGRAM, KnowledgeUnitType.FLOWCHART} and u.image_path:
                u.embedding_text = f"[image: {u.image_path}]"
        return units
    for i, u in enumerate(units):
        if u.type is KnowledgeUnitType.IMAGE and u.image_path:
            print(f"[VISION] describing {u.image_path} ({i+1}/{len(units)}) ...")
            process_image_unit(u)
    return units