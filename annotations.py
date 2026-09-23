"""Offline runner for annotating HORIZON-CL4 projects with Ollama.

The runner selects a bounded number of projects that have missing annotations
for the selected model, prompt version, and current objective text. Each result
is saved immediately so completed work survives an interrupted run.

This module is deliberately separate from the read-only Dash application.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"].rstrip("/")
OLLAMA_AUTH = (
    os.environ["OLLAMA_USERNAME"],
    os.environ["OLLAMA_PASSWORD"],
)

engine = create_engine(os.environ["SEDIA_DB_URL"])

DEFAULT_MODEL = "phi4-mini:latest"
PROMPT_VERSION = "v2"

# Keep accidental command-line runs small. The limit counts projects that
# require work, not simply the first projects returned by the database.
MAX_PROJECTS_PER_RUN = 100

SYSTEM_PROMPT = """You are an analyst evaluating EU-funded projects.

Return:
- label: 1 or 0
- rationale: one sentence explaining the decision using concrete evidence from the project description

Use:
1 = yes
0 = no, ambiguous, or irrelevant

Ensure that the label agrees with the rationale"""

QUESTIONS = {
    "d3": (
        "Does the project aim to build domestic capacity within the EU by "
        "developing infrastructure, platforms, facilities, production "
        "capabilities, or deployable technical systems?"
    ),
    "g1": (
        "Does the project commit to producing or using open-source software?"
    ),
    "g2": (
        "Does the project commit to open standards or interoperability that "
        "prevent single-vendor lock-in?"
    ),
}

logger = logging.getLogger("annotations")

def configure_logging():
    """Log concise progress to the terminal and a timestamped file."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"annotations_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # Suppress successful connection messages while retaining warnings
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_path


RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1]},
        "rationale": {"type": "string"},
    },
    "required": ["label", "rationale"],
    "additionalProperties": False,
}

# Failed rows may be replaced by a later retry. Successful rows are immutable:
# the WHERE clause prevents an existing success from being overwritten.
INSERT_ANNOTATION = text("""
    insert into project_annotations (
        reference,
        question_id,
        label,
        rationale,
        status,
        error_message,
        model,
        prompt_version,
        objective_hash
    )
    values (
        :reference,
        :question_id,
        :label,
        :rationale,
        :status,
        :error_message,
        :model,
        :prompt_version,
        :objective_hash
    )
    on conflict (
        reference,
        question_id,
        model,
        prompt_version,
        objective_hash
    )
    do update set
        label = excluded.label,
        rationale = excluded.rationale,
        status = excluded.status,
        error_message = excluded.error_message,
        annotated_at = now()
    where project_annotations.status = 'failed'
""")


def clean_objective(metadata_json):
    """Extract and normalize the first project objective from metadata JSON."""
    metadata = json.loads(metadata_json)
    objectives = metadata.get("objective") or []

    if not objectives:
        return None

    objective = BeautifulSoup(
        objectives[0],
        "html.parser",
    ).get_text(" ")

    objective = " ".join(objective.split())

    for punctuation in ",.;:!?":
        objective = objective.replace(f" {punctuation}", punctuation)

    return objective or None


def objective_hash(objective_text):
    """Return a stable identifier for the exact text sent to Ollama."""
    return hashlib.sha256(
        objective_text.encode("utf-8")
    ).hexdigest()


def validate_result(result):
    """Validate Ollama output beyond the JSON schema requested in the call."""
    label = result.get("label")
    rationale = result.get("rationale")

    # Use an exact type check because bool is a subclass of int in Python.
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"Invalid label: {label!r}")

    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Rationale is missing or empty")

    return {
        "label": label,
        "rationale": rationale.strip(),
    }


def annotate_one(objective_text, question_id, model):
    """Request and validate one project-question annotation from Ollama."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        auth=OLLAMA_AUTH,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Project description:\n{objective_text}\n\n"
                        f"Question:\n{QUESTIONS[question_id]}"
                    ),
                },
            ],
            "format": RESULT_SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()

    result = json.loads(
        response.json()["message"]["content"]
    )

    return {
        "model": model,
        "question_id": question_id,
        **validate_result(result),
    }


def annotate_safely(objective_text, question_id, model):
    """Return a persistable result for either success or an expected failure."""
    started = perf_counter()

    try:
        result = annotate_one(
            objective_text,
            question_id,
            model,
        )

        return {
            **result,
            "status": "success",
            "error_message": None,
            "elapsed_seconds": round(
                perf_counter() - started,
                1,
            ),
        }

    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        # Store expected request and response failures instead of terminating
        # the batch. Unexpected programming errors still propagate.
        return {
            "model": model,
            "question_id": question_id,
            "label": None,
            "rationale": None,
            "status": "failed",
            "error_message": (
                f"{type(exc).__name__}: {exc}"
            ),
            "elapsed_seconds": round(
                perf_counter() - started,
                1,
            ),
        }


def successful_question_ids(
    reference,
    model,
    prompt_version,
    obj_hash,
):
    """Return questions already completed for this exact annotation input."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                select question_id
                from project_annotations
                where reference = :reference
                  and model = :model
                  and prompt_version = :prompt_version
                  and objective_hash = :objective_hash
                  and status = 'success'
            """),
            {
                "reference": reference,
                "model": model,
                "prompt_version": prompt_version,
                "objective_hash": obj_hash,
            },
        ).scalars().all()

    return set(rows)


def save_annotation(result):
    """Persist one result and return the number of changed database rows."""
    row = {
        key: result[key]
        for key in (
            "reference",
            "question_id",
            "label",
            "rationale",
            "status",
            "error_message",
            "model",
            "prompt_version",
            "objective_hash",
        )
    }

    # One transaction per annotation preserves completed work if a later
    # Ollama request fails or the process is interrupted.
    with engine.begin() as conn:
        db_result = conn.execute(
            INSERT_ANNOTATION,
            row,
        )

    return db_result.rowcount


def select_projects_requiring_work(limit, model):
    """Select up to `limit` projects with at least one missing annotation."""
    selected = []

    with engine.connect() as conn:
        rows = conn.execute(text("""
            select "reference", "summary", "metadata"
            from raw_projects
            where nullif(btrim("metadata"), '') is not null
              and "metadata"::jsonb
                    #>> '{callIdentifier,0}'
                    like 'HORIZON-CL4%'
            order by "reference"
        """)).mappings().all()

    for row in rows:
        objective_text = clean_objective(row["metadata"])

        if not objective_text:
            continue

        obj_hash = objective_hash(objective_text)
        completed = successful_question_ids(
            row["reference"],
            model,
            PROMPT_VERSION,
            obj_hash,
        )
        missing = [
            question_id
            for question_id in QUESTIONS
            if question_id not in completed
        ]

        # A partially completed project counts as one selected project, but
        # only its missing questions will produce Ollama requests.
        if missing:
            selected.append({
                "reference": row["reference"],
                "summary": row["summary"],
                "objective_text": objective_text,
                "missing_question_ids": missing,
            })

        if len(selected) == limit:
            break

    return selected


def annotate_and_save_project(project, model):
    """Annotate and immediately persist every missing project question."""
    reference = project["reference"]
    objective_text = project["objective_text"]
    obj_hash = objective_hash(objective_text)

    results = []

    for question_id in project["missing_question_ids"]:
        result = {
            "reference": reference,
            "prompt_version": PROMPT_VERSION,
            "objective_hash": obj_hash,
            **annotate_safely(
                objective_text,
                question_id,
                model,
            ),
        }

        result["database_rows_changed"] = save_annotation(
            result
        )
        results.append(result)

        message = "%s %s %.1fs" % (
        question_id,
        result["status"],
        result["elapsed_seconds"],
    )

        if result["status"] == "success":
            logger.info(message)
        else:
            logger.warning(
                "%s — %s",
                message,
                result["error_message"],
            )

    return results


def run_annotations(projects, model):
    """Run one model sequentially over the selected projects."""
    results = []

    logger.info("Model: %s", model)
    logger.info("Prompt version: %s", PROMPT_VERSION)

    for index, project in enumerate(projects, start=1):
        logger.info(
            "\n[%d/%d] %s — %s",
            index,
            len(projects),
            project["reference"],
            project["summary"],
        )

        results.extend(
            annotate_and_save_project(project, model)
        )

    return results


def parse_args():
    """Parse the model and bounded project limit from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate unfinished HORIZON-CL4 projects "
            "using Ollama."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_PROJECTS_PER_RUN,
        help=(
            "Number of unfinished projects to process "
            f"(default and maximum: {MAX_PROJECTS_PER_RUN})"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main():
    """Select unfinished projects, annotate them, and report the outcome."""
    args = parse_args()
    log_path = configure_logging()

    logger.info("Log file: %s", log_path)

    if not 1 <= args.limit <= MAX_PROJECTS_PER_RUN:
        raise SystemExit(
            f"--limit must be between 1 and "
            f"{MAX_PROJECTS_PER_RUN}"
        )

    projects = select_projects_requiring_work(
        args.limit,
        args.model,
    )

    if not projects:
        logger.info("No projects require annotation.")
        return

    results = run_annotations(projects, args.model)
    successes = sum(
        result["status"] == "success"
        for result in results
    )
    failures = len(results) - successes

    logger.info(
        "\nFinished: %d projects, %d successful annotations, "
        "%d failed annotations.",
        len(projects),
        successes,
        failures,
    )

if __name__ == "__main__":
    main()