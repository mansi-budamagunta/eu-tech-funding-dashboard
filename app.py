import os
import re
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.environ["SEDIA_DB_URL"])

ANNOTATION_MODEL = "gemma3"
ANNOTATION_PROMPT_VERSION = "v1"

programme_query = text("""
    SELECT
        programme_id,
        programme_name,
        COUNT(*) AS project_count
    FROM raw_projects
    GROUP BY programme_id, programme_name
    ORDER BY project_count ASC
""")

with engine.connect() as conn:
    programmes = pd.read_sql(programme_query, conn)

total_projects = int(programmes["project_count"].sum())
total_programmes = len(programmes)

def compact_name(name):
    match = re.search(r"\(([^()]*)\)\s*$", name)
    if match:
        return match.group(1)

    return name if len(name) <= 30 else f"{name[:27]}..."

annotation_query = text("""
    WITH latest_annotations AS (
        SELECT
            reference,
            question_id,
            label,
            ROW_NUMBER() OVER (
                PARTITION BY reference, question_id
                ORDER BY annotated_at DESC, id DESC
            ) AS row_num
        FROM project_annotations
        WHERE status = 'success'
          AND model = :model
          AND prompt_version = :prompt_version
    )
    SELECT
        question_id,
        SUM(label) AS positive_count,
        COUNT(*) AS annotated_count,
        SUM(label)::float / COUNT(*) AS positive_share
    FROM latest_annotations
    WHERE row_num = 1
    GROUP BY question_id
    ORDER BY question_id
""")

with engine.connect() as conn:
    annotations = pd.read_sql(
        annotation_query,
        conn,
        params={
            "model": ANNOTATION_MODEL,
            "prompt_version": ANNOTATION_PROMPT_VERSION,
        },
    )

question_labels = {
    "d3": "Domestic EU capacity",
    "g1": "Open-source commitment",
    "g2": "Open standards / interoperability",
}

annotations["question"] = annotations["question_id"].map(question_labels)
annotations["result_label"] = (
    annotations["positive_count"].astype(int).astype(str)
    + " / "
    + annotations["annotated_count"].astype(int).astype(str)
)

programmes["display_name"] = programmes["programme_name"].map(compact_name)

figure = px.bar(
    programmes,
    x="project_count",
    y="display_name",
    orientation="h",
    text="project_count",
    custom_data=["programme_name"],
    labels={
        "project_count": "Projects",
        "display_name": "Programme",
    },
    title="Projects by programme",
    height=max(550, len(programmes) * 30),
)

figure.update_traces(
    texttemplate="%{text:,}",
    textposition="auto",
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Projects: %{x:,}"
        "<extra></extra>"
    ),
)

figure.update_layout(
    yaxis={
        "title": None,
        "automargin": False,
    },
    margin={"l": 110, "r": 40, "t": 60, "b": 50},
)

annotation_figure = px.bar(
    annotations,
    x="question",
    y="positive_share",
    text="result_label",
    labels={
        "question": "Question",
        "positive_share": "Positive share",
    },
    title=(
        f"Annotation plumbing test — {ANNOTATION_MODEL} / "
        f"{ANNOTATION_PROMPT_VERSION}"
    ),
    height=420,
)

annotation_figure.update_traces(
    textposition="auto",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Positive: %{customdata[0]:.0f}<br>"
        "Annotated: %{customdata[1]:.0f}<br>"
        "Positive share: %{y:.1%}"
        "<extra></extra>"
    ),
    customdata=annotations[["positive_count", "annotated_count"]],
)

annotation_figure.update_layout(
    yaxis={
        "tickformat": ".0%",
        "range": [0, 1],
        "title": "Positive share",
    },
    margin={"l": 80, "r": 40, "t": 60, "b": 100},
)


app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H1("SEDIA dashboard"),
        html.Div(
            [
                html.P(f"{total_projects:,} projects loaded"),
                html.P(f"{total_programmes:,} programmes"),
            ]
        ),
        dcc.Graph(figure=figure),
        html.H2("Project annotations"),
        html.P(
            "Plumbing test only: labels are not yet validated research findings."
        ),
        dcc.Graph(figure=annotation_figure),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "2rem",
    },
)

if __name__ == "__main__":
    app.run(debug=True)
