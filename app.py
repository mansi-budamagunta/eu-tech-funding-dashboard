import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.environ["SEDIA_DB_URL"])
MODEL, PROMPT_VERSION = "phi4-mini:latest", "v2"

SYSTEM_PROMPT = """You are an analyst evaluating EU-funded projects.

Return:
- label: 1 or 0
- rationale: one sentence explaining the decision using concrete evidence from the project description

Use:
1 = yes
0 = no, ambiguous, or irrelevant

Ensure that the label agrees with the rationale"""

QUESTION_LABELS = {
    "d3": "Domestic EU capacity",
    "g1": "Open-source commitment",
    "g2": "Open standards / interoperability",
}

QUESTIONS = {
    "d3": (
        "Does the project aim to build domestic capacity within the EU by "
        "developing infrastructure, platforms, facilities, production "
        "capabilities, or deployable technical systems?"
    ),
    "g1": "Does the project commit to producing or using open-source software?",
    "g2": (
        "Does the project commit to open standards or interoperability that "
        "prevent single-vendor lock-in?"
    ),
}

project_sql = text("""
    SELECT DISTINCT ON (reference)
        reference::text AS reference,
        metadata::jsonb #>> '{callIdentifier,0}' AS call_identifier,
        metadata::jsonb #>> '{euContributionAmount,0}' AS eu_contribution,
        metadata::jsonb #>> '{startDate,0}' AS start_date
    FROM raw_projects
    WHERE metadata::jsonb #>> '{callIdentifier,0}' LIKE 'HORIZON-CL4%'
    ORDER BY reference
""")
annotation_sql = text("""
    WITH ranked AS (
        SELECT reference, question_id, label,
               row_number() OVER (
                   PARTITION BY reference, question_id
                   ORDER BY annotated_at DESC, id DESC
               ) AS rn
        FROM project_annotations
        WHERE status = 'success'
          AND model = :model
          AND prompt_version = :prompt_version
          AND question_id IN ('d3', 'g1', 'g2')
    )
    SELECT reference, question_id, label
    FROM ranked WHERE rn = 1
""")

with engine.connect() as conn:
    projects = pd.read_sql(project_sql, conn)
    annotations = pd.read_sql(
        annotation_sql, conn, params={"model": MODEL, "prompt_version": PROMPT_VERSION}
    )

projects = projects.assign(
    eu_contribution=lambda x: pd.to_numeric(x.eu_contribution, errors="coerce"),
    start_date=lambda x: pd.to_datetime(x.start_date, errors="coerce", utc=True),
)
projects["start_year"] = projects.start_date.dt.year
annotations["label"] = pd.to_numeric(annotations.label, errors="coerce")
assert annotations.label.isin([0, 1]).all()
assert not annotations.duplicated(["reference", "question_id"]).any()

data = annotations.merge(projects, on="reference", validate="many_to_one")
complete_refs = data.groupby("reference").question_id.nunique().eq(len(QUESTION_LABELS))
complete_refs = complete_refs[complete_refs].index
complete = data[data.reference.isin(complete_refs)].copy()

coverage = pd.Series({
    "Total HORIZON-CL4 projects": projects.reference.nunique(),
    "Fully annotated projects": len(complete_refs),
    "Not yet fully annotated projects": projects.reference.nunique() - len(complete_refs),
    "Fully annotated projects with funding": complete.dropna(subset=["eu_contribution"]).reference.nunique(),
})

unweighted = (
    complete.groupby("question_id", as_index=False)
    .agg(positive_count=("label", "sum"), annotated_count=("reference", "nunique"), positive_share=("label", "mean"))
    .assign(
        question=lambda x: x.question_id.map(QUESTION_LABELS),
        result_label=lambda x: x.positive_count.astype(int).astype(str) + " / " + x.annotated_count.astype(str),
    )
)

funded = complete.dropna(subset=["eu_contribution"]).assign(
    positive_funding=lambda x: x.label * x.eu_contribution
)

def funding_summary(df, groups=()):
    group_cols = [*groups, "question_id"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            positive_funding=("positive_funding", "sum"),
            annotated_funding=("eu_contribution", "sum"),
            funded_project_count=("reference", "nunique"),
        )
        .assign(
            positive_funding_share=lambda x: x.positive_funding / x.annotated_funding,
            question=lambda x: x.question_id.map(QUESTION_LABELS),
        )
    )

funding = funding_summary(funded).assign(
    result_label=lambda x: x.positive_funding_share.map("{:.1%}".format)
)
trends = funding_summary(funded.dropna(subset=["start_year"]), ["start_year"])

order = list(QUESTION_LABELS.values())
unweighted_figure = px.bar(
    unweighted, x="question", y="positive_share", text="result_label",
    category_orders={"question": order}, range_y=[0, 1],
    title="Share of fully annotated projects with a positive label",
    labels={"positive_share": "Positive share", "question": "Question"},
)
unweighted_figure.update_traces(
    customdata=unweighted[["positive_count", "annotated_count"]],
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Positive projects: %{customdata[0]:,.0f}<br>"
        "Fully annotated projects: %{customdata[1]:,.0f}<br>"
        "Positive share: %{y:.1%}<extra></extra>"
    ),
)
funding_figure = px.bar(
    funding, x="question", y="positive_funding_share", text="result_label",
    category_orders={"question": order}, range_y=[0, 1],
    title="Funding-weighted share with a positive label",
    labels={"positive_funding_share": "Share of EU contribution", "question": "Question"},
)
funding_figure.update_traces(
    customdata=funding[["positive_funding", "annotated_funding",
                        "funded_project_count"]],
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Positive funding: €%{customdata[0]:,.0f}<br>"
        "Annotated funding: €%{customdata[1]:,.0f}<br>"
        "Projects with funding: %{customdata[2]:,.0f}<br>"
        "Funding-weighted share: %{y:.1%}<extra></extra>"
    ),
)
trend_figure = px.line(
    trends, x="start_year", y="positive_funding_share", facet_row="question",
    markers=True, category_orders={"question": order}, range_y=[0, 1],
    title="Funding-weighted positive share by project start year",
    labels={"start_year": "Start year", "positive_funding_share": "Share of EU contribution"},
    height=750,
)
for fig in (unweighted_figure, funding_figure, trend_figure):
    fig.update_yaxes(tickformat=".0%")
trend_figure.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

app = Dash(__name__)
server = app.server

def metric(label):
    return html.Div(
        [
            html.Strong(f"{int(coverage[label]):,}"),
            html.Div(label),
        ],
        style={
            "padding": "1rem",
            "border": "1px solid #ddd",
            "borderRadius": "0.5rem",
            "textAlign": "center",
            "flex": "1",
        },
    )

app.layout = html.Div(
    [
        html.H1("HORIZON-CL4 project annotations"),
        html.P(
            f"Exploratory model-generated annotations using {MODEL} / {PROMPT_VERSION}. "
            "These labels have not yet been manually validated."
        ),
        html.Div(
            [
                metric("Total HORIZON-CL4 projects"),
                metric("Fully annotated projects"),
                metric("Not yet fully annotated projects"),
                metric("Fully annotated projects with funding"),
            ],
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
        ),
        dcc.Graph(figure=unweighted_figure),
        dcc.Graph(figure=funding_figure),
        html.P(
            "Funding-weighted results use EU contribution and exclude projects "
            "without a usable funding value."
        ),
        html.Details(
            [
                html.Summary(f"Annotation methodology and prompts — {MODEL} / {PROMPT_VERSION}"),
                html.H3("Questions"),
                html.Ul([
                    html.Li([html.Strong(f"{qid}: "), question])
                    for qid, question in QUESTIONS.items()
                ]),
                html.H3("System prompt"),
                html.Pre(
                    SYSTEM_PROMPT,
                    style={
                        "whiteSpace": "pre-wrap",
                        "backgroundColor": "#f6f6f6",
                        "padding": "1rem",
                        "borderRadius": "0.5rem",
                    },
                ),
            ],
            style={"marginTop": "2rem"},
        ),        
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "2rem",
    },
)

if __name__ == "__main__":
    app.run(debug=True)
