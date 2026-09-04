import os
import re
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["SEDIA_DB_URL"])

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
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "2rem",
    },
)

if __name__ == "__main__":
    app.run(debug=True)
