# EU Tech Funding Dashboard

A dashboard investigating how projects funded under Horizon Europe’s
[Cluster 4: Digital, Industry and Space](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe/cluster-4-digital-industry-and-space_en)
align with EU technological-sovereignty goals.

It can be viewed here: https://dash.criticalinfralab.net/

The dashboard presents:

- annotation coverage;
- unweighted results by research question;
- funding-weighted results; and
- annual trends.

The annotations are model-generated and have not yet been manually validated.

Related repository:
[SEDIA API Fetchers](https://github.com/mansi-budamagunta/sedia-api-fetchers)

## Installation

```bash
git clone https://github.com/mansi-budamagunta/eu-tech-funding-dashboard.git
cd eu-tech-funding-dashboard

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Directory structure

```text
├── app.py            # Dash application and database queries
├── annotations.py    # Generates LLM-based annotations for selected projects
├── requirements.txt  # Python dependencies
└── README.md         # Installation and operation guide
```

## Database configuration

The dashboard requires an existing PostgreSQL database containing the project
and annotation data.

Provide its SQLAlchemy connection URL through `SEDIA_DB_URL`. It can be set in
the environment or in a `.env` file in the repository root:

```text
SEDIA_DB_URL=postgresql+psycopg://username:password@localhost:5432/database_name
```

The database account only needs read access to the tables used by the
dashboard.

## Run locally

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:8050>.

## Run with Gunicorn

For service use, run the Dash application through its exported WSGI server:

```bash
.venv/bin/gunicorn --bind 127.0.0.1:8050 app:server
```
