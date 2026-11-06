#!/usr/bin/env bash
# Prepare Airflow's metadata database (inside the lexrag Postgres) and the lex.uz pool,
# then run what was asked: `standalone` by default, or e.g. `airflow dags test lexrag_daily`.
set -euo pipefail

python - <<'PY'
import os, psycopg
url = os.environ["LEXRAG_DATABASE_URL"]
with psycopg.connect(url, autocommit=True) as conn:
    if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'airflow'").fetchone():
        conn.execute("CREATE DATABASE airflow")
        print("created database airflow")
PY

airflow db migrate >/dev/null
airflow pools set lexuz 1 "lex.uz: one request at a time (the client also keeps ~1 request/s)" >/dev/null

if [ "$#" -eq 0 ]; then
  exec airflow standalone
fi
exec "$@"
