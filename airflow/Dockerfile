# Airflow 3 with the lexrag package installed, so DAG tasks can import it.
#   docker compose --profile airflow up -d --build      → http://127.0.0.1:8080
FROM apache/airflow:3.3.2-python3.12

COPY --chown=airflow:root pyproject.toml uv.lock README.md /opt/lexrag/
COPY --chown=airflow:root src /opt/lexrag/src
RUN pip install --no-cache-dir /opt/lexrag

COPY --chown=airflow:root airflow/start.sh /opt/lexrag/start.sh
ENTRYPOINT ["/usr/bin/dumb-init", "--", "bash", "/opt/lexrag/start.sh"]
