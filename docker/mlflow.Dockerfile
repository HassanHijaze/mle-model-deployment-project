# MLflow server with the PostgreSQL driver.
# MLFLOW_VERSION comes from .env and should match the mlflow version in your project.
ARG MLFLOW_VERSION
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}

RUN pip install --no-cache-dir "psycopg[binary]==3.3.9"
