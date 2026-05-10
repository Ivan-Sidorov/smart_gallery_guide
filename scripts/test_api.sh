docker run --rm -d --name pg-smart-test \
  -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test \
  -p 5433:5432 postgres:16

export DATABASE_URL=postgresql+asyncpg://test:test@localhost:5433/test
.venv/bin/alembic upgrade head

export API_LOAD_ML=false
.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000