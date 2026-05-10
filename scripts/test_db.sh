docker run --rm -d --name pg-smart-test \
  -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test \
  -p 5433:5432 postgres:16

sleep 3

TEST_DATABASE_URL=postgresql+asyncpg://test:test@localhost:5433/test \
  .venv/bin/pytest tests/test_migrations.py -v

docker stop pg-smart-test