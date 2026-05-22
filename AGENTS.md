# mcp-recipes

## Build & run

```sh
cargo build
cargo run                          # start MCP server
cargo run -- --clear               # delete all recipes from DB
cargo run -- --index --path=recipes.json  # index recipes from JSON
```

No test suite, no linter config, no formatter config.

## Config

All config via `.env` or env vars. See `.env.example`.

- `EMBEDDING_MODE`: `bm25` (no embeddings needed), `ollama`, `openrouter`
- `MCP_TRANSPORT`: `stdio` (default, for MCP clients) or `http` (for curl testing)
- DB: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGSSLMODE`

**Required for openrouter mode:** `OPENROUTER_API_KEY`

## Schema

Two tables in PostgreSQL with `pgvector`:

- `recipes_data` — recipe metadata, full-text search vector (generated, Spanish config)
- `recipes_vector` — embedding vectors (FK to `recipes_data.id`, `ON DELETE CASCADE`)

Auto-migration creates tables, migrates from legacy `recipes` table, and manages indexes (`idx_recipes_data_*`, `idx_recipes_vector_embedding`).

## Architecture

Six source files:

| File | Role |
|---|---|
| `main.rs` | Entrypoint, CLI dispatch, server init (stdio or HTTP) |
| `config.rs` | Env var parsing, `EmbeddingMode`/`Transport` enums |
| `db.rs` | PostgreSQL pool, SQL queries, `PgVectorDb` struct |
| `embeddings.rs` | Embedding providers: Ollama API, OpenRouter API |
| `tools.rs` | MCP tool handlers (7 tools), `RecipesHandler` |
| `models.rs` | `Recipe`/`Ingredient` structs with null-string handling |

## MCP tools

- `search_recipes` — hybrid BM25 + vector. In `bm25` mode skips embedding and uses pure full-text. In `ollama`/`openrouter` mode uses 70% vector + 30% text weighting.
- `get_recipe_by_id` — full recipe by UUID
- `search_by_ingredients` — JSONB containment query (OR logic, exact `name` match)
- `search_by_filters` — AND filters on course/food_type/chef/difficulty/time
- `stats` — total and with-embeddings counts
- `index_recipes` — reads `recipes.json`, generates embeddings, upserts
- `clear_db` — deletes all rows from both tables

## HTTP transport quirks

When `MCP_TRANSPORT=http`:

- Server listens at `http://{MCP_HOST}:{MCP_PORT}`
- JSON-RPC endpoint is `POST /mcp`
- **Required headers:** `Content-Type: application/json`, `Accept: application/json, text/event-stream`
- **Session required:** call `initialize` first to get `Mcp-Session-Id` (returned as response header), include it in subsequent requests
- Response body is SSE format (`data: <json>\n\n`)
- No keepalive/health endpoint configured
