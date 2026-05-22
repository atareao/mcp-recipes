# MCP Recipes

An MCP (Model Context Protocol) server for searching and retrieving cooking recipes with hybrid BM25 + vector search.

## Features

- **Hybrid search** — combines BM25 full-text search with vector similarity (70% vector, 30% text) for better relevance
- **BM25-only mode** — works without any embedding service for zero-dependency deployments
- **Flexible embedding backends** — supports Ollama (local) or OpenRouter (remote API)
- **7 MCP tools** — search, filter, browse, and manage a recipe database
- **PostgreSQL + pgvector** — production-ready vector database with auto-migration
- **Dual transport** — stdio (for MCP clients) or HTTP (for development/testing)

## Quick start

### Prerequisites

- Rust 1.75+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension
- (optional) [Ollama](https://ollama.com/) with an embedding model, or an OpenRouter API key

### Setup

```sh
cp .env.example .env
# Edit .env with your database credentials
cargo build
cargo run
```

### Configuration

All config is via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `PGHOST` | `pgvector.one.belcar.corp` | PostgreSQL host |
| `PGPORT` | `5432` | PostgreSQL port |
| `PGDATABASE` | `mcp` | Database name |
| `PGUSER` | `rag_admin` | Database user |
| `PGPASSWORD` | `rag_password` | Database password |
| `PGSSLMODE` | `require` | SSL mode (use `disable` for local) |
| `EMBEDDING_MODE` | `ollama` | `bm25`, `ollama`, or `openrouter` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `bge-m3:latest` | Ollama embedding model |
| `OPENROUTER_API_KEY` | — | Required for `openrouter` mode |
| `OPENROUTER_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | OpenRouter model ID |
| `EMBEDDING_DIM` | `1024` | Embedding dimension (must match model) |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http` |
| `MCP_HOST` | `127.0.0.1` | HTTP listen host |
| `MCP_PORT` | `3011` | HTTP listen port |

## Usage

### CLI commands

```sh
# Start the MCP server (stdio)
cargo run

# Clear all recipes from the database
cargo run -- --clear

# Index recipes from a JSON file (requires embedding mode)
cargo run -- --index --path=recipes.json
```

### MCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `search_recipes` | `query` (req), `limit`, `offset`, `min_similarity` | Hybrid search over all recipes |
| `get_recipe_by_id` | `id` (req) | Full recipe details by UUID |
| `search_by_ingredients` | `ingredients` (req, array), `limit` | OR-match on ingredient names |
| `search_by_filters` | `course`, `food_type`, `chef`, `max_difficulty`, `max_total_time`, `limit` | AND filters on structured fields |
| `stats` | — | Database statistics |
| `index_recipes` | `path` | Index recipes from JSON file |
| `clear_db` | — | Delete all recipes |

### HTTP transport

When using `MCP_TRANSPORT=http`, the server exposes a JSON-RPC endpoint at `POST /mcp`.

```sh
# Initialize session
curl -X POST http://localhost:3011/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize"}'

# Use returned Mcp-Session-Id in subsequent requests
curl -X POST http://localhost:3011/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"stats"}}'
```

## Schema

Two tables with automatic migration from legacy schema:

```
recipes_data          recipes_vector
├── id (UUID PK)      ├── recipe_id (UUID PK FK → recipes_data.id)
├── slug              └── embedding (vector(1024))
├── url
├── title
├── description
├── prep_time_minutes
├── cook_time_minutes
├── difficulty
├── servings_count
├── servings_unit
├── courses (TEXT[])
├── food_types (TEXT[])
├── chef
├── ingredients (JSONB)
├── steps
└── search_vector (tsvector, GENERATED)
```

## Architecture

```
src/
├── main.rs          # Entrypoint, CLI dispatch, server init
├── config.rs        # Env var parsing, EmbeddingMode/Transport enums
├── db.rs            # PostgreSQL pool, SQL queries, migrations
├── embeddings.rs    # Ollama & OpenRouter embedding providers
├── tools.rs         # 7 MCP tool handlers
└── models.rs        # Recipe/Ingredient structs
```

## Tech stack

- **[rust-mcp-sdk](https://crates.io/crates/rust-mcp-sdk)** v0.9 — MCP server framework
- **PostgreSQL + pgvector** — vector storage with HNSW index
- **Ollama / OpenRouter** — embedding generation
- **sqlx** — async PostgreSQL driver with compile-time query checking
