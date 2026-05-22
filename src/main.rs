mod config;
mod db;
mod embeddings;
mod models;
mod tools;

use std::env;
use std::sync::Arc;
use std::time::Instant;

use rust_mcp_sdk::mcp_server::{
    HyperServerOptions, McpServerOptions, hyper_server, server_runtime,
};
use rust_mcp_sdk::schema::{
    Implementation, InitializeResult, ProtocolVersion, ServerCapabilities, ServerCapabilitiesTools,
};
use rust_mcp_sdk::{McpServer, StdioTransport, ToMcpServerHandler, TransportOptions};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

use config::{Config, EmbeddingMode, Transport};
use db::PgVectorDb;
use embeddings::Embedder;
use models::build_embed_text;
use tools::RecipesHandler;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("mcp_recipes=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = env::args().collect();
    debug!("CLI args: {:?}", args);

    if args.contains(&"--clear".to_string()) {
        debug!("Running --clear command");
        return cmd_clear().await;
    }

    if args.contains(&"--index".to_string()) {
        debug!("Running --index command");
        let config = Config::from_env()?;
        if matches!(config.embedding_mode, EmbeddingMode::Bm25) {
            return Err(anyhow::anyhow!("Indexing requires embeddings. Set EMBEDDING_MODE=ollama or EMBEDDING_MODE=openrouter"));
        }
        return cmd_index(&args).await;
    }

    let config = Config::from_env()?;
    debug!("Config loaded: transport={}, host={}, port={}, db_url={}, embedding_mode={}",
        config.transport, config.host, config.port,
        config.database_url.split('@').next().unwrap_or("unknown"),
        config.embedding_mode);

    info!("Transport: {}", config.transport);
    info!("Embedding mode: {}", config.embedding_mode);

    let db_start = Instant::now();
    let db = Arc::new(PgVectorDb::new(&config.database_url, config.embedding_dim).await?);
    debug!("Database initialized in {:.2?}", db_start.elapsed());

    let embedder = Arc::new(Embedder::new(config.clone()));

    if matches!(config.embedding_mode, EmbeddingMode::Ollama) {
        let warmup_start = Instant::now();
        if let Err(e) = embedder.warmup().await {
            tracing::warn!("Ollama warmup failed, continuing anyway: {}", e);
        } else {
            debug!("Ollama warmup completed in {:.2?}", warmup_start.elapsed());
        }
    } else {
        debug!("Warmup skipped for mode: {}", config.embedding_mode);
    }

    let handler = RecipesHandler::new(Arc::clone(&db), Arc::clone(&embedder))
        .to_mcp_server_handler();
    debug!("MCP handler created");

    let server_details = InitializeResult {
        server_info: Implementation {
            name: "mcp-recipes".into(),
            version: "0.1.0".into(),
            title: Some("MCP Recipes".into()),
            description: Some(
                "MCP server for searching and retrieving cooking recipes with hybrid BM25 + vector embeddings"
                    .into(),
            ),
            icons: vec![],
            website_url: None,
        },
        capabilities: ServerCapabilities {
            tools: Some(ServerCapabilitiesTools {
                list_changed: None,
            }),
            ..Default::default()
        },
        protocol_version: ProtocolVersion::V2025_11_25.into(),
        instructions: Some(
            "Use 'search_recipes' to find recipes with hybrid search. Use 'get_recipe_by_id' to read full recipe details. Use 'search_by_ingredients' to find recipes by ingredient. Use 'search_by_filters' for structured filtering by course, food type, chef, difficulty, or time. Use 'index_recipes' to load recipes from JSON into the database.".into(),
        ),
        meta: None,
    };

    match config.transport {
        Transport::Stdio => {
            info!("Starting MCP Recipes with stdio transport");
            let transport = StdioTransport::new(TransportOptions::default())
                .map_err(|e| anyhow::anyhow!("Transport error: {}", e))?;
            let options = McpServerOptions {
                server_details,
                transport,
                handler,
                task_store: None,
                client_task_store: None,
                message_observer: None,
            };
            let server = server_runtime::create_server(options);
            if let Err(e) = server.start().await {
                return Err(anyhow::anyhow!("Server error: {:?}", e));
            }
        }
        Transport::Http => {
            info!("MCP Recipes listening on http://{}:{}", config.host, config.port);

            let server = hyper_server::create_server(
                server_details,
                handler,
                HyperServerOptions {
                    host: config.host.clone(),
                    port: config.port,
                    event_store: None,
                    health_endpoint: None,
                    ..Default::default()
                },
            );
            if let Err(e) = server.start().await {
                return Err(anyhow::anyhow!("Server error: {:?}", e));
            }
        }
    }

    Ok(())
}

async fn cmd_clear() -> anyhow::Result<()> {
    let config = Config::from_env()?;
    debug!("Clear command: connecting to {}", config.database_url.split('@').nth(1).unwrap_or("unknown"));

    eprintln!("Connecting to database...");
    let db_start = Instant::now();
    let db = PgVectorDb::new(&config.database_url, config.embedding_dim).await?;
    debug!("Database connected in {:.2?}", db_start.elapsed());

    let (total, with_embeddings) = db.count().await?;
    eprintln!("Current recipes: {} total, {} with embeddings", total, with_embeddings);
    debug!("Count before clear: total={}, with_embeddings={}", total, with_embeddings);

    eprintln!("Deleting all recipes...");
    let clear_start = Instant::now();
    let deleted = db.clear_db().await?;
    debug!("Clear completed in {:.2?}, deleted {} records", clear_start.elapsed(), deleted);

    eprintln!("Deleted {} recipes from the database.", deleted);

    let (remaining, _) = db.count().await?;
    eprintln!("Remaining recipes: {}", remaining);
    debug!("Count after clear: remaining={}", remaining);

    Ok(())
}

async fn cmd_index(args: &[String]) -> anyhow::Result<()> {
    let config = Config::from_env()?;

    let path = args
        .iter()
        .find(|a| a.starts_with("--path="))
        .map(|a| a.trim_start_matches("--path="))
        .unwrap_or("recipes.json");

    debug!("Index command: path={}, db={}", path, config.database_url.split('@').nth(1).unwrap_or("unknown"));

    eprintln!("Connecting to database...");
    let db = PgVectorDb::new(&config.database_url, config.embedding_dim).await?;

    let embedder = Embedder::new(config.clone());
    if let Err(e) = embedder.warmup().await {
        eprintln!("Warning: Ollama warmup failed: {}", e);
    }

    eprintln!("Reading {}...", path);
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", path, e))?;

    let recipes: Vec<crate::models::Recipe> = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

    eprintln!("Found {} recipes. Indexing...", recipes.len());

    let index_start = Instant::now();
    let mut indexed = 0;
    let mut skipped = 0;
    let mut total_embed_time = std::time::Duration::ZERO;
    let mut total_upsert_time = std::time::Duration::ZERO;

    for (i, recipe) in recipes.iter().enumerate() {
        let embed_start = Instant::now();
        let text = build_embed_text(recipe);

        let embedding = match embedder.embed(&text).await {
            Ok(emb) => emb,
            Err(e) => {
                debug!("[{}/{}] '{}' - embedding FAILED: {}", i + 1, recipes.len(), recipe.title, e);
                skipped += 1;
                continue;
            }
        };
        total_embed_time += embed_start.elapsed();

        let upsert_start = Instant::now();
        let embedding_vec = pgvector::Vector::from(embedding);
        if let Err(e) = db.upsert_recipe(recipe, &embedding_vec).await {
            debug!("[{}/{}] '{}' - upsert FAILED: {}", i + 1, recipes.len(), recipe.title, e);
            skipped += 1;
            continue;
        }
        total_upsert_time += upsert_start.elapsed();
        indexed += 1;

        if (i + 1) % 10 == 0 || i == recipes.len() - 1 {
            let elapsed = index_start.elapsed();
            let eta_per = if i > 0 { elapsed / (i as u32 + 1) } else { std::time::Duration::ZERO };
            let eta = eta_per * (recipes.len() - i - 1) as u32;
            eprintln!("  [{}/{}] ({:.1}%) - indexed={}, skipped={}, ETA={:.2?}",
                i + 1, recipes.len(),
                ((i + 1) as f64 / recipes.len() as f64) * 100.0,
                indexed, skipped, eta);
        }
    }

    let total_elapsed = index_start.elapsed();
    eprintln!("\nIndexing complete in {:.2?}", total_elapsed);
    eprintln!("  Indexed: {}", indexed);
    eprintln!("  Skipped: {}", skipped);
    eprintln!("  Avg embed: {:.2?}", if indexed > 0 { total_embed_time / indexed as u32 } else { std::time::Duration::ZERO });
    eprintln!("  Avg upsert: {:.2?}", if indexed > 0 { total_upsert_time / indexed as u32 } else { std::time::Duration::ZERO });

    let (final_total, final_with_embeddings) = db.count().await.unwrap_or((0, 0));
    eprintln!("  DB total: {}, with embeddings: {}", final_total, final_with_embeddings);

    Ok(())
}