use std::env;

use tracing::debug;

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub embedding_mode: EmbeddingMode,
    pub ollama_base_url: String,
    pub ollama_model: String,
    pub openrouter_api_key: String,
    pub openrouter_model: String,
    pub embedding_dim: usize,
    pub transport: Transport,
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone)]
pub enum Transport {
    Stdio,
    Http,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transport::Stdio => write!(f, "stdio"),
            Transport::Http => write!(f, "http"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddingMode {
    Bm25,
    Ollama,
    OpenRouter,
}

impl std::fmt::Display for EmbeddingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingMode::Bm25 => write!(f, "bm25"),
            EmbeddingMode::Ollama => write!(f, "ollama"),
            EmbeddingMode::OpenRouter => write!(f, "openrouter"),
        }
    }
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        match dotenvy::dotenv() {
            Ok(path) => debug!("Loaded .env from: {:?}", path),
            Err(e) => debug!("No .env file loaded (or error): {}", e),
        }

        let transport_str = env::var("MCP_TRANSPORT")
            .unwrap_or_else(|_| "stdio".to_string())
            .to_lowercase();
        let transport = match transport_str.as_str() {
            "http" => Transport::Http,
            _ => Transport::Stdio,
        };
        debug!("MCP_TRANSPORT: {} (default: stdio)", transport_str);

        let host = env::var("MCP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        debug!("MCP_HOST: {} (default: 127.0.0.1)", host);

        let port = env::var("MCP_PORT")
            .unwrap_or_else(|_| "3003".to_string())
            .parse()
            .unwrap_or(3003);
        debug!("MCP_PORT: {} (default: 3003)", port);

        let pg_host = env::var("PGHOST").unwrap_or_else(|_| "pgvector.one.belcar.corp".to_string());
        debug!("PGHOST: {} (default: pgvector.one.belcar.corp)", pg_host);

        let pg_port = env::var("PGPORT")
            .unwrap_or_else(|_| "5432".to_string())
            .parse()
            .unwrap_or(5432);
        debug!("PGPORT: {} (default: 5432)", pg_port);

        let pg_database = env::var("PGDATABASE").unwrap_or_else(|_| "mcp".to_string());
        debug!("PGDATABASE: {} (default: mcp)", pg_database);

        let pg_user = env::var("PGUSER").unwrap_or_else(|_| "rag_admin".to_string());
        debug!("PGUSER: {} (default: rag_admin)", pg_user);

        let pg_password = env::var("PGPASSWORD").unwrap_or_else(|_| "rag_password".to_string());
        debug!("PGPASSWORD: [set] (default: rag_password)");

        let pg_sslmode = env::var("PGSSLMODE").unwrap_or_else(|_| "require".to_string());
        debug!("PGSSLMODE: {} (default: require)", pg_sslmode);

        let database_url = if pg_sslmode.is_empty() {
            format!(
                "postgres://{}:{}@{}:{}/{}",
                pg_user, pg_password, pg_host, pg_port, pg_database
            )
        } else {
            format!(
                "postgres://{}:{}@{}:{}/{}?sslmode={}",
                pg_user, pg_password, pg_host, pg_port, pg_database, pg_sslmode
            )
        };
        debug!("DATABASE_URL: postgres://{}@{}:{}/{}?sslmode={}",
            pg_user, pg_host, pg_port, pg_database, pg_sslmode);

        let embedding_mode_str = env::var("EMBEDDING_MODE")
            .unwrap_or_else(|_| "ollama".to_string())
            .to_lowercase();
        let embedding_mode = match embedding_mode_str.as_str() {
            "bm25" => EmbeddingMode::Bm25,
            "openrouter" => EmbeddingMode::OpenRouter,
            _ => EmbeddingMode::Ollama,
        };
        debug!("EMBEDDING_MODE: {} (default: ollama)", embedding_mode);

        let ollama_base_url = env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        debug!("OLLAMA_BASE_URL: {} (default: http://localhost:11434)", ollama_base_url);

        let ollama_model =
            env::var("OLLAMA_MODEL").unwrap_or_else(|_| "bge-m3:latest".to_string());
        debug!("OLLAMA_MODEL: {} (default: bge-m3:latest)", ollama_model);

        let openrouter_api_key = env::var("OPENROUTER_API_KEY").unwrap_or_default();
        debug!("OPENROUTER_API_KEY: {} (default: empty)", if openrouter_api_key.is_empty() { "[not set]" } else { "[set]" });

        let openrouter_model = env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "nomic-ai/nomic-embed-text-v1.5".to_string());
        debug!("OPENROUTER_MODEL: {} (default: nomic-ai/nomic-embed-text-v1.5)", openrouter_model);

        let embedding_dim = env::var("EMBEDDING_DIM")
            .unwrap_or_else(|_| "1024".to_string())
            .parse()
            .unwrap_or(1024);
        debug!("EMBEDDING_DIM: {} (default: 1024)", embedding_dim);

        Ok(Self {
            database_url,
            embedding_mode,
            ollama_base_url,
            ollama_model,
            openrouter_api_key,
            openrouter_model,
            embedding_dim,
            transport,
            host,
            port,
        })
    }
}
