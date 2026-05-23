use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::config::{Config, EmbeddingMode};

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct OpenRouterEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbedResponse {
    data: Vec<OpenRouterEmbeddingData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenRouterChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    type_: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChatResponse {
    choices: Vec<OpenRouterChatChoice>,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    format: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: ChatMessage,
}

pub struct Embedder {
    client: Client,
    config: Config,
}

impl Embedder {
    pub fn new(config: Config) -> Self {
        debug!("Embedder created: mode={}", config.embedding_mode);
        match &config.embedding_mode {
            EmbeddingMode::Bm25 => debug!("BM25-only mode - no embedding provider configured"),
            EmbeddingMode::Ollama => debug!("Ollama: url={}, model={}", config.ollama_base_url, config.ollama_model),
            EmbeddingMode::OpenRouter => debug!("OpenRouter: model={}", config.openrouter_model),
        }
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn embedding_mode(&self) -> &EmbeddingMode {
        &self.config.embedding_mode
    }

    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match &self.config.embedding_mode {
            EmbeddingMode::Bm25 => {
                anyhow::bail!("Embeddings not available in BM25-only mode")
            }
            EmbeddingMode::Ollama => self.embed_ollama(text).await,
            EmbeddingMode::OpenRouter => self.embed_openrouter(text).await,
        }
    }

    pub async fn generate(&self, system_prompt: &str, user_prompt: &str) -> anyhow::Result<String> {
        match &self.config.embedding_mode {
            EmbeddingMode::Bm25 => {
                anyhow::bail!("Recipe generation requires an LLM. Set EMBEDDING_MODE=ollama or EMBEDDING_MODE=openrouter")
            }
            EmbeddingMode::Ollama => self.generate_ollama(system_prompt, user_prompt).await,
            EmbeddingMode::OpenRouter => self.generate_openrouter(system_prompt, user_prompt).await,
        }
    }

    pub async fn warmup(&self) -> anyhow::Result<()> {
        match &self.config.embedding_mode {
            EmbeddingMode::Bm25 => {
                debug!("Warmup skipped in BM25-only mode");
                Ok(())
            }
            EmbeddingMode::Ollama => self.warmup_ollama().await,
            EmbeddingMode::OpenRouter => {
                debug!("Warmup skipped for OpenRouter (remote API)");
                Ok(())
            }
        }
    }

    async fn warmup_ollama(&self) -> anyhow::Result<()> {
        info!("Warming up Ollama embedding model...");
        let warmup_start = Instant::now();
        let url = format!("{}/api/embed", self.config.ollama_base_url);
        let request = OllamaEmbedRequest {
            model: self.config.ollama_model.clone(),
            input: "warmup".to_string(),
        };
        debug!("Warmup request: POST {} (model: {})", url, self.config.ollama_model);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send warmup request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama warmup error ({}): {}", status, body);
        }

        info!("Ollama model warmed up successfully in {:.2?}", warmup_start.elapsed());
        Ok(())
    }

    async fn embed_ollama(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.config.ollama_base_url);
        let text_len = text.chars().count();
        debug!("Embedding request (Ollama): model={}, text_length={} chars", self.config.ollama_model, text_len);

        let request = OllamaEmbedRequest {
            model: self.config.ollama_model.clone(),
            input: text.to_string(),
        };

        let do_request = || async {
            let req_start = Instant::now();
            let response = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await
                .context("Failed to send request to Ollama")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Ollama error ({}): {}", status, body);
            }

            let embed_response: OllamaEmbedResponse = response
                .json()
                .await
                .context("Failed to parse Ollama response")?;

            if embed_response.embeddings.is_empty() || embed_response.embeddings[0].is_empty() {
                anyhow::bail!(
                    "Ollama returned empty embeddings for text ({} chars)",
                    text.chars().count()
                );
            }

            let elapsed = req_start.elapsed();
            let dim = embed_response.embeddings[0].len();
            debug!("Ollama response: dim={}, time={:.2?}", dim, elapsed);

            Ok(embed_response.embeddings[0].clone())
        };

        match do_request().await {
            Ok(embedding) => {
                debug!("Embedding generated successfully (dim: {})", embedding.len());
                Ok(embedding)
            }
            Err(e) => {
                warn!("Ollama request failed, attempting warmup and retry: {}", e);
                if self.warmup_ollama().await.is_ok() {
                    match do_request().await {
                        Ok(embedding) => {
                            debug!("Retry succeeded after warmup (dim: {})", embedding.len());
                            return Ok(embedding);
                        }
                        Err(retry_err) => {
                            warn!("Ollama retry after warmup also failed: {}", retry_err);
                        }
                    }
                }
                Err(e)
            }
        }
    }

    async fn embed_openrouter(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let url = "https://openrouter.ai/api/v1/embeddings";
        let text_len = text.chars().count();
        debug!("Embedding request (OpenRouter): model={}, text_length={} chars", self.config.openrouter_model, text_len);

        let request = OpenRouterEmbedRequest {
            model: self.config.openrouter_model.clone(),
            input: vec![text.to_string()],
        };

        let req_start = Instant::now();
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.openrouter_api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenRouter")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter error ({}): {}", status, body);
        }

        let embed_response: OpenRouterEmbedResponse = response
            .json()
            .await
            .context("Failed to parse OpenRouter response")?;

        if embed_response.data.is_empty() || embed_response.data[0].embedding.is_empty() {
            anyhow::bail!(
                "OpenRouter returned empty embeddings for text ({} chars)",
                text.chars().count()
            );
        }

        let elapsed = req_start.elapsed();
        let embedding = embed_response.data[0].embedding.clone();
        let dim = embedding.len();
        debug!("OpenRouter response: dim={}, time={:.2?}", dim, elapsed);

        Ok(embedding)
    }

    async fn generate_openrouter(&self, system_prompt: &str, user_prompt: &str) -> anyhow::Result<String> {
        let url = "https://openrouter.ai/api/v1/chat/completions";
        debug!("Generate request (OpenRouter): model={}", self.config.openrouter_chat_model);

        let request = OpenRouterChatRequest {
            model: self.config.openrouter_chat_model.clone(),
            messages: vec![
                ChatMessage { role: "system".into(), content: system_prompt.to_string() },
                ChatMessage { role: "user".into(), content: user_prompt.to_string() },
            ],
            response_format: Some(ResponseFormat { type_: "json_object".into() }),
        };

        let req_start = Instant::now();
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.openrouter_api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send generate request to OpenRouter")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter generate error ({}): {}", status, body);
        }

        let chat_response: OpenRouterChatResponse = response
            .json()
            .await
            .context("Failed to parse OpenRouter chat response")?;

        let elapsed = req_start.elapsed();
        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        debug!("OpenRouter generate response in {:.2?} ({} chars)", elapsed, content.chars().count());

        Ok(content)
    }

    async fn generate_ollama(&self, system_prompt: &str, user_prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/api/chat", self.config.ollama_base_url);
        debug!("Generate request (Ollama): model={}", self.config.ollama_chat_model);

        let request = OllamaChatRequest {
            model: self.config.ollama_chat_model.clone(),
            messages: vec![
                ChatMessage { role: "system".into(), content: system_prompt.to_string() },
                ChatMessage { role: "user".into(), content: user_prompt.to_string() },
            ],
            stream: false,
            format: "json".into(),
        };

        let req_start = Instant::now();
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send generate request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama generate error ({}): {}", status, body);
        }

        let chat_response: OllamaChatResponse = response
            .json()
            .await
            .context("Failed to parse Ollama chat response")?;

        let elapsed = req_start.elapsed();
        let content = chat_response.message.content;
        debug!("Ollama generate response in {:.2?} ({} chars)", elapsed, content.chars().count());

        Ok(content)
    }
}
