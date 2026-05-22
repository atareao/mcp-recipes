use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use rust_mcp_sdk::{
    McpServer,
    mcp_server::ServerHandler,
    schema::{
        CallToolError, CallToolRequestParams, CallToolResult, ListToolsResult,
        PaginatedRequestParams, RpcError, Tool, ToolInputSchema,
    },
};
use serde_json::{Map, Value};
use thiserror::Error;
use tracing::{debug, info};

use crate::config::EmbeddingMode;
use crate::db::PgVectorDb;
use crate::embeddings::Embedder;

#[derive(Debug, Error)]
#[error("{0}")]
struct ToolError(String);

fn tool_err(msg: impl Into<String>) -> CallToolError {
    CallToolError::new(ToolError(msg.into()))
}

fn make_input_schema(
    properties: BTreeMap<String, Map<String, Value>>,
    required: Vec<String>,
) -> ToolInputSchema {
    ToolInputSchema::new(required, Some(properties), None)
}

fn empty_input_schema() -> ToolInputSchema {
    ToolInputSchema::new(vec![], None, None)
}

fn string_prop(desc: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".to_string(), Value::String("string".to_string()));
    m.insert("description".to_string(), Value::String(desc.to_string()));
    m
}

fn integer_prop(desc: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".to_string(), Value::String("integer".to_string()));
    m.insert("description".to_string(), Value::String(desc.to_string()));
    m
}

fn number_prop(desc: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".to_string(), Value::String("number".to_string()));
    m.insert("description".to_string(), Value::String(desc.to_string()));
    m
}

fn array_prop(desc: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".to_string(), Value::String("array".to_string()));
    m.insert("description".to_string(), Value::String(desc.to_string()));
    let mut items = Map::new();
    items.insert("type".to_string(), Value::String("string".to_string()));
    m.insert("items".to_string(), Value::Object(items));
    m
}

pub struct RecipesHandler {
    db: Arc<PgVectorDb>,
    embedder: Arc<Embedder>,
}

impl RecipesHandler {
    pub fn new(db: Arc<PgVectorDb>, embedder: Arc<Embedder>) -> Self {
        Self { db, embedder }
    }
}

#[async_trait]
impl ServerHandler for RecipesHandler {
    async fn handle_list_tools_request(
        &self,
        _request: Option<PaginatedRequestParams>,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        debug!("listing tools");

        Ok(ListToolsResult {
            tools: vec![
                Tool {
                    name: "search_recipes".into(),
                    description: Some("Hybrid search across recipes using BM25 full-text and vector similarity. Combines both with weighted scoring (70% vector, 30% text). Returns ranked recipe results with combined scores.".into()),
                    input_schema: make_input_schema(
                        BTreeMap::from([
                            ("query".into(), string_prop("Free-text search query (e.g., 'chicken with rice', 'quick vegetarian pasta')")),
                            ("limit".into(), integer_prop("Maximum number of results to return (default: 10)")),
                            ("offset".into(), integer_prop("Skip first N results for pagination (default: 0)")),
                            ("min_similarity".into(), number_prop("Minimum combined score threshold 0.0-1.0 (default: 0.3)")),
                        ]),
                        vec!["query".into()],
                    ),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "get_recipe_by_id".into(),
                    description: Some("Get the full details of a single recipe by its UUID. Returns complete recipe including ingredients, steps, times, and metadata.".into()),
                    input_schema: make_input_schema(
                        BTreeMap::from([
                            ("id".into(), string_prop("The UUID of the recipe")),
                        ]),
                        vec!["id".into()],
                    ),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "search_by_ingredients".into(),
                    description: Some("Find recipes that contain one or more of the specified ingredients. Uses JSONB containment query on the ingredients array (OR logic - matches any ingredient).".into()),
                    input_schema: make_input_schema(
                        BTreeMap::from([
                            ("ingredients".into(), array_prop("List of ingredient names to search for (OR logic - matches any)")),
                            ("limit".into(), integer_prop("Maximum number of results (default: 10)")),
                        ]),
                        vec!["ingredients".into()],
                    ),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "search_by_filters".into(),
                    description: Some("Search recipes using structured filters: course type, food type, chef, difficulty, and maximum total time. All filters are optional and combined with AND logic.".into()),
                    input_schema: make_input_schema(
                        BTreeMap::from([
                            ("course".into(), string_prop("Course type (e.g., 'Aperitivos', 'Primeros', 'Segundos', 'Postres')")),
                            ("food_type".into(), string_prop("Food type (e.g., 'Carne y Aves', 'Fruta y Verdura', 'Arroz, Legumbres y Cereales')")),
                            ("chef".into(), string_prop("Chef name (partial match, case-insensitive)")),
                            ("max_difficulty".into(), integer_prop("Maximum difficulty level (1=easy, higher=harder)")),
                            ("max_total_time".into(), integer_prop("Maximum total time in minutes (prep + cook)")),
                            ("limit".into(), integer_prop("Maximum number of results (default: 20)")),
                        ]),
                        vec![],
                    ),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "stats".into(),
                    description: Some("Show database statistics including total recipes indexed and recipes with embeddings.".into()),
                    input_schema: empty_input_schema(),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "index_recipes".into(),
                    description: Some("Index all recipes from the recipes.json file. Reads the JSON file, generates embeddings via Ollama, and stores everything in the database. Use --path argument to specify a different file path.".into()),
                    input_schema: make_input_schema(
                        BTreeMap::from([
                            ("path".into(), string_prop("Path to the recipes JSON file (default: recipes.json)")),
                        ]),
                        vec![],
                    ),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
                Tool {
                    name: "clear_db".into(),
                    description: Some("Delete ALL indexed recipes from the database. This operation is irreversible. Use this to start fresh or free up database space.".into()),
                    input_schema: empty_input_schema(),
                    annotations: None,
                    execution: None,
                    icons: vec![],
                    meta: None,
                    output_schema: None,
                    title: None,
                },
            ],
            meta: None,
            next_cursor: None,
        })
    }

    async fn handle_call_tool_request(
        &self,
        params: CallToolRequestParams,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        let tool_name = params.name.clone();
        debug!("tool call: {}", tool_name);
        let tool_start = Instant::now();

        let args = match params.arguments {
            Some(map) => serde_json::Value::Object(map),
            None => serde_json::Value::Null,
        };

        let result = match tool_name.as_str() {
            "search_recipes" => self.search_recipes(args).await,
            "get_recipe_by_id" => self.get_recipe_by_id(args).await,
            "search_by_ingredients" => self.search_by_ingredients(args).await,
            "search_by_filters" => self.search_by_filters(args).await,
            "stats" => self.stats().await,
            "index_recipes" => self.index_recipes(args).await,
            "clear_db" => self.clear_db().await,
            _ => Err(CallToolError::unknown_tool(tool_name.clone())),
        };

        debug!("tool '{}' completed in {:.2?}", tool_name, tool_start.elapsed());
        result
    }
}

impl RecipesHandler {
    async fn search_recipes(&self, args: serde_json::Value) -> Result<CallToolResult, CallToolError> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| tool_err("Missing 'query' parameter"))?;

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let min_similarity = args
            .get("min_similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3);

        debug!("search_recipes: query='{}', limit={}, offset={}, min_similarity={}", query, limit, offset, min_similarity);

        let results = match self.embedder.embedding_mode() {
            EmbeddingMode::Bm25 => {
                debug!("search_recipes: BM25-only mode, skipping embedding");
                self.db
                    .search_bm25_only(query, limit, offset, min_similarity)
                    .await
                    .map_err(|e| tool_err(format!("Search error: {}", e)))?
            }
            EmbeddingMode::Ollama | EmbeddingMode::OpenRouter => {
                let embed_start = Instant::now();
                let query_embedding = self
                    .embedder
                    .embed(query)
                    .await
                    .map_err(|e| tool_err(format!("Embedding error: {}", e)))?;
                debug!("search_recipes: embedding generated in {:.2?}", embed_start.elapsed());

                let query_embedding = pgvector::Vector::from(query_embedding);

                self.db
                    .search(&query_embedding, query, limit, offset, min_similarity)
                    .await
                    .map_err(|e| tool_err(format!("Search error: {}", e)))?
            }
        };

        if results.items.is_empty() {
            return Ok(CallToolResult::text_content(vec![
                format!(
                    "No results found for query: \"{}\" (min_similarity: {:.2})",
                    query, min_similarity
                )
                .into(),
            ]));
        }

        let mut output = String::new();
        output.push_str(&format!("# Search Results for: \"{}\"\n\n", query));
        output.push_str(&format!(
            "Found {} of {} total results (min_similarity: {:.2})\n\n---\n\n",
            results.items.len(),
            results.total,
            min_similarity
        ));

        for (i, r) in results.items.iter().enumerate() {
            let total_time = match (r.prep_time_minutes, r.cook_time_minutes) {
                (Some(p), Some(c)) => format!("{} min prep, {} min cook", p, c),
                (Some(p), None) => format!("{} min prep", p),
                (None, Some(c)) => format!("{} min cook", c),
                (None, None) => "No time info".to_string(),
            };

            let courses = if r.courses.is_empty() {
                String::new()
            } else {
                format!(" | Courses: {}", r.courses.join(", "))
            };

            let food_types = if r.food_types.is_empty() {
                String::new()
            } else {
                format!(" | Types: {}", r.food_types.join(", "))
            };

            let chef = if r.chef.is_empty() {
                String::new()
            } else {
                format!(" | Chef: {}", r.chef)
            };

            let ingredients_preview: Vec<_> = r
                .ingredients
                .iter()
                .take(5)
                .map(|ing| {
                    if let Some(q) = ing.quantity {
                        if !ing.unit.is_empty() {
                            format!("{} {} {}", q, ing.unit, ing.name)
                        } else {
                            ing.name.clone()
                        }
                    } else {
                        ing.name.clone()
                    }
                })
                .collect();

            let ingredients_str = if r.ingredients.len() > 5 {
                format!("{}, ... (+{} more)", ingredients_preview.join(", "), r.ingredients.len() - 5)
            } else {
                ingredients_preview.join(", ")
            };

            output.push_str(&format!(
                "## {} - Score: {:.3}\n\n**Title**: {}\n**URL**: {}\n**Time**: {}{}{}{}\n**Difficulty**: {} | **Servings**: {} {}\n**Ingredients**: {}\n**Description**: {}\n\n---\n\n",
                i + 1,
                r.combined_score,
                r.title,
                r.url,
                total_time,
                courses,
                food_types,
                chef,
                r.difficulty.map(|d| d.to_string()).unwrap_or_else(|| "N/A".to_string()),
                r.servings_count.map(|s| s.to_string()).unwrap_or_else(|| "N/A".to_string()),
                r.servings_unit,
                ingredients_str,
                r.description,
            ));
        }

        let next_offset = results.offset + results.limit;
        if next_offset < results.total as usize {
            output.push_str(&format!(
                "**Showing {}-{} of {} results. Use `offset={}` for next page.**\n",
                results.offset + 1,
                results.offset + results.items.len(),
                results.total,
                next_offset
            ));
        }

        Ok(CallToolResult::text_content(vec![output.into()]))
    }

    async fn get_recipe_by_id(&self, args: serde_json::Value) -> Result<CallToolResult, CallToolError> {
        let id_str = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| tool_err("Missing 'id' parameter"))?;

        let id = uuid::Uuid::parse_str(id_str)
            .map_err(|e| tool_err(format!("Invalid UUID '{}': {}", id_str, e)))?;

        debug!("get_recipe_by_id: id={}", id);

        match self.db.get_recipe_by_id(&id).await.map_err(|e| tool_err(format!("Error reading recipe: {}", e)))? {
            Some(recipe) => {
                let total_time = match (recipe.prep_time_minutes, recipe.cook_time_minutes) {
                    (Some(p), Some(c)) => format!("{} min", p + c),
                    (Some(p), None) => format!("{} min (prep only)", p),
                    (None, Some(c)) => format!("{} min (cook only)", c),
                    (None, None) => "No time info".to_string(),
                };

                let mut output = String::new();
                output.push_str(&format!("# {}\n\n", recipe.title));
                output.push_str(&format!("**URL**: {}\n\n", recipe.url));
                output.push_str(&format!("**Description**: {}\n\n", recipe.description));

                output.push_str("## Details\n\n");
                output.push_str(&format!("- **Prep time**: {} min\n", recipe.prep_time_minutes.map(|t| t.to_string()).unwrap_or_else(|| "N/A".to_string())));
                output.push_str(&format!("- **Cook time**: {} min\n", recipe.cook_time_minutes.map(|t| t.to_string()).unwrap_or_else(|| "N/A".to_string())));
                output.push_str(&format!("- **Total time**: {}\n", total_time));
                output.push_str(&format!("- **Difficulty**: {}\n", recipe.difficulty.map(|d| d.to_string()).unwrap_or_else(|| "N/A".to_string())));
                output.push_str(&format!("- **Servings**: {} {}\n", recipe.servings_count.map(|s| s.to_string()).unwrap_or_else(|| "N/A".to_string()), recipe.servings_unit));

                if !recipe.courses.is_empty() {
                    output.push_str(&format!("- **Courses**: {}\n", recipe.courses.join(", ")));
                }
                if !recipe.food_types.is_empty() {
                    output.push_str(&format!("- **Food types**: {}\n", recipe.food_types.join(", ")));
                }
                if !recipe.chef.is_empty() {
                    output.push_str(&format!("- **Chef**: {}\n", recipe.chef));
                }

                output.push_str(&format!("\n## Ingredients ({})\n\n", recipe.ingredients.len()));
                for ing in &recipe.ingredients {
                    if let Some(q) = ing.quantity {
                        if !ing.unit.is_empty() {
                            output.push_str(&format!("- {} {} {}\n", q, ing.unit, ing.name));
                        } else {
                            output.push_str(&format!("- {}\n", ing.name));
                        }
                    } else {
                        output.push_str(&format!("- {}\n", ing.name));
                    }
                }

                output.push_str(&format!("\n## Steps\n\n{}\n", recipe.steps));

                Ok(CallToolResult::text_content(vec![output.into()]))
            }
            None => Err(tool_err(format!("Recipe not found: {}", id))),
        }
    }

    async fn search_by_ingredients(&self, args: serde_json::Value) -> Result<CallToolResult, CallToolError> {
        let ingredients = args
            .get("ingredients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| tool_err("Missing 'ingredients' parameter (must be an array)"))?;

        let ingredient_names: Vec<String> = ingredients
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if ingredient_names.is_empty() {
            return Err(tool_err("'ingredients' array must not be empty"));
        }

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        debug!("search_by_ingredients: {:?}, limit={}", ingredient_names, limit);

        let recipes = self
            .db
            .search_by_ingredients(&ingredient_names, limit)
            .await
            .map_err(|e| tool_err(format!("Search error: {}", e)))?;

        if recipes.is_empty() {
            return Ok(CallToolResult::text_content(vec![
                format!(
                    "No recipes found with ingredients: {}",
                    ingredient_names.join(", ")
                )
                .into(),
            ]));
        }

        let mut output = String::new();
        output.push_str(&format!(
            "# Recipes with Ingredients: {}\n\n",
            ingredient_names.join(", ")
        ));
        output.push_str(&format!("Found {} recipes\n\n---\n\n", recipes.len()));

        for (_i, r) in recipes.iter().enumerate() {
            let matched_ingredients: Vec<_> = r
                .ingredients
                .iter()
                .filter(|ing| {
                    ingredient_names
                        .iter()
                        .any(|name| ing.name.to_lowercase().contains(&name.to_lowercase()))
                })
                .map(|ing| ing.name.clone())
                .collect();

            output.push_str(&format!(
                "## {}\n\n**URL**: {}\n**Matched**: {}\n**Description**: {}\n\n---\n\n",
                r.title,
                r.url,
                matched_ingredients.join(", "),
                r.description,
            ));
        }

        Ok(CallToolResult::text_content(vec![output.into()]))
    }

    async fn search_by_filters(&self, args: serde_json::Value) -> Result<CallToolResult, CallToolError> {
        let course = args.get("course").and_then(|v| v.as_str());
        let food_type = args.get("food_type").and_then(|v| v.as_str());
        let chef = args.get("chef").and_then(|v| v.as_str());
        let max_difficulty = args.get("max_difficulty").and_then(|v| v.as_i64()).map(|v| v as i32);
        let max_total_time = args.get("max_total_time").and_then(|v| v.as_i64()).map(|v| v as i32);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        debug!("search_by_filters: course={:?}, food_type={:?}, chef={:?}, max_difficulty={:?}, max_total_time={:?}, limit={}",
            course, food_type, chef, max_difficulty, max_total_time, limit);

        let recipes = self
            .db
            .search_by_filters(course, food_type, chef, max_difficulty, max_total_time, limit)
            .await
            .map_err(|e| tool_err(format!("Search error: {}", e)))?;

        if recipes.is_empty() {
            let mut filters = Vec::new();
            if let Some(c) = course {
                filters.push(format!("course={}", c));
            }
            if let Some(ft) = food_type {
                filters.push(format!("food_type={}", ft));
            }
            if let Some(ch) = chef {
                filters.push(format!("chef={}", ch));
            }
            if let Some(md) = max_difficulty {
                filters.push(format!("max_difficulty={}", md));
            }
            if let Some(mt) = max_total_time {
                filters.push(format!("max_total_time={}", mt));
            }

            return Ok(CallToolResult::text_content(vec![
                format!("No recipes found with filters: {}", filters.join(", ")).into(),
            ]));
        }

        let mut output = String::new();
        output.push_str("# Filtered Recipes\n\n");

        let mut filter_parts = Vec::new();
        if let Some(c) = course {
            filter_parts.push(format!("Course: {}", c));
        }
        if let Some(ft) = food_type {
            filter_parts.push(format!("Food Type: {}", ft));
        }
        if let Some(ch) = chef {
            filter_parts.push(format!("Chef: {}", ch));
        }
        if let Some(md) = max_difficulty {
            filter_parts.push(format!("Max Difficulty: {}", md));
        }
        if let Some(mt) = max_total_time {
            filter_parts.push(format!("Max Total Time: {} min", mt));
        }

        if !filter_parts.is_empty() {
            output.push_str(&format!("**Filters**: {}\n\n", filter_parts.join(" | ")));
        }

        output.push_str(&format!("Found {} recipes\n\n---\n\n", recipes.len()));

        for (_i, r) in recipes.iter().enumerate() {
            let total_time = match (r.prep_time_minutes, r.cook_time_minutes) {
                (Some(p), Some(c)) => format!("{} min", p + c),
                (Some(p), None) => format!("{} min (prep)", p),
                (None, Some(c)) => format!("{} min (cook)", c),
                (None, None) => "N/A".to_string(),
            };

            let courses = if r.courses.is_empty() {
                String::new()
            } else {
                format!(" | {}", r.courses.join(", "))
            };

            output.push_str(&format!(
                "## {}\n\n**URL**: {} | **Time**: {} | **Difficulty**: {}{}\n**Description**: {}\n\n---\n\n",
                r.title,
                r.url,
                total_time,
                r.difficulty.map(|d| d.to_string()).unwrap_or_else(|| "N/A".to_string()),
                courses,
                r.description,
            ));
        }

        Ok(CallToolResult::text_content(vec![output.into()]))
    }

    async fn stats(&self) -> Result<CallToolResult, CallToolError> {
        debug!("stats: querying database");
        let (total_recipes, with_embeddings) = self
            .db
            .count()
            .await
            .map_err(|e| tool_err(format!("Error getting stats: {}", e)))?;

        debug!("stats: total={}, with_embeddings={}", total_recipes, with_embeddings);

        Ok(CallToolResult::text_content(vec![
            format!(
                "# Database Statistics\n\n- **Total recipes indexed**: {}\n- **Recipes with embeddings**: {}\n- **Recipes without embeddings**: {}",
                total_recipes,
                with_embeddings,
                total_recipes - with_embeddings,
            )
            .into(),
        ]))
    }

    async fn index_recipes(&self, args: serde_json::Value) -> Result<CallToolResult, CallToolError> {
        if matches!(self.embedder.embedding_mode(), EmbeddingMode::Bm25) {
            return Err(tool_err("Indexing requires embeddings. Set EMBEDDING_MODE=ollama or EMBEDDING_MODE=openrouter"));
        }

        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("recipes.json");

        info!("index_recipes: starting, path={}", path);
        let index_start = Instant::now();

        let read_start = Instant::now();
        let content = std::fs::read_to_string(path)
            .map_err(|e| tool_err(format!("Failed to read {}: {}", path, e)))?;
        debug!("index_recipes: file read in {:.2?}, size={} bytes", read_start.elapsed(), content.len());

        let parse_start = Instant::now();
        let recipes: Vec<crate::models::Recipe> = serde_json::from_str(&content)
            .map_err(|e| tool_err(format!("Failed to parse JSON: {}", e)))?;
        debug!("index_recipes: parsed {} recipes in {:.2?}", recipes.len(), parse_start.elapsed());

        let (existing_total, existing_with_embeddings) = self.db.count().await
            .unwrap_or((0, 0));
        debug!("index_recipes: existing recipes in DB: total={}, with_embeddings={}", existing_total, existing_with_embeddings);

        let mut indexed = 0;
        let mut skipped = 0;
        let mut updated = 0;
        let mut total_embed_time = std::time::Duration::ZERO;
        let mut total_upsert_time = std::time::Duration::ZERO;

        for (i, recipe) in recipes.iter().enumerate() {
            let recipe_start = Instant::now();

            let embed_start = Instant::now();
            let text = build_embed_text(recipe);
            debug!("index_recipes: [{}/{}] '{}' - text_length={} chars", i + 1, recipes.len(), recipe.title, text.chars().count());

            let embedding = match self.embedder.embed(&text).await {
                Ok(emb) => emb,
                Err(e) => {
                    debug!("index_recipes: [{}/{}] '{}' - embedding FAILED: {}", i + 1, recipes.len(), recipe.title, e);
                    skipped += 1;
                    continue;
                }
            };
            total_embed_time += embed_start.elapsed();
            debug!("index_recipes: [{}/{}] '{}' - embedding OK in {:.2?} (dim: {})", i + 1, recipes.len(), recipe.title, embed_start.elapsed(), embedding.len());

            let embedding_vec = pgvector::Vector::from(embedding);

            let upsert_start = Instant::now();
            match self.db.upsert_recipe(recipe, &embedding_vec).await {
                Ok(_) => {
                    total_upsert_time += upsert_start.elapsed();
                    debug!("index_recipes: [{}/{}] '{}' - upserted in {:.2?}", i + 1, recipes.len(), recipe.title, upsert_start.elapsed());
                    indexed += 1;
                    updated += 1;
                }
                Err(e) => {
                    debug!("index_recipes: [{}/{}] '{}' - upsert FAILED: {}", i + 1, recipes.len(), recipe.title, e);
                    skipped += 1;
                }
            }

            let recipe_elapsed = recipe_start.elapsed();
            if (i + 1) % 10 == 0 || i == recipes.len() - 1 {
                let avg_embed = if indexed > 0 { total_embed_time / indexed as u32 } else { std::time::Duration::ZERO };
                let avg_upsert = if indexed > 0 { total_upsert_time / indexed as u32 } else { std::time::Duration::ZERO };
                let eta_per_recipe = if i > 0 { index_start.elapsed() / (i as u32 + 1) } else { std::time::Duration::ZERO };
                let remaining = (recipes.len() - i - 1) as u32;
                let eta = eta_per_recipe * remaining;
                info!("index_recipes: progress {}/{} ({:.1}%) - indexed={}, skipped={}, updated={} - avg_embed={:.2?}, avg_upsert={:.2?}, last_recipe={:.2?}, ETA={:.2?}",
                    i + 1, recipes.len(),
                    ((i + 1) as f64 / recipes.len() as f64) * 100.0,
                    indexed, skipped, updated,
                    avg_embed, avg_upsert, recipe_elapsed, eta);
            }
        }

        let total_elapsed = index_start.elapsed();
        info!("index_recipes: COMPLETE in {:.2?} - total={}, indexed={}, skipped={}, updated={}, embed_time={:.2?}, upsert_time={:.2?}",
            total_elapsed, recipes.len(), indexed, skipped, updated, total_embed_time, total_upsert_time);

        let (final_total, final_with_embeddings) = self.db.count().await
            .unwrap_or((0, 0));
        debug!("index_recipes: final DB state: total={}, with_embeddings={}", final_total, final_with_embeddings);

        Ok(CallToolResult::text_content(vec![
            format!(
                "# Indexing Complete\n\n- **Total recipes in file**: {}\n- **Successfully indexed**: {}\n- **Updated (upsert)**: {}\n- **Skipped (errors)**: {}\n- **Total time**: {:.2?}\n- **Avg embed time**: {:.2?}\n- **Avg upsert time**: {:.2?}",
                recipes.len(),
                indexed,
                updated,
                skipped,
                total_elapsed,
                if indexed > 0 { total_embed_time / indexed as u32 } else { std::time::Duration::ZERO },
                if indexed > 0 { total_upsert_time / indexed as u32 } else { std::time::Duration::ZERO },
            )
            .into(),
        ]))
    }

    async fn clear_db(&self) -> Result<CallToolResult, CallToolError> {
        debug!("clear_db: starting");
        let (before, _) = self.db.count().await.unwrap_or((0, 0));
        debug!("clear_db: {} recipes before deletion", before);

        let deleted = self
            .db
            .clear_db()
            .await
            .map_err(|e| tool_err(format!("Clear DB error: {}", e)))?;

        debug!("clear_db: {} recipes deleted", deleted);

        Ok(CallToolResult::text_content(vec![
            format!(
                "# Database Cleared\n\nDeleted **{}** recipes from the database.",
                deleted
            )
            .into(),
        ]))
    }
}

fn build_embed_text(recipe: &crate::models::Recipe) -> String {
    let mut parts = Vec::new();

    parts.push(format!("Title: {}", recipe.title));

    if !recipe.description.is_empty() {
        parts.push(format!("Description: {}", recipe.description));
    }

    if !recipe.courses.is_empty() {
        parts.push(format!("Courses: {}", recipe.courses.join(", ")));
    }

    if !recipe.food_types.is_empty() {
        parts.push(format!("Food types: {}", recipe.food_types.join(", ")));
    }

    if !recipe.chef.is_empty() {
        parts.push(format!("Chef: {}", recipe.chef));
    }

    if !recipe.ingredients.is_empty() {
        let ingredients_str = recipe
            .ingredients
            .iter()
            .map(|ing| {
                if ing.quantity.is_some() && !ing.unit.is_empty() {
                    format!(
                        "{} {} {}",
                        ing.quantity.unwrap(),
                        ing.unit,
                        ing.name
                    )
                } else {
                    ing.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("Ingredients: {}", ingredients_str));
    }

    if !recipe.steps.is_empty() {
        parts.push(format!("Steps: {}", recipe.steps));
    }

    parts.join("\n")
}
