use anyhow::Context;
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

use crate::models::Recipe;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub slug: String,
    pub url: String,
    pub title: String,
    pub description: String,
    pub prep_time_minutes: Option<i32>,
    pub cook_time_minutes: Option<i32>,
    pub difficulty: Option<i32>,
    pub servings_count: Option<i32>,
    pub servings_unit: String,
    pub courses: Vec<String>,
    pub food_types: Vec<String>,
    pub chef: String,
    pub ingredients: Vec<crate::models::Ingredient>,
    pub steps: String,
    pub similarity: f64,
    pub text_rank: f64,
    pub combined_score: f64,
}

#[derive(Debug)]
pub struct PaginatedResult<T> {
    pub items: Vec<T>,
    pub total: i64,
    pub offset: usize,
    pub limit: usize,
}

pub struct PgVectorDb {
    pool: PgPool,
    embedding_dim: usize,
}

impl PgVectorDb {
    pub async fn new(database_url: &str, embedding_dim: usize) -> anyhow::Result<Self> {
        info!("Connecting to PostgreSQL at {}", database_url);
        let conn_start = Instant::now();

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL")?;

        debug!("PostgreSQL connection established in {:.2?}", conn_start.elapsed());

        let db = Self {
            pool,
            embedding_dim,
        };

        db.run_migrations().await?;

        Ok(db)
    }

    async fn run_migrations(&self) -> anyhow::Result<()> {
        info!("Running database migrations");
        let mig_start = Instant::now();

        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector;")
            .execute(&self.pool)
            .await
            .context("Failed to create vector extension")?;
        debug!("Migration: vector extension created");

        let create_recipes_data_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS recipes_data (
                id UUID PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                prep_time_minutes INTEGER,
                cook_time_minutes INTEGER,
                difficulty INTEGER,
                servings_count INTEGER,
                servings_unit TEXT NOT NULL DEFAULT '',
                courses TEXT[] NOT NULL DEFAULT '{{}}',
                food_types TEXT[] NOT NULL DEFAULT '{{}}',
                chef TEXT NOT NULL DEFAULT '',
                ingredients JSONB NOT NULL DEFAULT '[]',
                steps TEXT NOT NULL DEFAULT '',
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('spanish', title), 'A') ||
                    setweight(to_tsvector('spanish', description), 'B') ||
                    setweight(to_tsvector('spanish', steps), 'C') ||
                    setweight(to_tsvector('spanish', COALESCE(regexp_replace(ingredients::text, '[{{}}"\\[\\],:]', ' ', 'g'), '')), 'D')
                ) STORED,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            "#,
        );

        sqlx::query(&create_recipes_data_sql)
            .execute(&self.pool)
            .await
            .context("Failed to create recipes_data table")?;
        debug!("Migration: recipes_data table created");

        let create_recipes_vector_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS recipes_vector (
                recipe_id UUID PRIMARY KEY REFERENCES recipes_data(id) ON DELETE CASCADE,
                embedding vector({})
            );
            "#,
            self.embedding_dim
        );

        sqlx::query(&create_recipes_vector_sql)
            .execute(&self.pool)
            .await
            .context("Failed to create recipes_vector table")?;
        debug!("Migration: recipes_vector table created");

        let has_old_table: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'recipes')",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        let has_new_data: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM recipes_data")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);

        if has_old_table && has_new_data == 0 {
            info!("Migrating data from 'recipes' to 'recipes_data' + 'recipes_vector'");
            let migrated = sqlx::query(
                r#"
                INSERT INTO recipes_data (id, slug, url, title, description, prep_time_minutes, cook_time_minutes, difficulty, servings_count, servings_unit, courses, food_types, chef, ingredients, steps)
                SELECT id, slug, url, title, description, prep_time_minutes, cook_time_minutes, difficulty, servings_count, servings_unit, courses, food_types, chef, ingredients, steps
                FROM recipes
                "#,
            )
            .execute(&self.pool)
            .await
            .context("Failed to migrate recipes_data")?;
            debug!("Migration: {} rows copied to recipes_data", migrated.rows_affected());

            let vector_migrated = sqlx::query(
                r#"
                INSERT INTO recipes_vector (recipe_id, embedding)
                SELECT id, embedding
                FROM recipes
                WHERE embedding IS NOT NULL
                "#,
            )
            .execute(&self.pool)
            .await
            .context("Failed to migrate recipes_vector")?;
            debug!("Migration: {} vectors copied to recipes_vector", vector_migrated.rows_affected());

            sqlx::query("DROP TABLE recipes")
                .execute(&self.pool)
                .await
                .context("Failed to drop old recipes table")?;
            info!("Migration complete: old 'recipes' table dropped");
        } else if has_old_table && has_new_data > 0 {
            sqlx::query("DROP TABLE recipes")
                .execute(&self.pool)
                .await
                .context("Failed to drop old recipes table")?;
            info!("Old 'recipes' table dropped (data already in new tables)");
        }

        let has_recipes_vector_idx: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_recipes_vector_embedding')",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !has_recipes_vector_idx {
            let idx_exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = 'idx_recipes_embedding')",
            )
            .fetch_one(&self.pool)
            .await
            .unwrap_or(false);

            if idx_exists {
                sqlx::query("DROP INDEX IF EXISTS idx_recipes_embedding")
                    .execute(&self.pool)
                    .await?;
                debug!("Migration: dropped old idx_recipes_embedding index");
            }

            sqlx::query(
                "CREATE INDEX IF NOT EXISTS idx_recipes_vector_embedding ON recipes_vector USING hnsw (embedding vector_cosine_ops);",
            )
            .execute(&self.pool)
            .await
            .context("Failed to create HNSW index on recipes_vector")?;
            debug!("Migration: HNSW embedding index created on recipes_vector");
        }

        sqlx::query("DROP INDEX IF EXISTS idx_recipes_search")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_recipes_data_search ON recipes_data USING GIN (search_vector);")
            .execute(&self.pool)
            .await
            .context("Failed to create full-text index")?;
        debug!("Migration: full-text search index created on recipes_data");

        sqlx::query("DROP INDEX IF EXISTS idx_recipes_courses")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_recipes_data_courses ON recipes_data USING GIN (courses);")
            .execute(&self.pool)
            .await
            .context("Failed to create courses index")?;
        debug!("Migration: courses GIN index created on recipes_data");

        sqlx::query("DROP INDEX IF EXISTS idx_recipes_food_types")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_recipes_data_food_types ON recipes_data USING GIN (food_types);")
            .execute(&self.pool)
            .await
            .context("Failed to create food_types index")?;
        debug!("Migration: food_types GIN index created on recipes_data");

        sqlx::query("DROP INDEX IF EXISTS idx_recipes_chef")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_recipes_data_chef ON recipes_data (chef);")
            .execute(&self.pool)
            .await
            .context("Failed to create chef index")?;
        debug!("Migration: chef btree index created on recipes_data");

        info!("Database migrations completed in {:.2?}", mig_start.elapsed());

        Ok(())
    }

    pub async fn upsert_recipe(&self, recipe: &Recipe, embedding: &Vector) -> anyhow::Result<()> {
        let upsert_start = Instant::now();
        let ingredients_json = serde_json::to_value(&recipe.ingredients)?;

        let mut tx = self.pool.begin().await.context("Failed to begin transaction")?;

        sqlx::query(
            r#"
            INSERT INTO recipes_data (
                id, slug, url, title, description, prep_time_minutes, cook_time_minutes,
                difficulty, servings_count, servings_unit, courses, food_types, chef,
                ingredients, steps
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (id) DO UPDATE SET
                slug = EXCLUDED.slug,
                url = EXCLUDED.url,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                prep_time_minutes = EXCLUDED.prep_time_minutes,
                cook_time_minutes = EXCLUDED.cook_time_minutes,
                difficulty = EXCLUDED.difficulty,
                servings_count = EXCLUDED.servings_count,
                servings_unit = EXCLUDED.servings_unit,
                courses = EXCLUDED.courses,
                food_types = EXCLUDED.food_types,
                chef = EXCLUDED.chef,
                ingredients = EXCLUDED.ingredients,
                steps = EXCLUDED.steps
            "#,
        )
        .bind(recipe.id)
        .bind(&recipe.slug)
        .bind(&recipe.url)
        .bind(&recipe.title)
        .bind(&recipe.description)
        .bind(recipe.prep_time_minutes)
        .bind(recipe.cook_time_minutes)
        .bind(recipe.difficulty)
        .bind(recipe.servings_count)
        .bind(&recipe.servings_unit)
        .bind(&recipe.courses)
        .bind(&recipe.food_types)
        .bind(&recipe.chef)
        .bind(&ingredients_json)
        .bind(&recipe.steps)
        .execute(&mut *tx)
        .await
        .context("Failed to upsert recipe_data")?;

        sqlx::query(
            r#"
            INSERT INTO recipes_vector (recipe_id, embedding)
            VALUES ($1, $2)
            ON CONFLICT (recipe_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
            "#,
        )
        .bind(recipe.id)
        .bind(embedding)
        .execute(&mut *tx)
        .await
        .context("Failed to upsert recipe_vector")?;

        tx.commit().await.context("Failed to commit transaction")?;

        debug!("Upserted recipe '{}' in {:.2?} (id: {})", recipe.title, upsert_start.elapsed(), recipe.id);
        Ok(())
    }

    pub async fn search(
        &self,
        query_embedding: &Vector,
        query_text: &str,
        limit: usize,
        offset: usize,
        min_similarity: f64,
    ) -> anyhow::Result<PaginatedResult<SearchResult>> {
        let search_start = Instant::now();
        debug!("Search: query='{}', limit={}, offset={}, min_similarity={}", query_text, limit, offset, min_similarity);

        let text_weight = 0.3;
        let vector_weight = 0.7;

        let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM recipes_vector")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);
        debug!("Search: total indexable recipes={}", total);

        let sql = r#"
            WITH vector_search AS (
                SELECT rd.id, rd.slug, rd.url, rd.title, rd.description, rd.prep_time_minutes,
                       rd.cook_time_minutes, rd.difficulty, rd.servings_count, rd.servings_unit,
                       rd.courses, rd.food_types, rd.chef, rd.ingredients, rd.steps,
                       (1 - (rv.embedding <=> $1))::double precision as sim_score
                FROM recipes_vector rv
                JOIN recipes_data rd ON rd.id = rv.recipe_id
                WHERE rv.embedding IS NOT NULL
                ORDER BY rv.embedding <=> $1
                LIMIT $4 OFFSET $6
            ),
            text_search AS (
                SELECT id,
                       ts_rank(search_vector, plainto_tsquery('spanish', $2))::double precision as txt_score
                FROM recipes_data
                WHERE search_vector @@ plainto_tsquery('spanish', $2)
            )
            SELECT vs.id, vs.slug, vs.url, vs.title, vs.description, vs.prep_time_minutes,
                   vs.cook_time_minutes, vs.difficulty, vs.servings_count, vs.servings_unit,
                   vs.courses, vs.food_types, vs.chef, vs.ingredients, vs.steps,
                   vs.sim_score as similarity,
                   COALESCE(ts.txt_score, 0.0) as text_rank,
                   ($5 * vs.sim_score + $3 * COALESCE(ts.txt_score, 0.0))::double precision as combined_score
            FROM vector_search vs
            LEFT JOIN text_search ts ON vs.id = ts.id
            WHERE ($5 * vs.sim_score + $3 * COALESCE(ts.txt_score, 0.0)) >= $7
            ORDER BY combined_score DESC
        "#;

        let query_start = Instant::now();
        let rows = sqlx::query(sql)
            .bind(query_embedding)
            .bind(query_text)
            .bind(text_weight)
            .bind(limit as i64)
            .bind(vector_weight)
            .bind(offset as i64)
            .bind(min_similarity)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| {
                tracing::error!("Search query failed: {:?}", e);
                e
            })
            .context("Failed to search")?;

        debug!("Search: query executed in {:.2?}, returned {} rows", query_start.elapsed(), rows.len());

        let items: Vec<SearchResult> = rows
            .into_iter()
            .map(|row| {
                let ingredients: Vec<crate::models::Ingredient> =
                    serde_json::from_value(row.get("ingredients")).unwrap_or_default();

                SearchResult {
                    id: row.get("id"),
                    slug: row.get("slug"),
                    url: row.get("url"),
                    title: row.get("title"),
                    description: row.get("description"),
                    prep_time_minutes: row.get("prep_time_minutes"),
                    cook_time_minutes: row.get("cook_time_minutes"),
                    difficulty: row.get("difficulty"),
                    servings_count: row.get("servings_count"),
                    servings_unit: row.get("servings_unit"),
                    courses: row.get("courses"),
                    food_types: row.get("food_types"),
                    chef: row.get("chef"),
                    ingredients,
                    steps: row.get("steps"),
                    similarity: row.get("similarity"),
                    text_rank: row.get("text_rank"),
                    combined_score: row.get("combined_score"),
                }
            })
            .collect();

        debug!("Search: total time {:.2?}, {} results", search_start.elapsed(), items.len());

        Ok(PaginatedResult {
            items,
            total,
            offset,
            limit,
        })
    }

    pub async fn search_bm25_only(
        &self,
        query_text: &str,
        limit: usize,
        offset: usize,
        min_similarity: f64,
    ) -> anyhow::Result<PaginatedResult<SearchResult>> {
        let search_start = Instant::now();
        debug!("BM25 search: query='{}', limit={}, offset={}, min_similarity={}", query_text, limit, offset, min_similarity);

        let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM recipes_data")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);
        debug!("BM25 search: total recipes={}", total);

        let sql = r#"
            SELECT id, slug, url, title, description, prep_time_minutes, cook_time_minutes,
                   difficulty, servings_count, servings_unit, courses, food_types, chef,
                   ingredients, steps,
                   ts_rank(search_vector, plainto_tsquery('spanish', $1))::double precision as text_rank
            FROM recipes_data
            WHERE search_vector @@ plainto_tsquery('spanish', $1)
            ORDER BY text_rank DESC
            LIMIT $2 OFFSET $3
        "#;

        let query_start = Instant::now();
        let rows = sqlx::query(sql)
            .bind(query_text)
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| {
                tracing::error!("BM25 search query failed: {:?}", e);
                e
            })
            .context("Failed to search")?;

        debug!("BM25 search: query executed in {:.2?}, returned {} rows", query_start.elapsed(), rows.len());

        let items: Vec<SearchResult> = rows
            .into_iter()
            .map(|row| {
                let ingredients: Vec<crate::models::Ingredient> =
                    serde_json::from_value(row.get("ingredients")).unwrap_or_default();
                let text_rank: f64 = row.get("text_rank");

                SearchResult {
                    id: row.get("id"),
                    slug: row.get("slug"),
                    url: row.get("url"),
                    title: row.get("title"),
                    description: row.get("description"),
                    prep_time_minutes: row.get("prep_time_minutes"),
                    cook_time_minutes: row.get("cook_time_minutes"),
                    difficulty: row.get("difficulty"),
                    servings_count: row.get("servings_count"),
                    servings_unit: row.get("servings_unit"),
                    courses: row.get("courses"),
                    food_types: row.get("food_types"),
                    chef: row.get("chef"),
                    ingredients,
                    steps: row.get("steps"),
                    similarity: 0.0,
                    text_rank,
                    combined_score: text_rank,
                }
            })
            .filter(|r| r.combined_score >= min_similarity)
            .collect();

        debug!("BM25 search: total time {:.2?}, {} results (after threshold)", search_start.elapsed(), items.len());

        Ok(PaginatedResult {
            items,
            total,
            offset,
            limit,
        })
    }

    pub async fn get_recipe_by_id(&self, id: &Uuid) -> anyhow::Result<Option<Recipe>> {
        debug!("Get recipe by id: {}", id);
        let get_start = Instant::now();

        let row = sqlx::query(
            r#"
            SELECT id, slug, url, title, description, prep_time_minutes, cook_time_minutes,
                   difficulty, servings_count, servings_unit, courses, food_types, chef,
                   ingredients, steps
            FROM recipes_data WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get recipe by id")?;

        match row {
            Some(r) => {
                let recipe = row_to_recipe(&r);
                debug!("Get recipe by id: found '{}' in {:.2?}", recipe.title, get_start.elapsed());
                Ok(Some(recipe))
            }
            None => {
                debug!("Get recipe by id: not found in {:.2?}", get_start.elapsed());
                Ok(None)
            }
        }
    }

    pub async fn search_by_ingredients(
        &self,
        ingredients: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<Recipe>> {
        debug!("Search by ingredients: {:?}, limit={}", ingredients, limit);
        let search_start = Instant::now();

        if ingredients.is_empty() {
            debug!("Search by ingredients: empty list, returning []");
            return Ok(vec![]);
        }

        let conditions: Vec<String> = (1..=ingredients.len())
            .map(|i| format!("ingredients @> ${}", i))
            .collect();

        let where_clause = conditions.join(" OR ");
        let sql = format!(
            "SELECT id, slug, url, title, description, prep_time_minutes, cook_time_minutes, difficulty, servings_count, servings_unit, courses, food_types, chef, ingredients, steps FROM recipes_data WHERE {} ORDER BY title LIMIT ${}",
            where_clause,
            ingredients.len() + 1
        );

        let mut query = sqlx::query(&sql);

        for ingredient in ingredients {
            let json = serde_json::json!([{"name": ingredient}]);
            query = query.bind(json);
        }
        query = query.bind(limit as i64);

        let rows = query
            .fetch_all(&self.pool)
            .await
            .context("Failed to search by ingredients")?;

        let recipes = rows.iter().map(|r| row_to_recipe(r)).collect::<Vec<_>>();
        debug!("Search by ingredients: {} results in {:.2?}", recipes.len(), search_start.elapsed());

        Ok(recipes)
    }

    pub async fn search_by_filters(
        &self,
        course: Option<&str>,
        food_type: Option<&str>,
        chef: Option<&str>,
        max_difficulty: Option<i32>,
        max_total_time: Option<i32>,
        limit: usize,
    ) -> anyhow::Result<Vec<Recipe>> {
        debug!("Search by filters: course={:?}, food_type={:?}, chef={:?}, max_difficulty={:?}, max_total_time={:?}, limit={}",
            course, food_type, chef, max_difficulty, max_total_time, limit);
        let search_start = Instant::now();

        let mut conditions = Vec::new();
        let mut param_idx = 1;

        if let Some(_c) = course {
            conditions.push(format!("courses @> ARRAY[${}]::text[]", param_idx));
            param_idx += 1;
        }

        if let Some(_ft) = food_type {
            conditions.push(format!("food_types @> ARRAY[${}]::text[]", param_idx));
            param_idx += 1;
        }

        if let Some(_ch) = chef {
            conditions.push(format!("chef ILIKE ${}", param_idx));
            param_idx += 1;
        }

        if let Some(_md) = max_difficulty {
            conditions.push(format!("difficulty <= ${}", param_idx));
            param_idx += 1;
        }

        if let Some(_mt) = max_total_time {
            conditions.push(format!(
                "COALESCE(prep_time_minutes, 0) + COALESCE(cook_time_minutes, 0) <= ${}",
                param_idx
            ));
            param_idx += 1;
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, slug, url, title, description, prep_time_minutes, cook_time_minutes, difficulty, servings_count, servings_unit, courses, food_types, chef, ingredients, steps FROM recipes_data {} ORDER BY title LIMIT ${}",
            where_clause, param_idx
        );

        let mut query = sqlx::query(&sql);

        if let Some(c) = course {
            query = query.bind(c);
        }
        if let Some(ft) = food_type {
            query = query.bind(ft);
        }
        if let Some(ch) = chef {
            query = query.bind(format!("%{}%", ch));
        }
        if let Some(md) = max_difficulty {
            query = query.bind(md);
        }
        if let Some(mt) = max_total_time {
            query = query.bind(mt);
        }
        query = query.bind(limit as i64);

        let rows = query
            .fetch_all(&self.pool)
            .await
            .context("Failed to search by filters")?;

        let recipes = rows.iter().map(|r| row_to_recipe(r)).collect::<Vec<_>>();
        debug!("Search by filters: {} results in {:.2?}", recipes.len(), search_start.elapsed());

        Ok(recipes)
    }

    pub async fn count(&self) -> anyhow::Result<(i64, i64)> {
        let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM recipes_data")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);

        let with_embeddings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM recipes_vector")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);

        debug!("Count: total={}, with_embeddings={}", total, with_embeddings);
        Ok((total, with_embeddings))
    }

    pub async fn clear_db(&self) -> anyhow::Result<i64> {
        let clear_start = Instant::now();
        let (before, _) = self.count().await.unwrap_or((0, 0));
        debug!("Clear DB: {} recipes before deletion", before);

        sqlx::query("DELETE FROM recipes_vector")
            .execute(&self.pool)
            .await
            .context("Failed to clear recipes_vector")?;

        let result = sqlx::query("DELETE FROM recipes_data")
            .execute(&self.pool)
            .await
            .context("Failed to clear recipes_data")?;

        let deleted = result.rows_affected() as i64;
        info!("Cleared {} recipes from database in {:.2?}", deleted, clear_start.elapsed());

        Ok(deleted)
    }
}

fn row_to_recipe(row: &sqlx::postgres::PgRow) -> Recipe {
    let ingredients: Vec<crate::models::Ingredient> =
        serde_json::from_value(row.get("ingredients")).unwrap_or_default();

    Recipe {
        id: row.get("id"),
        slug: row.get("slug"),
        url: row.get("url"),
        title: row.get("title"),
        description: row.get("description"),
        prep_time_minutes: row.get("prep_time_minutes"),
        cook_time_minutes: row.get("cook_time_minutes"),
        difficulty: row.get("difficulty"),
        servings_count: row.get("servings_count"),
        servings_unit: row.get("servings_unit"),
        courses: row.get("courses"),
        food_types: row.get("food_types"),
        chef: row.get("chef"),
        ingredients,
        steps: row.get("steps"),
    }
}
