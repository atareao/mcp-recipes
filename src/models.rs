use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

fn deserialize_null_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ingredient {
    pub name: String,
    pub quantity: Option<f64>,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub id: Uuid,
    pub slug: String,
    pub url: String,
    pub title: String,
    #[serde(deserialize_with = "deserialize_null_string")]
    pub description: String,
    pub prep_time_minutes: Option<i32>,
    pub cook_time_minutes: Option<i32>,
    pub difficulty: Option<i32>,
    pub servings_count: Option<i32>,
    pub servings_unit: String,
    pub courses: Vec<String>,
    pub food_types: Vec<String>,
    pub chef: String,
    pub ingredients: Vec<Ingredient>,
    pub steps: String,
}
