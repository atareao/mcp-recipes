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
    #[serde(default, deserialize_with = "deserialize_null_string")]
    pub description: String,
    pub prep_time_minutes: Option<i32>,
    pub cook_time_minutes: Option<i32>,
    pub difficulty: Option<i32>,
    pub servings_count: Option<i32>,
    #[serde(default)]
    pub servings_unit: String,
    #[serde(default)]
    pub courses: Vec<String>,
    #[serde(default)]
    pub food_types: Vec<String>,
    #[serde(default)]
    pub chef: String,
    #[serde(default)]
    pub ingredients: Vec<Ingredient>,
    #[serde(default)]
    pub steps: String,
}

pub fn build_embed_text(recipe: &Recipe) -> String {
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
                    format!("{} {} {}", ing.quantity.unwrap(), ing.unit, ing.name)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    mod deserialize_null_string {
        use super::*;

        #[test]
        fn returns_empty_string_when_null() {
            let input = json!(null);
            let result: String = serde_json::from_value(input).unwrap_or_default();
            assert_eq!(result, "");
        }

        #[test]
        fn returns_string_when_value_present() {
            let input = json!("hello");
            let result: String = serde_json::from_value(input).unwrap_or_default();
            assert_eq!(result, "hello");
        }
    }

    mod recipe_deserialization {
        use super::*;

        fn base_recipe_json() -> serde_json::Value {
            json!({
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "slug": "test-recipe",
                "url": "https://example.com/test",
                "title": "Test Recipe"
            })
        }

        #[test]
        fn handles_null_description() {
            let json = json!({
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "slug": "test-recipe",
                "url": "https://example.com/test",
                "title": "Test Recipe",
                "description": null
            });
            let recipe: Recipe = serde_json::from_value(json).unwrap();
            assert_eq!(recipe.description, "", "null description should become empty string");
        }

        #[test]
        fn handles_missing_optional_fields() {
            let json = base_recipe_json();
            let recipe: Recipe = serde_json::from_value(json).unwrap();
            assert_eq!(recipe.description, "");
            assert!(recipe.prep_time_minutes.is_none());
            assert!(recipe.cook_time_minutes.is_none());
            assert!(recipe.difficulty.is_none());
            assert!(recipe.servings_count.is_none());
            assert_eq!(recipe.servings_unit, "");
            assert!(recipe.courses.is_empty());
            assert!(recipe.food_types.is_empty());
            assert_eq!(recipe.chef, "");
            assert!(recipe.ingredients.is_empty());
            assert_eq!(recipe.steps, "");
        }

        #[test]
        fn handles_full_recipe() {
            let json = json!({
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "slug": "full-recipe",
                "url": "https://example.com/full",
                "title": "Full Recipe",
                "description": "A delicious test",
                "prep_time_minutes": 15,
                "cook_time_minutes": 30,
                "difficulty": 2,
                "servings_count": 4,
                "servings_unit": "raciones",
                "courses": ["Primeros", "Segundos"],
                "food_types": ["Carne y Aves"],
                "chef": "Test Chef",
                "ingredients": [
                    {"name": "pollo", "quantity": 500, "unit": "g"},
                    {"name": "sal", "quantity": null, "unit": ""}
                ],
                "steps": "Step 1\nStep 2"
            });
            let recipe: Recipe = serde_json::from_value(json).unwrap();
            assert_eq!(recipe.title, "Full Recipe");
            assert_eq!(recipe.description, "A delicious test");
            assert_eq!(recipe.prep_time_minutes, Some(15));
            assert_eq!(recipe.cook_time_minutes, Some(30));
            assert_eq!(recipe.difficulty, Some(2));
            assert_eq!(recipe.servings_count, Some(4));
            assert_eq!(recipe.servings_unit, "raciones");
            assert_eq!(recipe.courses, vec!["Primeros", "Segundos"]);
            assert_eq!(recipe.food_types, vec!["Carne y Aves"]);
            assert_eq!(recipe.chef, "Test Chef");
            assert_eq!(recipe.ingredients.len(), 2);
            assert_eq!(recipe.ingredients[0].name, "pollo");
            assert_eq!(recipe.ingredients[0].quantity, Some(500.0));
            assert_eq!(recipe.ingredients[0].unit, "g");
            assert_eq!(recipe.ingredients[1].name, "sal");
            assert_eq!(recipe.ingredients[1].quantity, None);
            assert_eq!(recipe.ingredients[1].unit, "");
            assert_eq!(recipe.steps, "Step 1\nStep 2");
        }
    }

    mod build_embed_text {
        use super::*;

        fn minimal_recipe() -> Recipe {
            Recipe {
                id: uuid::Uuid::nil(),
                slug: "test".to_string(),
                url: "https://example.com".to_string(),
                title: "Test Recipe".to_string(),
                description: String::new(),
                prep_time_minutes: None,
                cook_time_minutes: None,
                difficulty: None,
                servings_count: None,
                servings_unit: String::new(),
                courses: vec![],
                food_types: vec![],
                chef: String::new(),
                ingredients: vec![],
                steps: String::new(),
            }
        }

        fn full_recipe() -> Recipe {
            Recipe {
                id: uuid::Uuid::nil(),
                slug: "full".to_string(),
                url: "https://example.com".to_string(),
                title: "Full Recipe".to_string(),
                description: "A tasty dish".to_string(),
                prep_time_minutes: Some(15),
                cook_time_minutes: Some(30),
                difficulty: Some(2),
                servings_count: Some(4),
                servings_unit: "raciones".to_string(),
                courses: vec!["Primeros".to_string(), "Segundos".to_string()],
                food_types: vec!["Carne".to_string()],
                chef: "Chef".to_string(),
                ingredients: vec![
                    Ingredient { name: "pollo".to_string(), quantity: Some(500.0), unit: "g".to_string() },
                    Ingredient { name: "sal".to_string(), quantity: None, unit: String::new() },
                ],
                steps: "Cook it.".to_string(),
            }
        }

        #[test]
        fn includes_title_for_minimal_recipe() {
            let text = build_embed_text(&minimal_recipe());
            assert!(text.contains("Title: Test Recipe"));
        }

        #[test]
        fn skips_empty_fields() {
            let text = build_embed_text(&minimal_recipe());
            assert!(!text.contains("Description:"));
            assert!(!text.contains("Courses:"));
            assert!(!text.contains("Ingredients:"));
            assert!(!text.contains("Steps:"));
        }

        #[test]
        fn includes_all_fields_when_present() {
            let text = build_embed_text(&full_recipe());
            assert!(text.contains("Title: Full Recipe"));
            assert!(text.contains("Description: A tasty dish"));
            assert!(text.contains("Courses: Primeros, Segundos"));
            assert!(text.contains("Food types: Carne"));
            assert!(text.contains("Chef: Chef"));
            assert!(text.contains("Ingredients:"));
            assert!(text.contains("Steps: Cook it."));
        }

        #[test]
        fn formats_ingredients_with_quantity() {
            let text = build_embed_text(&full_recipe());
            assert!(text.contains("500 g pollo"));
        }

        #[test]
        fn formats_ingredients_without_quantity() {
            let text = build_embed_text(&full_recipe());
            assert!(text.contains("sal"));
        }

        #[test]
        fn joins_ingredients_with_comma() {
            let text = build_embed_text(&full_recipe());
            assert!(text.contains("500 g pollo, sal"));
        }
    }
}
