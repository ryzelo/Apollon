//! Structured results shared by the CLI and browser application.
use crate::model::Model;
use serde_json::{json, Value};
use std::path::Path;

pub fn tokens(m: &Model, ids: &[u32]) -> Vec<Value> {
    ids.iter().enumerate().map(|(index, &id)| json!({"id":id,"index":index,"text":m.tok.piece(id)})).collect()
}
pub fn probabilities(m: &Model, values: &[(u32, f32)]) -> Vec<Value> {
    values.iter().map(|&(id, probability)| json!({"id":id,"text":m.tok.piece(id),"probability":probability})).collect()
}
pub fn result(m: &Model, ids: &[u32], focus: usize, target: u32, mode: &str, output: &[(u32, f32)]) -> Value {
    json!({"schema_version":2,"mode":mode,"model":{"name":m.name,"layers":m.n_layer(),"architecture":m.arch},
        "tokens":tokens(m,ids),"focus":focus,"target":{"id":target,"text":m.tok.piece(target)},
        "output":probabilities(m,output),"components":[],"checks":[],"generated":[],"answer":"",
        "metric":"target_log_odds_with_minus_without"})
}
pub fn write(path: &Path, result: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec(result).map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}",path.display()))
}
