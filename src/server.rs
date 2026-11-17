//! Local web workspace. Jobs execute in isolated, cancellable CLI processes.
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    process::Command,
    sync::{watch, Mutex, Semaphore},
};
use tower_http::services::ServeDir;

#[derive(Clone, Serialize, Deserialize)]
pub struct Request {
    model: String,
    prompt: String,
    kind: String,
    #[serde(default = "first")]
    answer: usize,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    component: Option<String>,
    #[serde(default)]
    chat: bool,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    options: BTreeMap<String, String>,
}
fn first() -> usize {
    1
}
#[derive(Clone, Serialize, Deserialize)]
struct Job {
    id: String,
    status: String,
    request: Request,
    error: Option<String>,
    result: Option<Value>,
    /// last line the engine wrote while the run is in progress (filled on request, never stored)
    #[serde(default, skip_deserializing)]
    progress: Option<String>,
}
struct App {
    models: Mutex<Option<(std::time::Instant, Value)>>,
    jobs: Mutex<BTreeMap<String, Job>>,
    cancel: Mutex<BTreeMap<String, watch::Sender<bool>>>,
    slots: Semaphore,
    root: PathBuf,
}
type Shared = Arc<App>;
type ApiError = (StatusCode, Json<Value>);
fn error(code: StatusCode, message: impl ToString) -> ApiError {
    (code, Json(json!({"error":message.to_string()})))
}
fn trusted(headers: &HeaderMap) -> Result<(), ApiError> {
    if let Some(origin) = headers.get("origin") {
        let host = headers
            .get("host")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if origin.to_str().unwrap_or("") != format!("http://{host}") {
            return Err(error(
                StatusCode::FORBIDDEN,
                "Cross-origin requests are disabled",
            ));
        }
    }
    Ok(())
}
async fn local_only(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let host = request
        .headers()
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let hostname = host.split(':').next().unwrap_or("");
    if !["localhost", "127.0.0.1"].contains(&hostname) {
        return error(StatusCode::FORBIDDEN, "Use the local workspace address").into_response();
    }
    next.run(request).await
}
async fn save(app: &App, job: &Job) -> Result<(), String> {
    let dir = app.root.join(&job.id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec_pretty(job).map_err(|e| e.to_string())?;
    tokio::fs::write(dir.join("job.tmp"), bytes)
        .await
        .map_err(|e| e.to_string())?;
    tokio::fs::rename(dir.join("job.tmp"), dir.join("job.json"))
        .await
        .map_err(|e| e.to_string())
}
/// The local model list. Listing scans the model caches in a separate process, so it is kept for a minute.
async fn models(State(app): State<Shared>) -> Result<Json<Value>, ApiError> {
    let mut cache = app.models.lock().await;
    if let Some((at, value)) = cache.as_ref() {
        if at.elapsed().as_secs() < 60 {
            return Ok(Json(value.clone()));
        }
    }
    let value = scan_models().await?;
    *cache = Some((std::time::Instant::now(), value.clone()));
    Ok(Json(value))
}
async fn scan_models() -> Result<Value, ApiError> {
    let exe = std::env::current_exe().map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let output = Command::new(exe)
        .arg("--models-json")
        .output()
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    if !output.status.success() {
        return Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))
}
/// The last line of the engine's log: its progress messages end in "\r" or "\n".
async fn progress(app: &App, id: &str) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut file = tokio::fs::File::open(app.root.join(id).join("worker.log")).await.ok()?;
    let len = file.metadata().await.ok()?.len();
    file.seek(std::io::SeekFrom::Start(len.saturating_sub(2048))).await.ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).await.ok()?;
    String::from_utf8_lossy(&bytes)
        .split(['\r', '\n'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .last()
        .map(|line| line.chars().take(200).collect())
}
async fn list(State(app): State<Shared>) -> Json<Value> {
    let jobs = app.jobs.lock().await;
    let summaries: Vec<_> = jobs
        .values()
        .rev()
        .map(|j| json!({"id":j.id,"status":j.status,"request":j.request,"error":j.error}))
        .collect();
    Json(json!({"runs":summaries}))
}
async fn detail(State(app): State<Shared>, Path(id): Path<String>) -> Result<Json<Job>, ApiError> {
    let mut job = app
        .jobs
        .lock()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Run not found"))?;
    if job.status == "running" {
        job.progress = progress(&app, &id).await;
    }
    Ok(Json(job))
}
/// Removes a finished run and its files for good.
async fn remove(
    State(app): State<Shared>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    trusted(&headers)?;
    let mut jobs = app.jobs.lock().await;
    let job = jobs
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "Run not found"))?;
    if job.status == "queued" || job.status == "running" {
        return Err(error(StatusCode::CONFLICT, "Cancel the run before deleting it"));
    }
    let in_use = jobs.values().any(|other| {
        (other.status == "queued" || other.status == "running")
            && other.request.options.get("atlas-run") == Some(&id)
    });
    if in_use {
        return Err(error(StatusCode::CONFLICT, "A running trace is using this atlas"));
    }
    // job ids are digits only, so the path stays inside the run folder
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(error(StatusCode::BAD_REQUEST, "Invalid run id"));
    }
    tokio::fs::remove_dir_all(app.root.join(&id))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    jobs.remove(&id);
    Ok(Json(json!({"status":"deleted"})))
}
async fn create(
    State(app): State<Shared>,
    headers: HeaderMap,
    Json(request): Json<Request>,
) -> Result<Json<Job>, ApiError> {
    trusted(&headers)?;
    if request.model.trim().is_empty()
        || (!["neuron", "atlas"].contains(&request.kind.as_str())
            && request.prompt.trim().is_empty())
        || request.answer == 0
        || request.answer > 256
        || ![
            "generate",
            "decompose",
            "probe",
            "trace",
            "circuit",
            "neuron",
            "atlas",
            "tokens",
            "dump",
        ]
        .contains(&request.kind.as_str())
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Choose a model, a prompt, a supported analysis and answer token 1–256",
        ));
    }
    if ["probe", "trace", "neuron"].contains(&request.kind.as_str())
        && request.component.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err(error(StatusCode::BAD_REQUEST, "Select a component"));
    }
    // choices the engine takes from a fixed list
    for (key, allowed) in [
        ("gpu", &["auto", "metal", "cuda", "off"][..]),
        ("circuit-ablation", &["mean", "zero"][..]),
        ("circuit-norms", &["frozen", "live"][..]),
    ] {
        if let Some(v) = request.options.get(key).filter(|v| !v.is_empty()) {
            if !allowed.contains(&v.as_str()) {
                return Err(error(StatusCode::BAD_REQUEST, format!("{key} must be one of {}", allowed.join(", "))));
            }
        }
    }
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let job = Job {
        id: id.clone(),
        status: "queued".into(),
        request,
        error: None,
        result: None,
        progress: None,
    };
    save(&app, &job)
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let (tx, rx) = watch::channel(false);
    app.cancel.lock().await.insert(id.clone(), tx);
    app.jobs.lock().await.insert(id.clone(), job.clone());
    tokio::spawn(worker(app, id, rx));
    Ok(Json(job))
}
async fn cancel(
    State(app): State<Shared>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    trusted(&headers)?;
    if let Some(tx) = app.cancel.lock().await.get(&id) {
        let _ = tx.send(true);
        return Ok(Json(json!({"status":"cancelling"})));
    }
    Err(error(
        StatusCode::CONFLICT,
        "Run is already finished or does not exist",
    ))
}
async fn worker(app: Shared, id: String, mut cancel: watch::Receiver<bool>) {
    let permit = tokio::select! { p = app.slots.acquire() => p.ok(), _ = cancel.changed() => None };
    let mut job = app.jobs.lock().await[&id].clone();
    if permit.is_none() || *cancel.borrow() {
        job.status = "cancelled".into();
    } else {
        job.status = "running".into();
        app.jobs.lock().await.insert(id.clone(), job.clone());
        let _ = save(&app, &job).await;
        let outcome = execute(&app, &job, &mut cancel).await;
        match outcome {
            Ok(Some(result)) => {
                job.result = Some(result);
                job.status = "completed".into();
            }
            Ok(None) => job.status = "cancelled".into(),
            Err(e) => {
                job.status = "failed".into();
                job.error = Some(e);
            }
        }
    }
    if let Err(e) = save(&app, &job).await {
        job.error = Some(format!("Could not persist run: {e}"));
    }
    app.jobs.lock().await.insert(id.clone(), job);
    app.cancel.lock().await.remove(&id);
}
async fn execute(
    app: &App,
    job: &Job,
    cancel: &mut watch::Receiver<bool>,
) -> Result<Option<Value>, String> {
    let dir = app.root.join(&job.id);
    let result = dir.join("result.json");
    let log = std::fs::File::create(dir.join("worker.log")).map_err(|e| e.to_string())?;
    let stderr = log.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    let r = &job.request;
    let static_run = matches!(r.kind.as_str(), "neuron" | "atlas");
    let file_only = matches!(r.kind.as_str(), "atlas" | "tokens" | "dump");
    cmd.args(["-m", &r.model, "--no-color", "--html"])
        .arg(dir.join("report"));
    if !static_run {
        cmd.args(["--prompt", &r.prompt, "--answer", &r.answer.to_string()]);
    }
    if !file_only {
        cmd.arg("--result-json").arg(&result);
    }
    let mode_options: &[&str] = match r.kind.as_str() {
        "decompose" => &["top"],
        "probe" => &["probe-pos", "probe-max", "depth", "contrast"],
        "trace" => &["trace-depth", "trace-top", "scalars", "scalars-of"],
        "circuit" => &["circuit-kl", "circuit-depth", "circuit-tests", "circuit-ablation", "circuit-norms", "circuit-ref", "circuit-paths"],
        "atlas" => &["atlas-k", "atlas-layers"],
        _ => &[],
    };
    for key in mode_options
        .iter()
        .copied()
        .chain(["threads"])
        .chain((!matches!(r.kind.as_str(), "atlas" | "tokens")).then_some("gpu"))
        .chain((!static_run).then_some("pos"))
    {
        if let Some(value) = r.options.get(key).filter(|v| !v.trim().is_empty()) {
            cmd.arg(format!("--{key}")).arg(value);
        }
    }
    if r.kind == "decompose" && r.options.get("focus-only").is_some_and(|v| v == "true") {
        cmd.arg("--focus-only");
    }
    if r.kind == "trace" {
        if let Some(id) = r.options.get("atlas-run").filter(|v| !v.is_empty()) {
            let jobs = app.jobs.lock().await;
            let valid = jobs.get(id).is_some_and(|atlas| {
                atlas.status == "completed"
                    && atlas.request.kind == "atlas"
                    && atlas.request.model == r.model
            });
            if !valid {
                return Err("Choose a completed atlas built for this model.".into());
            }
            cmd.arg("--atlas").arg(app.root.join(id).join("atlas.apla"));
        }
    }
    if r.chat {
        cmd.arg("--chat");
    }
    if let Some(target) = &r.target {
        if !target.is_empty() {
            cmd.args(["--target", target]);
        }
    }
    match r.kind.as_str() {
        "decompose" => {
            cmd.arg("--decompose");
        }
        "probe" => {
            cmd.args(["--probe", r.component.as_deref().unwrap()]);
        }
        "trace" => {
            cmd.args(["--trace", r.component.as_deref().unwrap()]);
            cmd.arg("--dump-neuron").arg(dir.join("neuron-trace.json"));
        }
        "circuit" => {
            cmd.arg("--circuit");
            cmd.arg("--dump-circuit").arg(dir.join("circuit.json"));
        }
        "neuron" => {
            cmd.args(["--neuron", r.component.as_deref().unwrap()]);
        }
        "atlas" => {
            cmd.arg("--atlas").arg(dir.join("atlas.apla"));
        }
        "tokens" => {
            cmd.arg("--tokens-only");
        }
        "dump" => {
            cmd.arg("--dump").arg(dir.join("forward.bin"));
        }
        _ => {}
    }
    #[cfg(windows)]
    cmd.creation_flags(0x08000000);
    let mut child = cmd
        .stdout(log)
        .stderr(stderr)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| e.to_string())?;
    let status = tokio::select! { status=child.wait()=>status.map_err(|e| e.to_string())?, _=cancel.changed()=>{ let _=child.kill().await; let _=child.wait().await; return Ok(None); } };
    if !status.success() {
        let log = tokio::fs::read_to_string(dir.join("worker.log"))
            .await
            .unwrap_or_default();
        return Err(log
            .chars()
            .rev()
            .take(4000)
            .collect::<String>()
            .chars()
            .rev()
            .collect());
    }
    if r.kind == "tokens" {
        let output = tokio::fs::read_to_string(dir.join("worker.log"))
            .await
            .map_err(|e| e.to_string())?;
        let ids = output
            .lines()
            .find_map(|line| serde_json::from_str::<Vec<u32>>(line).ok())
            .ok_or("Tokenizer produced no token IDs")?;
        tokio::fs::write(
            dir.join("tokens.json"),
            serde_json::to_vec(&ids).map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|e| e.to_string())?;
    }
    if file_only {
        return Ok(Some(
            json!({"schema_version":2,"mode":r.kind,"model":{"name":r.model},"components":[],"checks":[],"tokens":[],"output":[]}),
        ));
    }
    let bytes = tokio::fs::read(result)
        .await
        .map_err(|e| format!("Analysis produced no structured result: {e}"))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| e.to_string())
}
async fn artifact_paths(app: &App, id: &str) -> Result<Vec<String>, ApiError> {
    if !app
        .jobs
        .lock()
        .await
        .get(id)
        .is_some_and(|job| job.status == "completed")
    {
        return Err(error(StatusCode::NOT_FOUND, "Completed run not found"));
    }
    let mut entries = tokio::fs::read_dir(app.root.join(id))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut files = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().into_owned();
        if (name.starts_with("report_") && name.ends_with(".html"))
            || matches!(name.as_str(), "atlas.apla" | "forward.bin" | "tokens.json" | "neuron-trace.json" | "circuit.json")
        {
            files.push(name);
        }
    }
    files.sort();
    Ok(files)
}

async fn artifacts(
    State(app): State<Shared>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(json!(artifact_paths(&app, &id).await?)))
}

async fn artifact(
    State(app): State<Shared>,
    Path((id, name)): Path<(String, String)>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse;
    if !artifact_paths(&app, &id).await?.contains(&name) {
        return Err(error(StatusCode::NOT_FOUND, "Artifact not found"));
    }
    let file = tokio::fs::File::open(app.root.join(id).join(&name))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let headers = if name.ends_with(".html") {
        [
            ("content-type", "text/html; charset=utf-8"),
            ("content-disposition", "inline"),
        ]
    } else {
        [
            ("content-type", "application/octet-stream"),
            ("content-disposition", "attachment"),
        ]
    };
    let body = axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(file));
    Ok((headers, body).into_response())
}
pub fn serve() -> Result<(), String> {
    tokio::runtime::Runtime::new().map_err(|e| e.to_string())?.block_on(async {
        let root=std::env::current_dir().map_err(|e| e.to_string())?.join("output/web-runs");
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let mut jobs=BTreeMap::new();
        for entry in std::fs::read_dir(&root).map_err(|e| e.to_string())?.flatten() {
            if let Ok(bytes)=std::fs::read(entry.path().join("job.json")) { if let Ok(mut job)=serde_json::from_slice::<Job>(&bytes) {
                if job.status=="running" || job.status=="queued" {job.status="interrupted".into(); job.error=Some("Server stopped before this run finished. Start a new run to retry.".into());}
                jobs.insert(job.id.clone(),job);
            }}
        }
        let app=Arc::new(App { models:Mutex::new(None),jobs:Mutex::new(jobs),cancel:Mutex::new(BTreeMap::new()),slots:Semaphore::new(1),root });
        let router=Router::new().route("/api/models",get(models)).route("/api/runs",get(list).post(create)).route("/api/runs/{id}",get(detail).delete(remove)).route("/api/runs/{id}/cancel",post(cancel)).route("/api/runs/{id}/artifacts",get(artifacts)).route("/api/runs/{id}/artifacts/{name}",get(artifact)).fallback_service(ServeDir::new("web/dist")).layer(axum::middleware::from_fn(local_only)).with_state(app);
        // APOLLON_PORT moves the workspace to another local port, for example to run two servers side by side
        let port=std::env::var("APOLLON_PORT").ok().and_then(|p| p.parse::<u16>().ok()).unwrap_or(8765);
        let listener=tokio::net::TcpListener::bind(("127.0.0.1",port)).await.map_err(|e|e.to_string())?;
        println!("Apollon workspace: http://127.0.0.1:{port}");
        axum::serve(listener,router).await.map_err(|e|e.to_string())
    })
}
