//! Tokenizer from GGUF metadata or tokenizer.json (Hugging Face).
//! Supported: byte-level BPE (GPT-2/Qwen/Llama-3/GPT-NeoX), SentencePiece BPE
//! with merges (Llama-2 tokenizer.json) and SentencePiece with scores (GGUF "llama").
use crate::gguf::Val;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    ByteLevel,
    SpmMerges,
    SpmScores,
}

pub struct Tokenizer {
    pub id_to_tok: Vec<String>,
    tok_to_id: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    scores: Vec<f32>,
    mode: Mode,
    pre: Option<Regex>,
    specials: Vec<(String, u32)>,
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    nfc: bool,
    prefix_space: bool,
    metaspace_split: bool,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    byte_tok: [Option<u32>; 256],
    pub info: String,
}

/// "a b" -> ("a", "b"), split at the first space from position 1 on (as in llama.cpp)
fn split_merge(m: &str) -> Option<(&str, &str)> {
    let start = m.char_indices().nth(1)?.0;
    let pos = m[start..].find(' ')? + start;
    Some((&m[..pos], &m[pos + 1..]))
}

fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    map
}

impl Tokenizer {
    fn base(id_to_tok: Vec<String>, mode: Mode) -> Self {
        let tok_to_id: HashMap<String, u32> = id_to_tok.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
        let mut byte_tok = [None; 256];
        for (b, slot) in byte_tok.iter_mut().enumerate() {
            *slot = tok_to_id.get(&format!("<0x{b:02X}>")).copied();
        }
        Tokenizer {
            id_to_tok,
            tok_to_id,
            merges: HashMap::new(),
            scores: Vec::new(),
            mode,
            pre: None,
            specials: Vec::new(),
            prefix: Vec::new(),
            suffix: Vec::new(),
            nfc: false,
            prefix_space: false,
            metaspace_split: false,
            byte_enc,
            byte_dec,
            byte_tok,
            info: String::new(),
        }
    }

    /// Prepend a token unless the tokenizer already does it (e.g. <bos> for Gemma)
    pub fn ensure_prefix(&mut self, tok: &str) -> bool {
        if !self.prefix.is_empty() {
            return false;
        }
        match self.tok_to_id.get(tok) {
            Some(&id) => {
                self.prefix.push(id);
                self.info.push_str(&format!(", {tok} added"));
                true
            }
            None => false,
        }
    }

    pub fn has_token(&self, s: &str) -> bool {
        self.tok_to_id.contains_key(s)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.specials.iter().any(|(_, i)| *i == id)
    }

    pub fn placeholder() -> Self {
        Self::base(Vec::new(), Mode::ByteLevel)
    }

    fn add_merges<'a>(&mut self, merges: impl Iterator<Item = (&'a str, &'a str)>) {
        for (rank, (a, b)) in merges.enumerate() {
            let (Some(&ia), Some(&ib)) = (self.tok_to_id.get(a), self.tok_to_id.get(b)) else { continue };
            let Some(&im) = self.tok_to_id.get(&format!("{a}{b}")) else { continue };
            self.merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
    }

    fn set_specials(&mut self, mut sp: Vec<(String, u32)>) {
        sp.retain(|(s, _)| !s.is_empty());
        sp.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        self.specials = sp;
    }

    pub fn from_gguf(meta: &HashMap<String, Val>) -> Result<Self, String> {
        let model = meta.get("tokenizer.ggml.model").and_then(|v| v.s()).unwrap_or("");
        let toks: Vec<String> = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.a())
            .ok_or("GGUF contains no tokenizer")?
            .iter()
            .map(|v| v.s().unwrap_or("").to_string())
            .collect();
        let types: Vec<u64> = meta
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.a())
            .map(|a| a.iter().map(|v| v.u().unwrap_or(1)).collect())
            .unwrap_or_default();
        let mut t = match model {
            "gpt2" => {
                let mut t = Self::base(toks, Mode::ByteLevel);
                let merges: Vec<String> = meta
                    .get("tokenizer.ggml.merges")
                    .and_then(|v| v.a())
                    .ok_or("GGUF: BPE without merges")?
                    .iter()
                    .map(|v| v.s().unwrap_or("").to_string())
                    .collect();
                t.add_merges(merges.iter().filter_map(|m| split_merge(m)));
                let pre = meta.get("tokenizer.ggml.pre").and_then(|v| v.s()).unwrap_or("default");
                let pat = match pre {
                    "qwen2" => QWEN2,
                    "llama-bpe" | "llama3" | "smaug-bpe" => LLAMA3,
                    _ => GPT2,
                };
                t.info = format!("byte-level BPE (pre: {pre}{})", if pat == GPT2 && !matches!(pre, "default" | "gpt-2" | "gpt2") { ", unknown regex -> GPT-2" } else { "" });
                t.pre = Some(Regex::new(pat).map_err(|e| e.to_string())?);
                t
            }
            "llama" => {
                let mut t = Self::base(toks, Mode::SpmScores);
                t.scores = meta
                    .get("tokenizer.ggml.scores")
                    .and_then(|v| v.a())
                    .map(|a| a.iter().map(|v| v.f().unwrap_or(0.0) as f32).collect())
                    .unwrap_or_default();
                t.prefix_space = meta.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.b()).unwrap_or(true);
                t.info = "SentencePiece (Scores)".into();
                t
            }
            "gemma4" | "gemma3" | "gemma" => {
                let mut t = Self::base(toks, Mode::SpmMerges);
                let merges: Vec<String> = meta
                    .get("tokenizer.ggml.merges")
                    .and_then(|v| v.a())
                    .ok_or("GGUF: SentencePiece BPE without merges")?
                    .iter()
                    .map(|v| v.s().unwrap_or("").to_string())
                    .collect();
                t.add_merges(merges.iter().filter_map(|m| split_merge(m)));
                t.prefix_space = meta.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.b()).unwrap_or(false);
                t.info = format!("SentencePiece BPE with merges ({model})");
                t
            }
            other => return Err(format!("tokenizer type '{other}' is not supported")),
        };
        let sp = types
            .iter()
            .enumerate()
            .filter(|(_, ty)| **ty == 3 || **ty == 4)
            .map(|(i, _)| (t.id_to_tok[i].clone(), i as u32))
            .collect();
        t.set_specials(sp);
        let add_bos = meta.get("tokenizer.ggml.add_bos_token").and_then(|v| v.b()).unwrap_or(t.mode == Mode::SpmScores);
        if add_bos {
            t.prefix = meta.get("tokenizer.ggml.bos_token_id").and_then(|v| v.u()).map(|v| vec![v as u32]).unwrap_or_default();
        }
        if meta.get("tokenizer.ggml.add_eos_token").and_then(|v| v.b()) == Some(true) {
            t.suffix = meta.get("tokenizer.ggml.eos_token_id").and_then(|v| v.u()).map(|v| vec![v as u32]).unwrap_or_default();
        }
        Ok(t)
    }

    pub fn from_json(j: &Value, cfg: Option<&Value>) -> Result<Self, String> {
        let m = &j["model"];
        if m["type"].as_str() != Some("BPE") {
            return Err(format!("tokenizer.json: model type {:?} is not supported", m["type"]));
        }
        let vocab = m["vocab"].as_object().ok_or("tokenizer.json: no vocabulary")?;
        let mut added: Vec<(String, u32)> = j["added_tokens"]
            .as_array()
            .map(|a| a.iter().filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32))).collect())
            .unwrap_or_default();
        let n = vocab.values().filter_map(|v| v.as_u64()).max().unwrap_or(0).max(added.iter().map(|a| a.1 as u64).max().unwrap_or(0)) as usize + 1;
        let mut toks = vec![String::new(); n];
        for (k, v) in vocab {
            if let Some(i) = v.as_u64() {
                toks[i as usize] = k.clone();
            }
        }
        for (s, i) in &added {
            toks[*i as usize] = s.clone();
        }
        // determine normalizer / pre-tokenizer
        let mut flat = Vec::new();
        fn collect<'a>(v: &'a Value, key: &str, out: &mut Vec<&'a Value>) {
            if v.is_null() {
                return;
            }
            if let Some(list) = v[key].as_array() {
                for x in list {
                    collect(x, key, out);
                }
            } else {
                out.push(v);
            }
        }
        collect(&j["normalizer"], "normalizers", &mut flat);
        let mut pres = Vec::new();
        collect(&j["pre_tokenizer"], "pretokenizers", &mut pres);
        let spm_norm = flat.iter().any(|n| n["type"] == "Replace" && n["content"].as_str() == Some("\u{2581}"));
        let metaspace = pres.iter().find(|p| p["type"] == "Metaspace");
        let mode = if spm_norm || metaspace.is_some() { Mode::SpmMerges } else { Mode::ByteLevel };
        let mut t = Self::base(toks, mode);
        t.nfc = flat.iter().any(|n| n["type"] == "NFC");
        let merges = m["merges"].as_array().ok_or("tokenizer.json: no merges")?;
        let pairs: Vec<(String, String)> = merges
            .iter()
            .filter_map(|x| {
                if let Some(s) = x.as_str() {
                    s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))
                } else {
                    let a = x.as_array()?;
                    Some((a.first()?.as_str()?.to_string(), a.get(1)?.as_str()?.to_string()))
                }
            })
            .collect();
        t.add_merges(pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())));
        if mode == Mode::ByteLevel {
            let split = pres.iter().find(|p| p["type"] == "Split").and_then(|p| p["pattern"]["Regex"].as_str());
            let bl = pres.iter().find(|p| p["type"] == "ByteLevel");
            let pat = match (split, bl) {
                (Some(p), _) => Some(p.to_string()),
                (None, Some(b)) if b["use_regex"].as_bool().unwrap_or(true) => Some(GPT2.to_string()),
                _ => None,
            };
            if let Some(b) = bl {
                t.prefix_space = b["add_prefix_space"].as_bool().unwrap_or(false);
            }
            if let Some(p) = &pat {
                t.pre = Some(Regex::new(p).map_err(|e| format!("regex from tokenizer.json: {e}"))?);
            }
            t.info = "byte-level BPE (tokenizer.json)".into();
        } else {
            t.prefix_space = flat.iter().any(|n| n["type"] == "Prepend")
                || metaspace.map(|m| m["prepend_scheme"].as_str().unwrap_or("always") != "never" && m["add_prefix_space"].as_bool().unwrap_or(true)).unwrap_or(false);
            t.metaspace_split = metaspace.map(|m| m["split"].as_bool().unwrap_or(true)).unwrap_or(false);
            t.info = "SentencePiece BPE (tokenizer.json)".into();
        }
        // BOS from the post-processor or tokenizer_config.json
        let mut posts = Vec::new();
        collect(&j["post_processor"], "processors", &mut posts);
        for p in posts {
            if p["type"] == "TemplateProcessing" {
                if let Some(single) = p["single"].as_array() {
                    let mut after = false;
                    for item in single {
                        if item.get("Sequence").is_some() {
                            after = true;
                        } else if let Some(id) = item["SpecialToken"]["id"].as_str().and_then(|s| t.tok_to_id.get(s).copied()) {
                            if after { t.suffix.push(id) } else { t.prefix.push(id) }
                        }
                    }
                }
            } else if p["type"] == "RobertaProcessing" || p["type"] == "BertProcessing" {
                let get = |k: &str| p[k].as_array().and_then(|a| a.first()).and_then(|v| v.as_str()).and_then(|s| t.tok_to_id.get(s).copied());
                if let (Some(c), Some(sep)) = (get("cls"), get("sep")) {
                    t.prefix = vec![c];
                    t.suffix = vec![sep];
                }
            }
        }
        if let Some(c) = cfg {
            if c["add_bos_token"].as_bool() == Some(false) {
                t.prefix.clear();
            }
        }
        added.retain(|(s, _)| !s.is_empty());
        t.set_specials(added);
        Ok(t)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = self.prefix.clone();
        // split off special/added tokens literally (longest first)
        let mut seg_start = 0;
        let mut i = 0;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let hit = self.specials.iter().find(|(s, _)| text[i..].starts_with(s.as_str()));
            if let Some((s, id)) = hit {
                if seg_start < i {
                    self.encode_plain(&text[seg_start..i], &mut out);
                }
                out.push(*id);
                i += s.len();
                seg_start = i;
            } else {
                i += 1;
            }
        }
        if seg_start < text.len() {
            self.encode_plain(&text[seg_start..], &mut out);
        }
        out.extend(&self.suffix);
        out
    }

    /// GPT-2/RoBERTa style: vocab.json + merges.txt (byte-level BPE)
    pub fn from_vocab_merges(vocab: &Value, merges_txt: &str, specials: &[String], prefix: &[String], suffix: &[String]) -> Result<Self, String> {
        let v = vocab.as_object().ok_or("vocab.json: not an object")?;
        let n = v.values().filter_map(|x| x.as_u64()).max().unwrap_or(0) as usize + 1;
        let mut toks = vec![String::new(); n];
        for (k, i) in v {
            if let Some(i) = i.as_u64() {
                toks[i as usize] = k.clone();
            }
        }
        let mut t = Self::base(toks, Mode::ByteLevel);
        let pairs: Vec<(&str, &str)> = merges_txt.lines().filter(|l| !l.starts_with("#version") && !l.is_empty()).filter_map(split_merge).collect();
        t.add_merges(pairs.into_iter());
        t.pre = Some(Regex::new(GPT2).map_err(|e| e.to_string())?);
        let sp: Vec<(String, u32)> = specials.iter().filter_map(|s| Some((s.clone(), *t.tok_to_id.get(s)?))).collect();
        t.prefix = prefix.iter().filter_map(|s| t.tok_to_id.get(s).copied()).collect();
        t.suffix = suffix.iter().filter_map(|s| t.tok_to_id.get(s).copied()).collect();
        t.set_specials(sp);
        t.info = "byte-level BPE (vocab.json + merges.txt)".into();
        Ok(t)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let text: String = if self.nfc { text.nfc().collect() } else { text.to_string() };
        match self.mode {
            Mode::ByteLevel => {
                let text = if self.prefix_space && !text.starts_with(' ') { format!(" {text}") } else { text };
                let words: Vec<&str> = match &self.pre {
                    Some(re) => re.find_iter(&text).filter_map(|m| m.ok()).map(|m| m.as_str()).collect(),
                    None => vec![text.as_str()],
                };
                for w in words {
                    let syms: Vec<u32> = w
                        .bytes()
                        .filter_map(|b| self.tok_to_id.get(&self.byte_enc[b as usize].to_string()).copied())
                        .collect();
                    out.extend(self.bpe(syms));
                }
            }
            Mode::SpmMerges => {
                let mut s = text.replace(' ', "\u{2581}");
                if self.prefix_space {
                    s.insert(0, '\u{2581}');
                }
                let pieces: Vec<String> = if self.metaspace_split {
                    let mut v = Vec::new();
                    let mut cur = String::new();
                    for c in s.chars() {
                        if c == '\u{2581}' && !cur.is_empty() {
                            v.push(std::mem::take(&mut cur));
                        }
                        cur.push(c);
                    }
                    if !cur.is_empty() {
                        v.push(cur);
                    }
                    v
                } else {
                    vec![s]
                };
                for p in pieces {
                    let mut syms = Vec::new();
                    for c in p.chars() {
                        let cs = c.to_string();
                        match self.tok_to_id.get(&cs) {
                            Some(&id) => syms.push(id),
                            None => {
                                for b in cs.bytes() {
                                    if let Some(id) = self.byte_tok[b as usize] {
                                        syms.push(id);
                                    }
                                }
                            }
                        }
                    }
                    out.extend(self.bpe(syms));
                }
            }
            Mode::SpmScores => {
                let mut s = text.replace(' ', "\u{2581}");
                if self.prefix_space {
                    s.insert(0, '\u{2581}');
                }
                let mut syms: Vec<String> = s.chars().map(|c| c.to_string()).collect();
                loop {
                    let mut best: Option<(f32, usize)> = None;
                    for i in 0..syms.len().saturating_sub(1) {
                        let cat = format!("{}{}", syms[i], syms[i + 1]);
                        if let Some(&id) = self.tok_to_id.get(&cat) {
                            let sc = self.scores.get(id as usize).copied().unwrap_or(0.0);
                            if best.map_or(true, |b| sc > b.0) {
                                best = Some((sc, i));
                            }
                        }
                    }
                    let Some((_, i)) = best else { break };
                    let right = syms.remove(i + 1);
                    syms[i].push_str(&right);
                }
                for s in syms {
                    match self.tok_to_id.get(&s) {
                        Some(&id) => out.push(id),
                        None => {
                            for b in s.bytes() {
                                if let Some(id) = self.byte_tok[b as usize] {
                                    out.push(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn bpe(&self, mut syms: Vec<u32>) -> Vec<u32> {
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, m)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.map_or(true, |b| rank < b.0) {
                        best = Some((rank, i, m));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i, m)) => {
                    syms[i] = m;
                    syms.remove(i + 1);
                }
            }
        }
        syms
    }

    /// Readable text of a token (bytes that do not form a whole character as \xNN)
    pub fn piece(&self, id: u32) -> String {
        let s = self.id_to_tok.get(id as usize).cloned().unwrap_or_else(|| format!("<{id}>"));
        if self.specials.iter().any(|(_, i)| *i == id) {
            return s;
        }
        let bytes: Vec<u8> = match self.mode {
            Mode::ByteLevel => {
                if s.chars().all(|c| self.byte_dec.contains_key(&c)) {
                    s.chars().map(|c| self.byte_dec[&c]).collect()
                } else {
                    s.into_bytes()
                }
            }
            _ => {
                if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
                    u8::from_str_radix(&s[3..5], 16).map(|b| vec![b]).unwrap_or_else(|_| s.clone().into_bytes())
                } else {
                    s.replace('\u{2581}', " ").into_bytes()
                }
            }
        };
        match String::from_utf8(bytes.clone()) {
            Ok(t) => t,
            Err(_) => bytes.iter().map(|b| if b.is_ascii() { (*b as char).to_string() } else { format!("\\x{b:02x}") }).collect(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_tok.len()
    }
}
