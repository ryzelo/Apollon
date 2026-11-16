//! Reading GGUF: metadata + tensor directory. The file is only memory-mapped.
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Val {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    A(Vec<Val>),
}

impl Val {
    pub fn u(&self) -> Option<u64> {
        match self {
            Val::U(v) => Some(*v),
            Val::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn f(&self) -> Option<f64> {
        match self {
            Val::F(v) => Some(*v),
            Val::U(v) => Some(*v as f64),
            Val::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn s(&self) -> Option<&str> {
        if let Val::S(s) = self { Some(s) } else { None }
    }
    pub fn a(&self) -> Option<&[Val]> {
        if let Val::A(a) = self { Some(a) } else { None }
    }
    pub fn b(&self) -> Option<bool> {
        match self {
            Val::B(b) => Some(*b),
            Val::U(v) => Some(*v != 0),
            _ => None,
        }
    }
}

pub struct TInfo {
    pub ne: Vec<usize>,
    pub typ: u32,
    pub off: usize,
}

pub struct Gguf {
    pub data: &'static [u8],
    pub meta: HashMap<String, Val>,
    pub tensors: HashMap<String, TInfo>,
}

/// Memory-map a file; the mapping lives until the program exits.
pub fn map_file(path: &Path) -> Result<&'static [u8], String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{}: {e}", path.display()))?;
    let m: &'static memmap2::Mmap = Box::leak(Box::new(m));
    crate::gpu::register_region(&m[..]);
    Ok(&m[..])
}

struct Rd {
    b: &'static [u8],
    p: usize,
}

impl Rd {
    fn take(&mut self, n: usize) -> Result<&'static [u8], String> {
        if self.p + n > self.b.len() {
            return Err("GGUF: file ends too early".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32) -> Result<Val, String> {
        Ok(match t {
            0 => Val::U(self.take(1)?[0] as u64),
            1 => Val::I(self.take(1)?[0] as i8 as i64),
            2 => Val::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Val::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Val::U(self.u32()? as u64),
            5 => Val::I(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            6 => Val::F(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Val::B(self.take(1)?[0] != 0),
            8 => Val::S(self.string()?),
            9 => {
                let at = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.val(at)?);
                }
                Val::A(v)
            }
            10 => Val::U(self.u64()?),
            11 => Val::I(self.u64()? as i64),
            12 => Val::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(format!("GGUF: unknown value type {t}")),
        })
    }
}

pub fn open(path: &Path) -> Result<Gguf, String> {
    let data = map_file(path)?;
    let mut r = Rd { b: data, p: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(format!("{}: not a GGUF file", path.display()));
    }
    let ver = r.u32()?;
    if ver < 2 {
        return Err("GGUF version 1 is not supported".into());
    }
    let nt = r.u64()? as usize;
    let nkv = r.u64()? as usize;
    let mut meta = HashMap::new();
    for _ in 0..nkv {
        let k = r.string()?;
        let t = r.u32()?;
        let v = r.val(t)?;
        meta.insert(k, v);
    }
    let mut infos = Vec::with_capacity(nt);
    for _ in 0..nt {
        let name = r.string()?;
        let nd = r.u32()? as usize;
        let mut ne = Vec::with_capacity(nd);
        for _ in 0..nd {
            ne.push(r.u64()? as usize);
        }
        let typ = r.u32()?;
        let off = r.u64()? as usize;
        infos.push((name, ne, typ, off));
    }
    let align = meta.get("general.alignment").and_then(|v| v.u()).unwrap_or(32) as usize;
    let base = r.p.div_ceil(align) * align;
    let tensors = infos
        .into_iter()
        .map(|(n, ne, typ, off)| (n, TInfo { ne, typ, off: base + off }))
        .collect();
    Ok(Gguf { data, meta, tensors })
}
