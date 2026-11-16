//! Terminal styling, number formatting and small HTML helpers shared by the reports.
use crate::model::Model;

pub struct Style {
    pub color: bool,
    pub width: usize,
}

impl Style {
    fn paint(&self, s: &str, code: &str) -> String {
        if self.color { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
    }
    pub fn bold(&self, s: &str) -> String {
        self.paint(s, "1")
    }
    pub fn dim(&self, s: &str) -> String {
        self.paint(s, "2")
    }
    pub fn yellow(&self, s: &str) -> String {
        self.paint(s, "33")
    }
    pub fn green(&self, s: &str) -> String {
        self.paint(s, "32")
    }
    pub fn red(&self, s: &str) -> String {
        self.paint(s, "31")
    }
}

/// Fixed-point number with `p` decimals and a typographic minus sign.
pub fn num(x: f32, p: usize) -> String {
    let s = format!("{:.*}", p, x);
    if s.starts_with('-') { s.replacen('-', "−", 1) } else { s }
}

/// Probability as a percentage, with more decimals for small values.
pub fn pct(p: f32) -> String {
    if p >= 0.0995 {
        format!("{}%", num(p * 100.0, 0))
    } else if p >= 0.001 {
        format!("{}%", num(p * 100.0, 1))
    } else {
        format!("{}%", num(p * 100.0, 2))
    }
}

/// Change of a probability in percentage points: "+9 pp", "−0.3 pp".
pub fn pp(x: f32) -> String {
    let v = x * 100.0;
    let sign = if v > 0.0 { "+" } else { "" };
    if v.abs() >= 9.95 {
        format!("{sign}{} pp", num(v, 0))
    } else {
        format!("{sign}{} pp", num(v, 1))
    }
}

/// 1234567 -> "1,234,567"
pub fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn superscript(e: i32) -> String {
    e.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect()
}

/// e^ln as "m·10ⁿ" (arbitrarily large or small, no overflow).
pub fn sci_exp(ln: f64) -> String {
    let l10 = ln / std::f64::consts::LN_10;
    let mut e = l10.floor();
    let mut mant = 10f64.powf(l10 - e).round();
    if mant >= 10.0 {
        mant = 1.0;
        e += 1.0;
    }
    format!("{}·10{}", mant as i32, superscript(e as i32))
}

/// Short scientific notation for check deviations: "2.1e-7".
pub fn sci(x: f32) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let mut e = x.abs().log10().floor() as i32;
    let mut mant = x / 10f32.powi(e);
    if mant.abs() >= 9.95 {
        e += 1;
        mant /= 10.0;
    }
    format!("{}e{}", num(mant, 1), e)
}

/// Factor e^dl, readable even when huge or tiny: "×3.42", "×1,204", "×9·10¹³", "÷250".
pub fn factor(dl: f32) -> String {
    let f = (dl as f64).exp();
    if f >= 1e6 {
        format!("×{}", sci_exp(dl as f64))
    } else if f >= 10.0 {
        format!("×{}", thousands(f.round() as usize))
    } else if f >= 0.01 {
        format!("×{}", num(f as f32, 2))
    } else if f >= 1e-6 {
        format!("÷{}", thousands((1.0 / f).round() as usize))
    } else {
        format!("÷{}", sci_exp(-dl as f64))
    }
}

/// Make control characters in a token visible.
pub fn esc_piece(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\n' => "⏎".to_string(),
            '\t' => "⇥".to_string(),
            '\r' => "␍".to_string(),
            c if c.is_control() => format!("\\u{:04x}", c as u32),
            c => c.to_string(),
        })
        .collect()
}

/// Token in quotes: "\" Paris\""
pub fn tok(m: &Model, id: u32) -> String {
    format!("\"{}\"", esc_piece(&m.tok.piece(id)))
}

pub fn trunc(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    if c.len() <= n { s.to_string() } else { c[..n.saturating_sub(1)].iter().collect::<String>() + "…" }
}

pub fn pad(s: &str, n: usize) -> String {
    let l = s.chars().count();
    if l >= n { s.to_string() } else { format!("{s}{}", " ".repeat(n - l)) }
}

pub fn padl(s: &str, n: usize) -> String {
    let l = s.chars().count();
    if l >= n { s.to_string() } else { format!("{}{s}", " ".repeat(n - l)) }
}

// ---------------------------------------------------------------------- HTML

pub fn esc_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A token as an HTML chip.
pub fn chip(m: &Model, id: u32) -> String {
    format!("<span class=\"tok\">{}</span>", esc_html(&esc_piece(&m.tok.piece(id))))
}

/// The prompt as a row of chips, the focus token outlined.
pub fn prompt_html(m: &Model, ids: &[u32], focus: usize) -> String {
    ids.iter()
        .enumerate()
        .map(|(j, id)| format!("<span class=\"pt{}\" title=\"[{j}]\">{}</span>", if j == focus { " fo" } else { "" }, esc_html(&esc_piece(&m.tok.piece(*id)))))
        .collect()
}

/// Factor on the odds, colored by direction (more than ×2 up or down).
pub fn factor_html(x: f32) -> String {
    let cls = if x > 0.693 { "pos" } else if x < -0.693 { "neg" } else { "mu" };
    format!("<span class=\"{cls}\">{}</span>", esc_html(&factor(x)))
}

pub const CSS: &str = r#"
:root{--bg:#f8f7f4;--fg:#1c1b19;--mu:#6d6a63;--line:#e3e0d8;--card:#fff;--chip:#efece5;--pos:#1d7a4c;--neg:#b33a2e;--acc:#2f5fb3;--bar:#c8d5ef;--kb:#e9e2cf}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131312;--fg:#ebeae6;--mu:#9b978f;--line:#2d2c29;--card:#1b1b19;--chip:#282724;--pos:#5fc38d;--neg:#ee7d71;--acc:#81a8ee;--bar:#2c4068;--kb:#3a3526}}
:root[data-theme="dark"]{--bg:#131312;--fg:#ebeae6;--mu:#9b978f;--line:#2d2c29;--card:#1b1b19;--chip:#282724;--pos:#5fc38d;--neg:#ee7d71;--acc:#81a8ee;--bar:#2c4068;--kb:#3a3526}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1320px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 6px}
h4{font-size:13px;margin:14px 0 6px;color:var(--mu);font-weight:600}
h5{font-size:12px;margin:12px 0 4px;color:var(--mu);font-weight:600}
.meta,.mu{color:var(--mu)}
.check{color:var(--mu);font-size:12.5px}
.prompt{font-family:ui-monospace,Menlo,monospace;font-size:13px;line-height:1.9;word-break:break-word}
.pt{background:var(--chip);border-radius:3px;padding:1px 2px;margin-right:2px;white-space:pre}
.pt.fo{outline:2px solid var(--acc)}
.tok{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--chip);border-radius:3px;padding:0 4px;white-space:pre}
.st{display:inline-flex;gap:4px;align-items:baseline;margin:0 8px 2px 0;white-space:nowrap}
.x .st{margin-right:6px}
.pc{color:var(--mu);font-size:12px;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
details.info{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:12px 0}
details.info summary{cursor:pointer;font-weight:600}
summary{list-style:none;cursor:pointer}summary::-webkit-details-marker{display:none}
table.tbl{border-collapse:collapse;font-size:13px}table.tbl td,table.tbl th{padding:2px 8px 2px 0;text-align:left}table.tbl th{color:var(--mu);font-weight:600;font-size:12px}
.mb{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.mb span{display:block;height:100%;background:var(--acc)}
.foot{margin-top:16px;font-size:12px}
"#;
