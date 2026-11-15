//! Dequantization (GGML block formats + F32/F16/BF16). Formulas as in ggml-quants.c.
#![allow(clippy::needless_range_loop)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4NL,
    IQ4XS,
    MXFP4,
}
use DType::*;

impl DType {
    pub fn from_ggml(t: u32) -> Option<Self> {
        Some(match t {
            0 => F32,
            1 => F16,
            30 => BF16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            20 => IQ4NL,
            23 => IQ4XS,
            39 => MXFP4,
            _ => return None,
        })
    }
    pub fn from_st(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(F32),
            "F16" => Some(F16),
            "BF16" => Some(BF16),
            _ => None,
        }
    }
    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            F32 => (1, 4),
            F16 | BF16 => (1, 2),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            IQ4NL => (32, 18),
            IQ4XS => (256, 136),
            MXFP4 => (32, 17),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            F32 => "F32",
            F16 => "F16",
            BF16 => "BF16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            IQ4NL => "IQ4_NL",
            IQ4XS => "IQ4_XS",
            MXFP4 => "MXFP4",
        }
    }
}

pub fn ggml_name(t: u32) -> String {
    let names = [
        (0, "F32"), (1, "F16"), (2, "Q4_0"), (3, "Q4_1"), (6, "Q5_0"), (7, "Q5_1"), (8, "Q8_0"), (9, "Q8_1"),
        (10, "Q2_K"), (11, "Q3_K"), (12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K"), (15, "Q8_K"), (16, "IQ2_XXS"),
        (17, "IQ2_XS"), (18, "IQ3_XXS"), (19, "IQ1_S"), (20, "IQ4_NL"), (21, "IQ3_S"), (22, "IQ2_S"),
        (23, "IQ4_XS"), (24, "I8"), (25, "I16"), (26, "I32"), (27, "I64"), (28, "F64"), (29, "IQ1_M"),
        (30, "BF16"), (34, "TQ1_0"), (35, "TQ2_0"), (39, "MXFP4"),
    ];
    names.iter().find(|(i, _)| *i == t).map(|(_, n)| n.to_string()).unwrap_or(format!("type {t}"))
}

#[inline]
pub fn f16(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: u32 = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn h(b: &[u8], i: usize) -> f32 {
    f16(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// src must hold exactly dst.len() elements (in whole blocks).
pub fn dequant(dt: DType, src: &[u8], dst: &mut [f32]) {
    match dt {
        F32 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        F16 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        BF16 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        _ => {
            let (be, bb) = dt.block();
            let f: fn(&[u8], &mut [f32]) = match dt {
                Q4_0 => q4_0,
                Q4_1 => q4_1,
                Q5_0 => q5_0,
                Q5_1 => q5_1,
                Q8_0 => q8_0,
                Q2K => q2_k,
                Q3K => q3_k,
                Q4K => q4_k,
                Q5K => q5_k,
                Q6K => q6_k,
                IQ4NL => iq4_nl,
                IQ4XS => iq4_xs,
                MXFP4 => mxfp4,
                F32 | F16 | BF16 => unreachable!(),
            };
            for (blk, out) in src.chunks_exact(bb).zip(dst.chunks_exact_mut(be)) {
                f(blk, out);
            }
        }
    }
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..16 {
        let q = b[2 + j];
        y[j] = ((q & 0xF) as i32 - 8) as f32 * d;
        y[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    for j in 0..16 {
        let q = b[4 + j];
        y[j] = (q & 0xF) as f32 * d + m;
        y[j + 16] = (q >> 4) as f32 * d + m;
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for j in 0..16 {
        let q = b[6 + j] as u32;
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = (((q & 0xF) | xh0) as i32 - 16) as f32 * d;
        y[j + 16] = (((q >> 4) | xh1) as i32 - 16) as f32 * d;
    }
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for j in 0..16 {
        let q = b[8 + j] as u32;
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = ((q & 0xF) | xh0) as f32 * d + m;
        y[j + 16] = ((q >> 4) | xh1) as f32 * d + m;
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..32 {
        y[j] = (b[2 + j] as i8) as f32 * d;
    }
}

fn q2_k(b: &[u8], y: &mut [f32]) {
    let sc = &b[0..16];
    let qs = &b[16..80];
    let (d, dmin) = (h(b, 80), h(b, 82));
    let (mut is, mut o) = (0, 0);
    for n in 0..2 {
        let q = &qs[32 * n..];
        let mut shift = 0;
        for _ in 0..4 {
            for half in 0..2 {
                let s = sc[is];
                is += 1;
                let (dl, ml) = (d * (s & 0xF) as f32, dmin * (s >> 4) as f32);
                for l in 0..16 {
                    y[o] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                    o += 1;
                }
            }
            shift += 2;
        }
    }
}

fn q3_k(b: &[u8], y: &mut [f32]) {
    let hm = &b[0..32];
    let qs = &b[32..96];
    let d = h(b, 108);
    let rd = |i: usize| u32::from_le_bytes([b[96 + 4 * i], b[97 + 4 * i], b[98 + 4 * i], b[99 + 4 * i]]);
    let (a0, a1, tmp) = (rd(0), rd(1), rd(2));
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let aux = [
        (a0 & k2) | ((tmp & k1) << 4),
        (a1 & k2) | (((tmp >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
    ];
    let mut scales = [0i8; 16];
    for i in 0..4 {
        for (j, byte) in aux[i].to_le_bytes().iter().enumerate() {
            scales[4 * i + j] = *byte as i8;
        }
    }
    let (mut is, mut o, mut m) = (0usize, 0usize, 1u8);
    for n in 0..2 {
        let q = &qs[32 * n..];
        let mut shift = 0;
        for _ in 0..4 {
            for half in 0..2 {
                let dl = d * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let li = l + 16 * half;
                    let v = ((q[li] >> shift) & 3) as i32 - if hm[li] & m != 0 { 0 } else { 4 };
                    y[o] = dl * v as f32;
                    o += 1;
                }
            }
            shift += 2;
            m = m.wrapping_shl(1);
        }
    }
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let qs = &b[16..144];
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1, dmin * m1, d * s2, dmin * m2);
        let q = &qs[32 * j..32 * j + 32];
        for l in 0..32 {
            y[64 * j + l] = d1 * (q[l] & 0xF) as f32 - mm1;
            y[64 * j + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
        }
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    let (mut u1, mut u2) = (1u8, 2u8);
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1, dmin * m1, d * s2, dmin * m2);
        let q = &qs[32 * j..32 * j + 32];
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[64 * j + l] = d1 * ((q[l] & 0xF) as i32 + hi1) as f32 - mm1;
            y[64 * j + 32 + l] = d2 * ((q[l] >> 4) as i32 + hi2) as f32 - mm2;
        }
        u1 = u1.wrapping_shl(2);
        u2 = u2.wrapping_shl(2);
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let ql = &b[64 * n..64 * n + 64];
        let qh = &b[128 + 32 * n..128 + 32 * n + 32];
        let sc = &b[192 + 8 * n..192 + 8 * n + 8];
        let o = 128 * n;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

const KV_IQ4NL: [f32; 16] = [-127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.];
const KV_MXFP4: [f32; 16] = [0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.];

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..16 {
        y[j] = d * KV_IQ4NL[(b[2 + j] & 0xF) as usize];
        y[j + 16] = d * KV_IQ4NL[(b[2 + j] >> 4) as usize];
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]) as u32;
    let sl = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..8 {
        let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            let q = qs[16 * ib + j];
            y[32 * ib + j] = dl * KV_IQ4NL[(q & 0xF) as usize];
            y[32 * ib + 16 + j] = dl * KV_IQ4NL[(q >> 4) as usize];
        }
    }
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    let d = 2f64.powi(b[0] as i32 - 128) as f32;
    for j in 0..16 {
        y[j] = KV_MXFP4[(b[1 + j] & 0xF) as usize] * d;
        y[j + 16] = KV_MXFP4[(b[1 + j] >> 4) as usize] * d;
    }
}
