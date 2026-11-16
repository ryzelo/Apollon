// apollon GPU kernels – one source for Metal (MSL) and CUDA (NVRTC).
//
// dequant: unpacks rows of a weight matrix into float, exactly as src/quant.rs and tensor.rs do on the
//          CPU (all GGML block formats, F32/F16/BF16, MLX affine 2–8 bit).
// gemm_nt: C[i*ldc + j] = Σ_k A[i*lda + k] · B[j*ldb + k]. Every output is summed over k in the same
//          order whatever the size of the batch, so a row gives the same result alone or among others.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define GLOBAL device
#define SHARED threadgroup
#define INLINE inline
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define AS_FLOAT(x) as_type<float>(x)
#define CONST_TABLE constant
#define F2(a, b) float2((a), (b))
typedef char i8;
#else
#define GLOBAL
#define SHARED
#define INLINE __device__ __forceinline__
#define BARRIER() __syncthreads()
#define AS_FLOAT(x) __uint_as_float(x)
#define CONST_TABLE __constant__
#define F2(a, b) make_float2((a), (b))
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long long ulong;
typedef signed char i8;
#endif

// formats (same numbers as gpu/mod.rs)
#define FMT_F32 0
#define FMT_F16 1
#define FMT_BF16 2
#define FMT_Q4_0 3
#define FMT_Q4_1 4
#define FMT_Q5_0 5
#define FMT_Q5_1 6
#define FMT_Q8_0 7
#define FMT_Q2K 8
#define FMT_Q3K 9
#define FMT_Q4K 10
#define FMT_Q5K 11
#define FMT_Q6K 12
#define FMT_IQ4NL 13
#define FMT_IQ4XS 14
#define FMT_MXFP4 15
#define FMT_MLX 100

struct DqP {
    ulong off_w, off_s, off_b; // byte offsets into the weight, scale and bias buffers
    uint fmt, cols, row_bytes, row0, nrows;
    uint chunk;     // values unpacked by one thread (a GGML block, an MLX group, 32 plain values)
    uint blk_bytes; // bytes of one GGML block
    uint wpr, ng, bits, gs, sdt, bdt, has_b; // MLX: u32 words per row, groups, bits, group size, dtypes
};

struct GemmP {
    ulong a_off, b_off, c_off; // in floats
    uint M, N, K, lda, ldb, ldc;
};

INLINE uint rd16(GLOBAL const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8); }
INLINE uint rd32(GLOBAL const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24); }

// f16 -> f32 bit by bit, as quant::f16 (exact, subnormals included)
INLINE float h2f(uint h) {
    uint sign = (h >> 15) << 31;
    uint e = (h >> 10) & 0x1f;
    uint m = h & 0x3ff;
    uint bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            uint ee = 127 - 15 + 1;
            while ((m & 0x400) == 0) {
                m <<= 1;
                ee -= 1;
            }
            bits = sign | (ee << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    return AS_FLOAT(bits);
}

INLINE float hval(GLOBAL const uchar* b, uint i) { return h2f(rd16(b + i)); }

// one scalar of type dt (0 F32, 1 F16, 2 BF16)
INLINE float scal(GLOBAL const uchar* b, uint dt, ulong i) {
    if (dt == 0) return AS_FLOAT(rd32(b + 4 * i));
    if (dt == 1) return h2f(rd16(b + 2 * i));
    return AS_FLOAT(rd16(b + 2 * i) << 16);
}

CONST_TABLE float KV_IQ4NL[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
CONST_TABLE float KV_MXFP4[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

INLINE void dq_q4_0(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0);
    for (uint j = 0; j < 16; j++) {
        uint q = b[2 + j];
        y[j] = (float)((int)(q & 0xF) - 8) * d;
        y[j + 16] = (float)((int)(q >> 4) - 8) * d;
    }
}

INLINE void dq_q4_1(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0), m = hval(b, 2);
    for (uint j = 0; j < 16; j++) {
        uint q = b[4 + j];
        y[j] = (float)(q & 0xF) * d + m;
        y[j + 16] = (float)(q >> 4) * d + m;
    }
}

INLINE void dq_q5_0(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0);
    uint qh = rd32(b + 2);
    for (uint j = 0; j < 16; j++) {
        uint q = b[6 + j];
        uint xh0 = ((qh >> j) << 4) & 0x10;
        uint xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = (float)((int)((q & 0xF) | xh0) - 16) * d;
        y[j + 16] = (float)((int)((q >> 4) | xh1) - 16) * d;
    }
}

INLINE void dq_q5_1(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0), m = hval(b, 2);
    uint qh = rd32(b + 4);
    for (uint j = 0; j < 16; j++) {
        uint q = b[8 + j];
        uint xh0 = ((qh >> j) << 4) & 0x10;
        uint xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = (float)((q & 0xF) | xh0) * d + m;
        y[j + 16] = (float)((q >> 4) | xh1) * d + m;
    }
}

INLINE void dq_q8_0(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0);
    for (uint j = 0; j < 32; j++) y[j] = (float)((i8)b[2 + j]) * d;
}

INLINE void dq_q2k(GLOBAL const uchar* b, GLOBAL float* y) {
    GLOBAL const uchar* sc = b;
    GLOBAL const uchar* qs = b + 16;
    float d = hval(b, 80), dmin = hval(b, 82);
    uint is = 0, o = 0;
    for (uint n = 0; n < 2; n++) {
        GLOBAL const uchar* q = qs + 32 * n;
        uint shift = 0;
        for (uint k = 0; k < 4; k++) {
            for (uint hh = 0; hh < 2; hh++) {
                uint s = sc[is++];
                float dl = d * (float)(s & 0xF), ml = dmin * (float)(s >> 4);
                for (uint l = 0; l < 16; l++) y[o++] = dl * (float)((q[l + 16 * hh] >> shift) & 3) - ml;
            }
            shift += 2;
        }
    }
}

INLINE void dq_q3k(GLOBAL const uchar* b, GLOBAL float* y) {
    GLOBAL const uchar* hm = b;
    GLOBAL const uchar* qs = b + 32;
    float d = hval(b, 108);
    uint a0 = rd32(b + 96), a1 = rd32(b + 100), tmp = rd32(b + 104);
    uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint aux[4];
    aux[0] = (a0 & k2) | ((tmp & k1) << 4);
    aux[1] = (a1 & k2) | (((tmp >> 2) & k1) << 4);
    aux[2] = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    uint is = 0, o = 0, m = 1;
    for (uint n = 0; n < 2; n++) {
        GLOBAL const uchar* q = qs + 32 * n;
        uint shift = 0;
        for (uint k = 0; k < 4; k++) {
            for (uint hh = 0; hh < 2; hh++) {
                int sc = (int)(i8)((aux[is / 4] >> (8 * (is % 4))) & 0xFF);
                float dl = d * (float)(sc - 32);
                is++;
                for (uint l = 0; l < 16; l++) {
                    uint li = l + 16 * hh;
                    int v = (int)((q[li] >> shift) & 3) - ((hm[li] & m) != 0 ? 0 : 4);
                    y[o++] = dl * (float)v;
                }
            }
            shift += 2;
            m = (m << 1) & 0xFF;
        }
    }
}

INLINE float2 smk4(uint j, GLOBAL const uchar* q) {
    if (j < 4) return F2((float)(q[j] & 63), (float)(q[j + 4] & 63));
    return F2((float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)));
}

INLINE void dq_q4k(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0), dmin = hval(b, 2);
    GLOBAL const uchar* sc = b + 4;
    GLOBAL const uchar* qs = b + 16;
    for (uint j = 0; j < 4; j++) {
        float2 a = smk4(2 * j, sc), c = smk4(2 * j + 1, sc);
        float d1 = d * a.x, m1 = dmin * a.y, d2 = d * c.x, m2 = dmin * c.y;
        GLOBAL const uchar* q = qs + 32 * j;
        for (uint l = 0; l < 32; l++) {
            y[64 * j + l] = d1 * (float)(q[l] & 0xF) - m1;
            y[64 * j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
        }
    }
}

INLINE void dq_q5k(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0), dmin = hval(b, 2);
    GLOBAL const uchar* sc = b + 4;
    GLOBAL const uchar* qh = b + 16;
    GLOBAL const uchar* qs = b + 48;
    uint u1 = 1, u2 = 2;
    for (uint j = 0; j < 4; j++) {
        float2 a = smk4(2 * j, sc), c = smk4(2 * j + 1, sc);
        float d1 = d * a.x, m1 = dmin * a.y, d2 = d * c.x, m2 = dmin * c.y;
        GLOBAL const uchar* q = qs + 32 * j;
        for (uint l = 0; l < 32; l++) {
            int hi1 = (qh[l] & u1) != 0 ? 16 : 0;
            int hi2 = (qh[l] & u2) != 0 ? 16 : 0;
            y[64 * j + l] = d1 * (float)((int)(q[l] & 0xF) + hi1) - m1;
            y[64 * j + 32 + l] = d2 * (float)((int)(q[l] >> 4) + hi2) - m2;
        }
        u1 = (u1 << 2) & 0xFF;
        u2 = (u2 << 2) & 0xFF;
    }
}

INLINE void dq_q6k(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 208);
    for (uint n = 0; n < 2; n++) {
        GLOBAL const uchar* ql = b + 64 * n;
        GLOBAL const uchar* qh = b + 128 + 32 * n;
        GLOBAL const uchar* sc = b + 192 + 8 * n;
        uint o = 128 * n;
        for (uint l = 0; l < 32; l++) {
            uint is = l / 16;
            int q1 = (int)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[o + l] = d * (float)((i8)sc[is]) * (float)q1;
            y[o + l + 32] = d * (float)((i8)sc[is + 2]) * (float)q2;
            y[o + l + 64] = d * (float)((i8)sc[is + 4]) * (float)q3;
            y[o + l + 96] = d * (float)((i8)sc[is + 6]) * (float)q4;
        }
    }
}

INLINE void dq_iq4nl(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0);
    for (uint j = 0; j < 16; j++) {
        y[j] = d * KV_IQ4NL[b[2 + j] & 0xF];
        y[j + 16] = d * KV_IQ4NL[b[2 + j] >> 4];
    }
}

INLINE void dq_iq4xs(GLOBAL const uchar* b, GLOBAL float* y) {
    float d = hval(b, 0);
    uint sh = rd16(b + 2);
    GLOBAL const uchar* sl = b + 4;
    GLOBAL const uchar* qs = b + 8;
    for (uint ib = 0; ib < 8; ib++) {
        int ls = (int)((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | ((int)((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        for (uint j = 0; j < 16; j++) {
            uint q = qs[16 * ib + j];
            y[32 * ib + j] = dl * KV_IQ4NL[q & 0xF];
            y[32 * ib + 16 + j] = dl * KV_IQ4NL[q >> 4];
        }
    }
}

INLINE void dq_mxfp4(GLOBAL const uchar* b, GLOBAL float* y) {
    // 2^(e-128), built bit by bit (also the two subnormal cases e = 0, 1)
    uint e = b[0];
    uint bits = e >= 2 ? (e - 1) << 23 : (0x00400000u >> (1 - e));
    float d = AS_FLOAT(bits);
    for (uint j = 0; j < 16; j++) {
        y[j] = KV_MXFP4[b[1 + j] & 0xF] * d;
        y[j + 16] = KV_MXFP4[b[1 + j] >> 4] * d;
    }
}

INLINE void dequant_body(GLOBAL const uchar* W, GLOBAL const uchar* S, GLOBAL const uchar* B, GLOBAL float* out, DqP p, uint gid) {
    uint per_row = p.fmt == FMT_MLX ? p.ng : (p.cols + p.chunk - 1) / p.chunk;
    uint r = gid / per_row, c = gid % per_row;
    if (r >= p.nrows) return;
    ulong row = (ulong)p.row0 + r;
    GLOBAL float* y = out + (ulong)r * p.cols;
    if (p.fmt == FMT_MLX) {
        GLOBAL const uchar* w = W + p.off_w + row * p.wpr * 4;
        float s = scal(S + p.off_s, p.sdt, row * p.ng + c);
        float bias = p.has_b != 0 ? scal(B + p.off_b, p.bdt, row * p.ng + c) : 0.0f;
        uint mask = (1u << p.bits) - 1;
        for (uint col = c * p.gs; col < (c + 1) * p.gs; col++) {
            uint bit = col * p.bits;
            uint k = bit / 32, sh = bit % 32;
            uint v = rd32(w + 4 * k) >> sh;
            if (sh + p.bits > 32) v |= rd32(w + 4 * (k + 1)) << (32 - sh);
            y[col] = s * (float)(v & mask) + bias;
        }
        return;
    }
    GLOBAL const uchar* rw = W + p.off_w + row * p.row_bytes;
    if (p.fmt <= FMT_BF16) {
        uint c0 = c * p.chunk, c1 = min(c0 + p.chunk, p.cols);
        for (uint col = c0; col < c1; col++) {
            if (p.fmt == FMT_F32) y[col] = AS_FLOAT(rd32(rw + 4 * col));
            else if (p.fmt == FMT_F16) y[col] = h2f(rd16(rw + 2 * col));
            else y[col] = AS_FLOAT(rd16(rw + 2 * col) << 16);
        }
        return;
    }
    GLOBAL const uchar* blk = rw + (ulong)c * p.blk_bytes;
    GLOBAL float* yy = y + (ulong)c * p.chunk;
    switch (p.fmt) {
        case FMT_Q4_0: dq_q4_0(blk, yy); break;
        case FMT_Q4_1: dq_q4_1(blk, yy); break;
        case FMT_Q5_0: dq_q5_0(blk, yy); break;
        case FMT_Q5_1: dq_q5_1(blk, yy); break;
        case FMT_Q8_0: dq_q8_0(blk, yy); break;
        case FMT_Q2K: dq_q2k(blk, yy); break;
        case FMT_Q3K: dq_q3k(blk, yy); break;
        case FMT_Q4K: dq_q4k(blk, yy); break;
        case FMT_Q5K: dq_q5k(blk, yy); break;
        case FMT_Q6K: dq_q6k(blk, yy); break;
        case FMT_IQ4NL: dq_iq4nl(blk, yy); break;
        case FMT_IQ4XS: dq_iq4xs(blk, yy); break;
        case FMT_MXFP4: dq_mxfp4(blk, yy); break;
        default: break;
    }
}

// tiles: 64 × 64 outputs per group, 16 values of k at a time, 16 × 16 threads with 4 × 4 outputs each
#define TM 64
#define TN 64
#define TK 16

INLINE void gemm_body(GLOBAL const float* A, GLOBAL const float* B, GLOBAL float* C, GemmP p, uint bx, uint by, uint tx, uint ty, SHARED float* As, SHARED float* Bs) {
    uint i0 = by * TM, j0 = bx * TN;
    uint tid = ty * 16 + tx;
    float acc[4][4];
    for (uint q = 0; q < 4; q++)
        for (uint w = 0; w < 4; w++) acc[q][w] = 0.0f;
    A += p.a_off;
    B += p.b_off;
    C += p.c_off;
    for (uint k0 = 0; k0 < p.K; k0 += TK) {
        for (uint e = tid; e < TM * TK; e += 256) {
            uint ii = e / TK, kk = e % TK;
            uint gi = i0 + ii, gk = k0 + kk;
            As[kk * TM + ii] = (gi < p.M && gk < p.K) ? A[(ulong)gi * p.lda + gk] : 0.0f;
        }
        for (uint e = tid; e < TN * TK; e += 256) {
            uint jj = e / TK, kk = e % TK;
            uint gj = j0 + jj, gk = k0 + kk;
            Bs[kk * TN + jj] = (gj < p.N && gk < p.K) ? B[(ulong)gj * p.ldb + gk] : 0.0f;
        }
        BARRIER();
        for (uint kk = 0; kk < TK; kk++) {
            float a[4], b[4];
            for (uint q = 0; q < 4; q++) {
                a[q] = As[kk * TM + ty * 4 + q];
                b[q] = Bs[kk * TN + tx * 4 + q];
            }
            for (uint q = 0; q < 4; q++)
                for (uint w = 0; w < 4; w++) acc[q][w] += a[q] * b[w];
        }
        BARRIER();
    }
    for (uint q = 0; q < 4; q++) {
        uint gi = i0 + ty * 4 + q;
        if (gi >= p.M) continue;
        for (uint w = 0; w < 4; w++) {
            uint gj = j0 + tx * 4 + w;
            if (gj < p.N) C[(ulong)gi * p.ldc + gj] = acc[q][w];
        }
    }
}

#ifdef __METAL_VERSION__
// Metal: the same product on the simdgroup matrix units. 4 simdgroups per group, 64 × 64 outputs, k in
// steps of 32; every output is again summed over k in a fixed order (independent of the batch size).
#include <metal_simdgroup_matrix>
kernel void gemm_sg(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[2 * 64 * 32];
    threadgroup float* As = sh;           // [64 m][32 k]
    threadgroup float* Bs = sh + 64 * 32; // [32 k][64 n]
    uint i0 = tg.y * 64, j0 = tg.x * 64;
    uint sr = sgi / 2, sc = sgi % 2;
    device const float* a = A + p.a_off;
    device const float* b = B + p.b_off;
    simdgroup_float8x8 acc[4][4];
    _Pragma("unroll") for (uint i = 0; i < 4; i++)
        _Pragma("unroll") for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.K; k0 += 32) {
        for (uint e = tid; e < 64 * 32; e += 128) {
            uint r = e / 32, kk = e % 32;
            uint gk = k0 + kk;
            uint gi = i0 + r, gj = j0 + r;
            As[e] = (gi < p.M && gk < p.K) ? a[(ulong)gi * p.lda + gk] : 0.0f;
            Bs[kk * 64 + r] = (gj < p.N && gk < p.K) ? b[(ulong)gj * p.ldb + gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 am[4], bm[4];
            _Pragma("unroll") for (uint i = 0; i < 4; i++) simdgroup_load(am[i], As + (sr * 32 + i * 8) * 32 + kk, 32);
            _Pragma("unroll") for (uint j = 0; j < 4; j++) simdgroup_load(bm[j], Bs + kk * 64 + sc * 32 + j * 8, 64);
            _Pragma("unroll") for (uint i = 0; i < 4; i++)
                _Pragma("unroll") for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // through threadgroup memory, so that the edges can be cut off
    _Pragma("unroll") for (uint i = 0; i < 4; i++)
        _Pragma("unroll") for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], sh + sgi * 1024 + (i * 8) * 32 + j * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* c = C + p.c_off;
    for (uint e = tid; e < 4096; e += 128) {
        uint g = e / 1024, l = e % 1024;
        uint gi = i0 + (g / 2) * 32 + l / 32, gj = j0 + (g % 2) * 32 + l % 32;
        if (gi < p.M && gj < p.N) c[(ulong)gi * p.ldc + gj] = sh[e];
    }
}

kernel void dequant(device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]], device const uchar* Bb [[buffer(2)]], device float* out [[buffer(3)]],
                    constant DqP& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    dequant_body(W, S, Bb, out, p, gid);
}
kernel void gemm_nt(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[TK * TM];
    threadgroup float Bs[TK * TN];
    gemm_body(A, B, C, p, tg.x, tg.y, lid.x, lid.y, As, Bs);
}
#else
extern "C" __global__ void dequant(const uchar* W, const uchar* S, const uchar* Bb, float* out, DqP p) {
    dequant_body(W, S, Bb, out, p, blockIdx.x * blockDim.x + threadIdx.x);
}
extern "C" __global__ void gemm_nt(const float* A, const float* B, float* C, GemmP p) {
    __shared__ float As[TK * TM];
    __shared__ float Bs[TK * TN];
    gemm_body(A, B, C, p, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, As, Bs);
}
#endif
