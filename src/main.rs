//! Lookup grids measured inside a native Rust loop, which is where they would
//! actually live if the model core is Rust and Python is only a wrapper.
//!
//! With no interpreter and no FFI boundary in the way, a lookup costs what the
//! hardware charges: a cache read. This measures three things that a Rust model
//! could plausibly use, against computing the same values inline:
//!
//!   lut1   a 1D table for an activation (tanh, sigmoid, gelu). This is the one
//!          a transformer actually needs, because activations take one input.
//!   grid3  a 24^3 table for a three-input function. Real, but only for the
//!          parts of a model that genuinely take three bounded inputs.
//!   direct computing the value instead, in the same loop
//!
//! Sizes are swept because the whole result depends on whether the table stays
//! in L1. A table that does not fit is a memory round trip, and a memory round
//! trip is slower than arithmetic.

use std::time::Instant;

const Q: usize = 20_000_000;

#[inline(always)]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    // tanh approximation, the one most implementations use
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

struct Lut1 {
    table: Vec<f32>,
    lo: f32,
    scale: f32,
}

impl Lut1 {
    fn build(f: fn(f32) -> f32, lo: f32, hi: f32, n: usize) -> Self {
        let mut table = Vec::with_capacity(n);
        for i in 0..n {
            let x = lo + (hi - lo) * i as f32 / (n - 1) as f32;
            table.push(f(x));
        }
        Lut1 { table, lo, scale: (n - 1) as f32 / (hi - lo) }
    }

    #[inline(always)]
    fn get(&self, x: f32) -> f32 {
        let idx = ((x - self.lo) * self.scale) as usize;
        // clamp instead of branching out of bounds; the table covers [-8, 8]
        let idx = idx.min(self.table.len() - 1);
        self.table[idx]
    }
}

struct Grid3 {
    cells: Vec<f32>,
    lo: f32,
    scale: f32,
    n: usize,
}

impl Grid3 {
    fn build(lo: f32, hi: f32, n: usize) -> Self {
        let mut axis = vec![0f32; n];
        for i in 0..n {
            axis[i] = (lo + (hi - lo) * i as f32 / (n - 1) as f32).tanh();
        }
        let mut cells = vec![0f32; n * n * n];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    cells[(x * n + y) * n + z] = axis[x] + axis[y] + axis[z];
                }
            }
        }
        Grid3 { cells, lo, scale: (n - 1) as f32 / (hi - lo), n }
    }

    #[inline(always)]
    fn get(&self, x: f32, y: f32, z: f32) -> f32 {
        let ix = (((x - self.lo) * self.scale) as usize).min(self.n - 1);
        let iy = (((y - self.lo) * self.scale) as usize).min(self.n - 1);
        let iz = (((z - self.lo) * self.scale) as usize).min(self.n - 1);
        self.cells[(ix * self.n + iy) * self.n + iz]
    }
}

fn main() {
    let mut seed: u64 = 0x2545F4914F6CDD1D;
    let mut xs = vec![0f32; Q];
    for x in xs.iter_mut() {
        let r = xorshift(&mut seed);
        // uniform in [-8, 8]
        *x = ((r >> 11) as f32 / (1u64 << 53) as f32) * 16.0 - 8.0;
    }

    println!("{:>10} {:>10} {:>12} {:>12} {:>10}", "table", "KB", "direct ns", "lookup ns", "speedup");

    // ---- 1D activation tables -------------------------------------------
    let t0 = Instant::now();
    let mut sink = 0f32;
    for &x in &xs {
        sink += x.tanh();
    }
    let tanh_direct = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
    println!("{:>10} {:>10} {:>12.2} {:>12} {:>10}", "tanh inline", "-", tanh_direct, "-", "-");

    for n in (256usize, 4_096, 65_536, 1_048_576) {
        let lut = Lut1::build(f32::tanh, -8.0, 8.0, n);
        let t0 = Instant::now();
        let mut s = 0f32;
        for &x in &xs {
            s += lut.get(x);
        }
        let ns = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
        println!("{:>10} {:>10} {:>12.2} {:>12.2} {:>9.2}x",
            format!("tanh lut{n}"), n * 4 / 1024, tanh_direct, ns, tanh_direct / ns);
        sink += s;
    }

    let t0 = Instant::now();
    for &x in &xs {
        sink += sigmoid(x);
    }
    let sig_direct = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
    let lut = Lut1::build(sigmoid, -8.0, 8.0, 4_096);
    let t0 = Instant::now();
    for &x in &xs {
        sink += lut.get(x);
    }
    let sig_lut = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
    println!("{:>10} {:>10} {:>12.2} {:>12.2} {:>9.2}x",
        "sigmoid lut", 4096 * 4 / 1024, sig_direct, sig_lut, sig_direct / sig_lut);

    let t0 = Instant::now();
    for &x in &xs {
        sink += gelu(x);
    }
    let gelu_direct = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
    let lut = Lut1::build(gelu, -8.0, 8.0, 4_096);
    let t0 = Instant::now();
    for &x in &xs {
        sink += lut.get(x);
    }
    let gelu_lut = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
    println!("{:>10} {:>10} {:>12.2} {:>12.2} {:>9.2x}",
        "gelu lut", 4096 * 4 / 1024, gelu_direct, gelu_lut, gelu_direct / gelu_lut);

    // ---- 3D grid ---------------------------------------------------------
    println!();
    for n in (16usize, 24, 32, 64) {
        let g = Grid3::build(-8.0, 8.0, n);
        let t0 = Instant::now();
        let mut s = 0f32;
        let mut sd: u64 = 1;
        for _ in 0..Q {
            let a = (xorshift(&mut sd) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            let b = (xorshift(&mut sd) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            let c = (xorshift(&mut sd) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            s += g.get(a, b, c);
        }
        let grid_ns = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;

        let t0 = Instant::now();
        let mut sd2: u64 = 1;
        for _ in 0..Q {
            let a = (xorshift(&mut sd2) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            let b = (xorshift(&mut sd2) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            let c = (xorshift(&mut sd2) >> 11) as f32 / (1u64 << 53) as f32 * 16.0 - 8.0;
            s += a.tanh() + b.tanh() + c.tanh();
        }
        let direct_ns = t0.elapsed().as_secs_f64() / Q as f64 * 1e9;
        println!("{:>10} {:>10} {:>12.2} {:>12.2} {:>9.2}x",
            format!("grid{}^3", n), n * n * n * 4 / 1024, direct_ns, grid_ns, direct_ns / grid_ns);
        sink += s;
    }

    println!("\nchecksum {}", sink);
    println!("\nreading: a table that stays in L1 wins; one that spills to RAM does");
    println!("not. The crossover is the L1/L2 size, not the algorithm.");
}
