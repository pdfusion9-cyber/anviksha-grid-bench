//! How fast is a 24^3 lookup grid when the surrounding loop is also native?
//!
//! The Python measurements were: 1,660 ns for a per-query loop, 9.7 ns for a
//! vectorised gather, 57 ns to just compute the value. The first number is not
//! the grid -- it is the interpreter. This measures the same grid with no
//! interpreter in the way, so the grid's real cost can be seen.
//!
//! Three things are timed:
//!   build    fill 24^3 = 13,824 cells
//!   lookup   read one cell per query, in a tight native loop
//!   compute  calculate the value instead of reading it

use std::time::Instant;

const N: usize = 24;
const LO: f32 = -8.0;
const HI: f32 = 8.0;
const SCALE: f32 = 40.0; // values land in [-3, 3], so int8 has room

#[inline(always)]
fn index(x: usize, y: usize, z: usize) -> usize {
    (x * N + y) * N + z
}

fn main() {
    let mut axis = [0f32; N];
    for i in 0..N {
        axis[i] = LO + (HI - LO) * i as f32 / (N - 1) as f32;
    }

    // ---- build ----------------------------------------------------------
    let t0 = Instant::now();
    let mut grid = vec![0i8; N * N * N];
    let mut tanh_axis = [0f32; N];
    for i in 0..N {
        tanh_axis[i] = axis[i].tanh();
    }
    for x in 0..N {
        for y in 0..N {
            for z in 0..N {
                let v = tanh_axis[x] + tanh_axis[y] + tanh_axis[z];
                grid[index(x, y, z)] = (v * SCALE) as i8;
            }
        }
    }
    let build = t0.elapsed();
    println!(
        "grid {}^3 = {} cells = {} KB",
        N,
        N * N * N,
        grid.len() / 1024
    );
    println!("build   {:>10.3} ms   ({:.1} M cells/s)",
        build.as_secs_f64() * 1e3,
        (N * N * N) as f64 / build.as_secs_f64() / 1e6);

    // ---- queries --------------------------------------------------------
    const Q: usize = 10_000_000;
    let mut qx = vec![0u8; Q];
    let mut qy = vec![0u8; Q];
    let mut qz = vec![0u8; Q];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for i in 0..Q {
        qx[i] = (next() % N as u64) as u8;
        qy[i] = (next() % N as u64) as u8;
        qz[i] = (next() % N as u64) as u8;
    }

    let mut sink: i64 = 0;

    let t0 = Instant::now();
    for i in 0..Q {
        sink += grid[index(qx[i] as usize, qy[i] as usize, qz[i] as usize)] as i64;
    }
    let lookup = t0.elapsed();
    let lookup_ns = lookup.as_secs_f64() / Q as f64 * 1e9;
    println!("lookup  {:>10.2} ns/query   ({:.1} M queries/s)",
        lookup_ns, Q as f64 / lookup.as_secs_f64() / 1e6);

    let t0 = Instant::now();
    for i in 0..Q {
        let v = axis[qx[i] as usize].tanh()
            + axis[qy[i] as usize].tanh()
            + axis[qz[i] as usize].tanh();
        sink += (v * SCALE) as i64;
    }
    let compute = t0.elapsed();
    let compute_ns = compute.as_secs_f64() / Q as f64 * 1e9;
    println!("compute {:>10.2} ns/query   ({:.1} M queries/s)",
        compute_ns, Q as f64 / compute.as_secs_f64() / 1e6);

    println!("\nlookup vs compute: {:.2}x faster", compute_ns / lookup_ns);
    println!("checksum {}", sink);

    // what the same grid costs when the whole batch is processed natively
    let t0 = Instant::now();
    let mut total: i64 = 0;
    for chunk in 0..Q / 64 {
        let mut acc: i64 = 0;
        for i in chunk * 64..(chunk + 1) * 64 {
            acc += grid[index(qx[i] as usize, qy[i] as usize, qz[i] as usize)] as i64;
        }
        total += acc;
    }
    let blocked = t0.elapsed();
    println!(
        "lookup in 64-query blocks: {:.2} ns/query  (checksum {})",
        blocked.as_secs_f64() / Q as f64 * 1e9,
        total
    );
}
