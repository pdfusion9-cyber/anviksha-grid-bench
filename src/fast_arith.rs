//! All four operations, with the dispatch overhead removed.
//!
//! The previous version measured its own abstraction. One generic
//! `recall(op, a, b)` added a four-way match, a range check, an Option and a
//! sentinel comparison to every call -- about 2 ns, which is more than the
//! operation being saved. Every width then measured ~1.0x, and that number was
//! about the dispatcher, not about the table.
//!
//! Two changes remove it:
//!
//!   every cell is valid      subtraction stores negatives, division stores the
//!                            quotient for b>0, so no cell needs a sentinel and
//!                            no lookup needs a validity check
//!
//!   one function per op      no enum, no match, no Option. A lookup is a
//!                            multiply-add and a load, which is what the table
//!                            was supposed to be all along
//!
//! Tables are i32 rather than i64, which halves their footprint and keeps more
//! of them in cache -- the second thing the previous run could not measure.
//!
//! Correctness is checked first, over every cell and over the decomposition
//! path, because a table that is fast and wrong is worse than no table.

use std::hint::black_box;
use std::time::Instant;

pub struct ArithTable {
    pub n: usize,
    add: Vec<i32>,
    sub: Vec<i32>,
    mul: Vec<i32>,
    div: Vec<i32>,
}

impl ArithTable {
    pub fn new(n: usize) -> Self {
        let mut add = vec![0i32; n * n];
        let mut sub = vec![0i32; n * n];
        let mut mul = vec![0i32; n * n];
        let mut div = vec![0i32; n * n];
        for a in 0..n {
            for b in 0..n {
                let i = a * n + b;
                add[i] = (a + b) as i32;
                // negatives are stored too, so every cell is answerable
                sub[i] = a as i32 - b as i32;
                mul[i] = (a * b) as i32;
                // b == 0 stays 0 and is never queried: callers check the divisor
                div[i] = if b == 0 { 0 } else { (a / b) as i32 };
            }
        }
        Self { n, add, sub, mul, div }
    }

    pub fn bytes(&self) -> usize {
        self.add.len() * 4 * std::mem::size_of::<i32>()
    }

    #[inline(always)]
    pub fn add(&self, a: usize, b: usize) -> i32 {
        self.add[a * self.n + b]
    }

    #[inline(always)]
    pub fn sub(&self, a: usize, b: usize) -> i32 {
        self.sub[a * self.n + b]
    }

    #[inline(always)]
    pub fn mul(&self, a: usize, b: usize) -> i32 {
        self.mul[a * self.n + b]
    }

    /// Divisor must be non-zero; the caller checks, as it would for any divide.
    #[inline(always)]
    pub fn div(&self, a: usize, b: usize) -> i32 {
        self.div[a * self.n + b]
    }

    /// Past the table: split into table-sized digits. Multiplication only --
    /// addition and subtraction do not need it, and division past the table is
    /// long division rather than a lookup, so computing it is honest.
    pub fn mul_big(&self, a: u64, b: u64) -> i64 {
        if a < self.n as u64 && b < self.n as u64 {
            return self.mul(a as usize, b as usize) as i64;
        }
        let n = self.n as u64;
        let (ah, al) = ((a / n) as i64, (a % n) as i64);
        let (bh, bl) = ((b / n) as i64, (b % n) as i64);
        ah * bh * (n as i64 * n as i64)
            + ah * bl * n as i64
            + al * bh * n as i64
            + al * bl
    }
}

fn main() {
    println!("==============================================================================");
    println!("arithmetic tables with the dispatch overhead removed");
    println!("==============================================================================\n");

    // ---- correctness: every cell, all four operations ----
    let t = ArithTable::new(100);
    let mut wrong = 0;
    for a in 0..100usize {
        for b in 0..100usize {
            if t.add(a, b) != (a + b) as i32 { wrong += 1; }
            if t.sub(a, b) != (a as i32 - b as i32) { wrong += 1; }
            if t.mul(a, b) != (a * b) as i32 { wrong += 1; }
            if b != 0 && t.div(a, b) != (a / b) as i32 { wrong += 1; }
        }
    }
    println!("every cell against the CPU, all four operations: {wrong} wrong  (zero required)");

    let mut big_wrong = 0;
    for a in 0..500u64 {
        for b in 0..500u64 {
            if t.mul_big(a, b) != (a * b) as i64 { big_wrong += 1; }
        }
    }
    println!("decomposition over 250,000 pairs past the table:  {big_wrong} wrong  (zero required)\n");

    // ---- operands prepared outside the timed region ----
    const N: usize = 20_000_000;
    let idx: Vec<(usize, usize)> = (0..N)
        .map(|i| {
            let b = 1 + ((i * 7919) % 99);
            let a = ((i * 104729) % 99) * b;
            (a % 100, b % 100)
        })
        .collect();

    println!("{:<20} {:>11} {:>11} {:>9}", "operation", "table ns", "cpu ns", "speedup");
    println!("{}", "-".repeat(55));

    let mut checksum = 0i64;

    macro_rules! bench {
        ($name:expr, $tab:expr, $cpu:expr) => {{
            let t0 = Instant::now();
            let mut s = 0i64;
            for &(a, b) in idx.iter() {
                s += black_box($tab(black_box(a), black_box(b))) as i64;
            }
            let tab_ns = t0.elapsed().as_nanos() as f64 / N as f64;

            let t0 = Instant::now();
            let mut s2 = 0i64;
            for &(a, b) in idx.iter() {
                s2 += black_box($cpu(black_box(a), black_box(b))) as i64;
            }
            let cpu_ns = t0.elapsed().as_nanos() as f64 / N as f64;
            checksum += s + s2;
            println!("{:<20} {:>11.3} {:>11.3} {:>8.2}x", $name, tab_ns, cpu_ns, cpu_ns / tab_ns);
        }};
    }

    bench!("addition", |a: usize, b: usize| t.add(a, b),
           |a: usize, b: usize| (a + b) as i32);
    bench!("subtraction", |a: usize, b: usize| t.sub(a, b),
           |a: usize, b: usize| a as i32 - b as i32);
    bench!("multiplication", |a: usize, b: usize| t.mul(a, b),
           |a: usize, b: usize| (a * b) as i32);
    bench!("division", |a: usize, b: usize| t.div(a, b),
           |a: usize, b: usize| if b == 0 { 0 } else { (a / b) as i32 });

    println!("\nchecksum {checksum}");

    // ---- how wide before the table stops paying ----
    println!("\n{}", "=".repeat(76));
    println!("table width against cost, through the specialised lookup only");
    println!("{}", "=".repeat(76));
    println!("{:<8} {:>10} {:>11} {:>11} {:>9}", "width", "size", "table ns", "cpu ns", "speedup");
    for width in [20usize, 50, 100, 300, 1000] {
        let tbl = ArithTable::new(width);
        let mb = tbl.bytes() as f64 / (1024.0 * 1024.0);
        let iters = if width >= 1000 { 4_000_000 } else { 8_000_000 };
        let ops: Vec<(usize, usize)> = (0..iters)
            .map(|i| (i % width, (i * 7 + 3) % width))
            .collect();

        let t0 = Instant::now();
        let mut s = 0i64;
        for &(a, b) in ops.iter() {
            s += black_box(tbl.mul(black_box(a), black_box(b))) as i64;
        }
        let tab_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = Instant::now();
        let mut s2 = 0i64;
        for &(a, b) in ops.iter() {
            s2 += black_box((black_box(a) * black_box(b)) as i32) as i64;
        }
        let cpu_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        println!("{:<8} {:>7.2} MB {:>11.3} {:>11.3} {:>8.2}x",
                 width, mb, tab_ns, cpu_ns, cpu_ns / tab_ns);
        std::hint::black_box(s + s2);
    }

    println!("\nreading: with the dispatcher gone, the numbers say what they claim to.");
    println!("A lookup is an index computation and a load; that is cheaper than an");
    println!("expensive operation and not cheaper than a cheap one. The width sweep");
    println!("shows where the table stops fitting and the load starts to cost more.");
}
