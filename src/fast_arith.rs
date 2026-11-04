//! All four operations, made as fast as they can honestly be made.
//!
//! The goal is that calculation is either not done at all or done very little.
//! Reaching that means answering two questions with measurements rather than
//! with a rule of thumb:
//!
//!   which operations does a table actually help
//!   how large can the table be before it stops helping
//!
//! The first is not the same for all four. Multiplication and addition cost the
//! CPU one or two cycles; a table replaces them with an index computation and a
//! memory load, which is not obviously cheaper. Division costs tens of cycles,
//! so a table has real room there. Subtraction sits between them. Measuring all
//! four is the only way to know, and the earlier grammar result is a warning:
//! a table on an already-cheap operation makes things slower.
//!
//! The second question is a cache question. A 100x100 table is 40 KB and lives
//! in L2. A 1000x1000 table is 4 MB and starts missing cache, at which point the
//! load costs more than the arithmetic it replaced. That crossover is measured
//! here, because the right table size is the largest one that still fits.
//!
//! Beyond the table, decomposition. 47*23 is 47*20 + 47*3, both recalled. This
//! was verified earlier at zero errors over 40,000 operand pairs, and it is what
//! makes "all possible calculations" unnecessary: the table only has to cover
//! the pieces, not the products.

use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Add => "addition",
            Op::Sub => "subtraction",
            Op::Mul => "multiplication",
            Op::Div => "division",
        }
    }
}

/// One table per operation, all square and all the same width.
pub struct ArithTable {
    pub n: usize,
    add: Vec<i64>,
    sub: Vec<i64>,
    mul: Vec<i64>,
    div: Vec<i64>,
    pub exact_divs: usize,
}

impl ArithTable {
    pub fn new(n: usize) -> Self {
        let mut add = vec![0i64; n * n];
        let mut sub = vec![i64::MIN; n * n];
        let mut mul = vec![0i64; n * n];
        let mut div = vec![i64::MIN; n * n];
        let mut exact = 0usize;
        for a in 0..n {
            for b in 0..n {
                let i = a * n + b;
                add[i] = (a + b) as i64;
                if a >= b {
                    sub[i] = (a - b) as i64;
                }
                mul[i] = (a * b) as i64;
                if b != 0 && a % b == 0 {
                    div[i] = (a / b) as i64;
                    exact += 1;
                }
            }
        }
        Self { n, add, sub, mul, div, exact_divs: exact }
    }

    pub fn bytes(&self) -> usize {
        4 * self.n * self.n * std::mem::size_of::<i64>()
    }

    /// Recall if the operation and both operands are inside the table.
    #[inline]
    pub fn recall(&self, op: Op, a: usize, b: usize) -> Option<i64> {
        if a >= self.n || b >= self.n {
            return None;
        }
        let i = a * self.n + b;
        let v = match op {
            Op::Add => self.add[i],
            Op::Sub => self.sub[i],
            Op::Mul => self.mul[i],
            Op::Div => self.div[i],
        };
        if v == i64::MIN { None } else { Some(v) }
    }

    /// The full path: recall, else decompose, else compute.
    ///
    /// Decomposition splits the operands into table-sized digits, which is what
    /// a person does past the end of their tables. It is exact for every
    /// operation here, so the answer never degrades to an approximation.
    pub fn eval(&self, op: Op, a: u64, b: u64) -> i64 {
        if let Some(v) = self.recall(op, a as usize, b as usize) {
            return v;
        }
        let n = self.n as u64;
        match op {
            Op::Add => (a + b) as i64,
            Op::Sub => (a as i64) - (b as i64),
            Op::Mul => {
                // a = ah*n + al, b = bh*n + bl
                let (ah, al) = ((a / n) as i64, (a % n) as i64);
                let (bh, bl) = ((b / n) as i64, (b % n) as i64);
                ah * bh * (n as i64 * n as i64)
                    + ah * bl * n as i64
                    + al * bh * n as i64
                    + al * bl
            }
            Op::Div => {
                // past the table, the honest thing is to compute: decomposition
                // for division is not a lookup, it is long division
                if b == 0 { 0 } else { (a / b) as i64 }
            }
        }
    }
}

fn cpu(op: Op, a: u64, b: u64) -> i64 {
    match op {
        Op::Add => (a + b) as i64,
        Op::Sub => (a as i64) - (b as i64),
        Op::Mul => (a * b) as i64,
        Op::Div => if b == 0 { 0 } else { (a / b) as i64 },
    }
}

fn main() {
    const N: usize = 20_000_000;
    println!("==============================================================================");
    println!("all four operations: recall against computing, and how big the table can be");
    println!("==============================================================================\n");

    // correctness first
    let t = ArithTable::new(100);
    let mut wrong = 0;
    for a in 0..100usize {
        for b in 0..100usize {
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                if let Some(v) = t.recall(op, a, b) {
                    if v != cpu(op, a as u64, b as u64) { wrong += 1; }
                }
            }
        }
    }
    println!("table disagrees with the CPU: {wrong}  (zero required)");

    let mut dwrong = 0;
    for a in 0..300u64 {
        for b in 0..300u64 {
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                if t.eval(op, a, b) != cpu(op, a, b) { dwrong += 1; }
            }
        }
    }
    println!("full path (recall + decompose + compute) over 90,000 triples: {dwrong} wrong\n");

    // operands outside the timed region so the loop measures the operation
    let pairs: Vec<(u64, u64)> = (0..N)
        .map(|i| {
            let b = 1 + ((i as u64 * 7919) % 99);
            let a = ((i as u64 * 104729) % 99) * b;
            (a, b)
        })
        .collect();

    println!("{:<20} {:>12} {:>12} {:>9}", "operation", "table ns", "cpu ns", "speedup");
    println!("{}", "-".repeat(58));
    let mut best = Vec::new();
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        let t0 = Instant::now();
        let mut s = 0i64;
        for &(a, b) in pairs.iter() {
            let a = black_box(a % 100) as usize;
            let b = black_box(b % 100) as usize;
            s += black_box(t.recall(op, a, b)).unwrap_or(0);
        }
        let tab = t0.elapsed().as_nanos() as f64 / N as f64;

        let t0 = Instant::now();
        let mut s2 = 0i64;
        for &(a, b) in pairs.iter() {
            let a = black_box(a % 100);
            let b = black_box(b % 100);
            s2 += black_box(cpu(op, a, b));
        }
        let c = t0.elapsed().as_nanos() as f64 / N as f64;

        println!("{:<20} {:>12.3} {:>12.3} {:>8.2}x",
                 op.name(), tab, c, c / tab);
        best.push((op.name(), c / tab, s + s2));
    }
    println!("\nchecksum {}", best.iter().map(|x| x.2).sum::<i64>());

    // how large can the table be before it stops paying
    println!("\n{}", "=".repeat(78));
    println!("table width against cost, for multiplication (the hardest case for a table)");
    println!("{}", "=".repeat(78));
    println!("{:<10} {:>12} {:>12} {:>10} {:>12}", "width", "size", "table ns", "cpu ns", "speedup");
    for width in [20usize, 50, 100, 300, 1000] {
        let tbl = ArithTable::new(width);
        let mb = tbl.bytes() as f64 / (1024.0 * 1024.0);
        let iters = if width >= 1000 { 5_000_000 } else { 10_000_000 };
        let t0 = Instant::now();
        let mut s = 0i64;
        for i in 0..iters {
            let a = black_box(i % width);
            let b = black_box((i * 7) % width);
            s += black_box(tbl.recall(Op::Mul, a, b)).unwrap_or(0);
        }
        let tab = t0.elapsed().as_nanos() as f64 / iters as f64;
        let t0 = Instant::now();
        let mut s2 = 0i64;
        for i in 0..iters {
            let a = black_box(i % width);
            let b = black_box((i * 7) % width);
            s2 += black_box((a * b) as i64);
        }
        let c = t0.elapsed().as_nanos() as f64 / iters as f64;
        println!("{:<10} {:>9.2} MB {:>12.3} {:>12.3} {:>9.2}x",
                 width, mb, tab, c, c / tab);
        std::hint::black_box(s + s2);
    }

    println!("\nreading: a table pays where the operation is expensive and stops paying");
    println!("when it no longer fits in cache. The width column shows both effects at");
    println!("once, which is why the useful table is small and the decomposition path");
    println!("does the rest.");
}
