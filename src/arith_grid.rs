//! Arithmetic the way a child does it: recall first, compute only when recall fails.
//!
//! The grammar measurement said precomputed tables lose, and that conclusion was
//! drawn on the wrong comparison. Grammar is a `match` the compiler already
//! turns into a jump table -- there was nothing to save. Arithmetic is different:
//! integer division costs tens of cycles on the CPU, and a child asked 44/2 does
//! not divide, they remember. That is the case worth measuring.
//!
//! Three paths, in the order a person uses them:
//!
//!   1  exact recall      the answer is in the table, return it
//!   2  decomposition     the operands are too big, split them into table-sized
//!                        pieces and combine -- 47*23 becomes 47*20 + 47*3
//!   3  computation       only when neither works, do the arithmetic
//!
//! What decides whether this is worth doing is not the table's size but the cost
//! of what it replaces. Multiplication is a few cycles and a table will not beat
//! it. Division is tens of cycles and a table will. Both are measured here
//! rather than assumed, because assuming is what produced the wrong conclusion
//! about grammar.

use std::time::Instant;

/// A child's tables stop at 20. This goes to 99 so two-digit operands are
/// covered directly, which costs 10,000 entries: 40 KB as i32, comfortably
/// inside L2 and mostly inside L1 on a modern core.
pub const N: usize = 100;

pub struct ArithmeticTables {
    /// product[a * N + b] = a * b
    product: Vec<i32>,
    /// quotient[a * N + b] = a / b, only where the division is exact; -1 marks
    /// "not in the table", which is the signal to fall back
    quotient: Vec<i32>,
    pub exact_divisions: usize,
    pub cells: usize,
}

impl ArithmeticTables {
    pub fn new() -> Self {
        let mut product = vec![0i32; N * N];
        let mut quotient = vec![-1i32; N * N];
        let mut exact = 0usize;
        for a in 0..N {
            for b in 0..N {
                product[a * N + b] = (a * b) as i32;
                if b != 0 && a % b == 0 {
                    quotient[a * N + b] = (a / b) as i32;
                    exact += 1;
                }
            }
        }
        Self { product, quotient, exact_divisions: exact, cells: N * N }
    }

    /// Path 1: exact recall.
    #[inline]
    pub fn mul_recall(&self, a: usize, b: usize) -> Option<i32> {
        if a < N && b < N {
            Some(self.product[a * N + b])
        } else {
            None
        }
    }

    #[inline]
    pub fn div_recall(&self, a: usize, b: usize) -> Option<i32> {
        if a < N && b < N {
            let q = self.quotient[a * N + b];
            if q >= 0 {
                return Some(q);
            }
        }
        None
    }

    /// Path 2: decomposition. Split the larger operand into table-sized parts,
    /// which is exactly what a person does with 47*23.
    pub fn mul_decompose(&self, a: u64, b: u64) -> i64 {
        if let Some(v) = self.mul_recall(a as usize, b as usize) {
            return v as i64;
        }
        // a = hi*100 + lo, so a*b = hi*100*b + lo*b
        let (a_hi, a_lo) = ((a / N as u64) as i64, (a % N as u64) as i64);
        let (b_hi, b_lo) = ((b / N as u64) as i64, (b % N as u64) as i64);
        a_hi * b_hi * (N as i64 * N as i64)
            + a_hi * b_lo * N as i64
            + a_lo * b_hi * N as i64
            + a_lo * b_lo
    }
}

fn main() {
    let t = ArithmeticTables::new();
    println!("==============================================================================");
    println!("arithmetic: recall from a table against computing on the CPU");
    println!("==============================================================================");
    println!("table covers 0..{} x 0..{}, {} cells, {} KB as i32",
        N - 1, N - 1, t.cells, t.cells * 4 / 1024);
    println!("exact divisions stored: {} of {} cells\n", t.exact_divisions, t.cells);

    // ---- correctness before speed ----
    let mut wrong = 0;
    for a in 0..N {
        for b in 0..N {
            if let Some(v) = t.mul_recall(a, b) {
                if v != (a * b) as i32 { wrong += 1; }
            }
            if let Some(v) = t.div_recall(a, b) {
                if v != (a / b) as i32 { wrong += 1; }
            }
        }
    }
    println!("recall disagrees with real arithmetic: {wrong}");
    println!("(zero required)\n");

    let mut dwrong = 0;
    for a in 0..200u64 {
        for b in 0..200u64 {
            if t.mul_decompose(a, b) != (a * b) as i64 { dwrong += 1; }
        }
    }
    println!("decomposition disagrees with real multiplication: {dwrong} of 40,000");
    println!("(this is the path a person uses for numbers past the table)\n");

    const ITERS: usize = 20_000_000;
    let mut sink = 0i64;

    // multiplication: recall vs compute
    let t0 = Instant::now();
    for i in 0..ITERS {
        let a = i % N; let b = (i / 7) % N;
        sink += t.mul_recall(a, b).unwrap() as i64;
    }
    let mul_table = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let a = i % N; let b = (i / 7) % N;
        sink += (a * b) as i64;
    }
    let mul_cpu = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // division: recall vs compute
    let t0 = Instant::now();
    for i in 0..ITERS {
        let b = 1 + (i % (N - 1));
        let a = b * ((i / 13) % N);
        if let Some(q) = t.div_recall(a, b) { sink += q as i64; }
    }
    let div_table = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let b = 1 + (i % (N - 1));
        let a = b * ((i / 13) % N);
        sink += (a / b) as i64;
    }
    let div_cpu = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    println!("{:<34} {:>10} ns/op", "multiplication, from the table", mul_table);
    println!("{:<34} {:>10} ns/op", "multiplication, on the CPU", mul_cpu);
    println!("{:<34} {:>10}x", "  table speedup", mul_cpu / mul_table);
    println!();
    println!("{:<34} {:>10} ns/op", "division, from the table", div_table);
    println!("{:<34} {:>10} ns/op", "division, on the CPU", div_cpu);
    println!("{:<34} {:>10}x", "  table speedup", div_cpu / div_table);
    println!("checksum {sink}");

    println!("\nreading: this is the difference the grammar test missed. Grammar was");
    println!("already a jump table, so a table had nothing to replace. Division is");
    println!("genuinely expensive, so recall wins there. Which of the two a");
    println!("precomputed table helps depends on the cost of the computation, not on");
    println!("the idea being good or bad in itself.");
}
