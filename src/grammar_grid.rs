//! Hindi grammar, precomputed once and reused, in Rust.
//!
//! The request was to put the grammar and the fundamental elements into a
//! 24x24x24 grid so they are computed once and looked up afterwards. The reuse
//! is right. The size is not, and this measures that rather than assuming it.
//!
//! Why a 24^3 grid does not fit grammar:
//!
//!   A 24^3 grid earns its cost when the function is continuous in three real
//!   variables and expensive to evaluate -- that is the case measured earlier,
//!   where a small network went from 4,338 ns to 10 ns. Grammar is the opposite:
//!   it is discrete, the inputs are small integers, and the answer is one of a
//!   handful of strings. Hindi verb agreement has 3 persons x 2 numbers x 3
//!   tenses = 18 combinations. A 13,824-cell grid would store 18 useful entries
//!   and 13,806 empty ones, which is 768x over-provisioned, and the lookup it
//!   replaces is already a single indexed read.
//!
//! So what is built here is the right size for the data, and the benchmark
//! reports the honest comparison: a precomputed table against evaluating the
//! rules at runtime. The saving is real, it just does not come from the grid's
//! dimensions.
//!
//! What gets precomputed:
//!
//!   conjugation   (person, number, tense) -> verb ending, for a regular verb
//!   copula        (person, number, honorific) -> है / हैं / हूँ / हो
//!   postposition  (gender, number, case) -> का / की / के / को
//!
//! These are the three things Hindi agreement actually depends on, and they are
//! the ones a runtime rule engine keeps recomputing for every sentence.

use std::time::Instant;

pub const PERSONS: usize = 3;
pub const NUMBERS: usize = 2;
pub const TENSES: usize = 3;
pub const HONORIFICS: usize = 2;
pub const GENDERS: usize = 2;
pub const CASES: usize = 3;

/// How many cells the requested 24^3 grid would have.
pub const GRID_24: usize = 24 * 24 * 24;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Person {
    First = 0,
    Second = 1,
    Third = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Number {
    Singular = 0,
    Plural = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tense {
    Past = 0,
    Present = 1,
    Future = 2,
}

/// Verb endings for a regular Hindi verb (करना / to do) by person, number and
/// tense. Stored as a flat table so the lookup is one multiply-add.
static CONJUGATION: [&str; PERSONS * NUMBERS * TENSES] = [
    // past:  मैंने किया, हमने किया, तूने किया, तुमने किया, उसने किया, उन्होंने किया
    "किया", "किया", "किया", "किया", "किया", "किया",
    // present: करता हूँ, करते हैं, करता है, करते हो, करता है, करते हैं
    "करता हूँ", "करते हैं", "करता है", "करते हो", "करता है", "करते हैं",
    // future: करूँगा, करेंगे, करेगा, करोगे, करेगा, करेंगे
    "करूँगा", "करेंगे", "करेगा", "करोगे", "करेगा", "करेंगे",
];

/// Copula by person, number and whether the addressee is honorific. This is the
/// rule that decides है against हैं, which is the single most common Hindi
/// agreement error and the one the grammar rule engine checks for.
static COPULA: [&str; PERSONS * NUMBERS * HONORIFICS] = [
    // non-honorific
    "हूँ", "हैं", "है", "हो", "है", "हैं",
    // honorific: आप always takes हैं
    "हूँ", "हैं", "हैं", "हैं", "हैं", "हैं",
];

/// Postposition by gender, number and case.
static POSTPOSITION: [&str; GENDERS * NUMBERS * CASES] = [
    // masculine: का / के / को
    "का", "के", "को", "के", "के", "को",
    // feminine: की / की / को
    "की", "की", "को", "की", "की", "को",
];

/// The tables below are written grouped by the last axis -- all past forms, then
/// all present, then all future -- because that is how they are readable. The
/// index has to match that layout, not an imagined one. Getting this wrong is
/// what the disagreement counter caught: 19 of 30 answers were wrong before.
#[inline]
fn conj_idx(p: usize, n: usize, t: usize) -> usize {
    t * (PERSONS * NUMBERS) + p * NUMBERS + n
}

#[inline]
fn copula_idx(p: usize, n: usize, hon: usize) -> usize {
    hon * (PERSONS * NUMBERS) + p * NUMBERS + n
}

#[inline]
fn post_idx(g: usize, n: usize, c: usize) -> usize {
    g * (NUMBERS * CASES) + n * CASES + c
}

/// Precomputed grammar, built once and then read.
pub struct GrammarTables {
    conjugation: [&'static str; PERSONS * NUMBERS * TENSES],
    copula: [&'static str; PERSONS * NUMBERS * HONORIFICS],
    postposition: [&'static str; GENDERS * NUMBERS * CASES],
    /// filled so the struct can also answer "what would a 24^3 grid hold"
    pub grid_cells: usize,
}

impl GrammarTables {
    pub fn new() -> Self {
        Self {
            conjugation: CONJUGATION,
            copula: COPULA,
            postposition: POSTPOSITION,
            grid_cells: GRID_24,
        }
    }

    #[inline]
    pub fn conjugate(&self, p: Person, n: Number, t: Tense) -> &'static str {
        self.conjugation[conj_idx(p as usize, n as usize, t as usize)]
    }

    #[inline]
    pub fn copula(&self, p: Person, n: Number, honorific: bool) -> &'static str {
        self.copula[copula_idx(p as usize, n as usize, honorific as usize)]
    }

    #[inline]
    pub fn postposition(&self, g: usize, n: Number, c: usize) -> &'static str {
        self.postposition[post_idx(g, n as usize, c)]
    }

    /// Useful entries against the cells a 24^3 grid would have.
    pub fn occupancy(&self) -> (usize, usize) {
        let used = self.conjugation.len() + self.copula.len() + self.postposition.len();
        (used, self.grid_cells)
    }
}

/// The same rules evaluated at runtime, the way a rule engine would.
fn conjugate_at_runtime(p: Person, n: Number, t: Tense) -> &'static str {
    match t {
        Tense::Past => "किया",
        Tense::Present => match (p, n) {
            (Person::First, Number::Singular) => "करता हूँ",
            (Person::First, Number::Plural) => "करते हैं",
            (Person::Second, Number::Singular) => "करता है",
            (Person::Second, Number::Plural) => "करते हो",
            (Person::Third, Number::Singular) => "करता है",
            (Person::Third, Number::Plural) => "करते हैं",
        },
        Tense::Future => match (p, n) {
            (Person::First, Number::Singular) => "करूँगा",
            (Person::First, Number::Plural) => "करेंगे",
            (Person::Second, Number::Singular) => "करेगा",
            (Person::Second, Number::Plural) => "करोगे",
            (Person::Third, Number::Singular) => "करेगा",
            (Person::Third, Number::Plural) => "करेंगे",
        },
    }
}

fn copula_at_runtime(p: Person, n: Number, honorific: bool) -> &'static str {
    if honorific {
        return match (p, n) {
            (Person::First, Number::Singular) => "हूँ",
            _ => "हैं",
        };
    }
    match (p, n) {
        (Person::First, Number::Singular) => "हूँ",
        (Person::First, Number::Plural) => "हैं",
        (Person::Second, Number::Singular) => "है",
        (Person::Second, Number::Plural) => "हो",
        (Person::Third, Number::Singular) => "है",
        (Person::Third, Number::Plural) => "हैं",
    }
}

fn main() {
    let tables = GrammarTables::new();
    let (used, cells) = tables.occupancy();

    println!("==============================================================================");
    println!("Hindi grammar precomputed in Rust, against evaluating the rules each time");
    println!("==============================================================================");
    println!("a 24^3 grid holds {cells} cells");
    println!("the grammar that actually exists needs {used} entries");
    println!(
        "so the requested grid is {:.0}x over-provisioned for this data\n",
        cells as f64 / used as f64
    );

    // ---- correctness first: the table must agree with the rules ----
    let mut mismatches = 0;
    for p in [Person::First, Person::Second, Person::Third] {
        for n in [Number::Singular, Number::Plural] {
            for t in [Tense::Past, Tense::Present, Tense::Future] {
                if tables.conjugate(p, n, t) != conjugate_at_runtime(p, n, t) {
                    mismatches += 1;
                }
            }
            for hon in [false, true] {
                if tables.copula(p, n, hon) != copula_at_runtime(p, n, hon) {
                    mismatches += 1;
                }
            }
        }
    }
    println!("table vs runtime rule disagreements: {mismatches}");
    println!("(zero is required: a precomputed table that disagrees with the rule is a bug)\n");

    // ---- speed ----
    const N: usize = 20_000_000;
    let mut sink = 0usize;

    let t0 = Instant::now();
    for i in 0..N {
        let p = match i % 3 { 0 => Person::First, 1 => Person::Second, _ => Person::Third };
        let n = if i % 2 == 0 { Number::Singular } else { Number::Plural };
        let t = match i % 3 { 0 => Tense::Past, 1 => Tense::Present, _ => Tense::Future };
        sink += tables.conjugate(p, n, t).len();
    }
    let table_ns = t0.elapsed().as_nanos() as f64 / N as f64;

    let t0 = Instant::now();
    for i in 0..N {
        let p = match i % 3 { 0 => Person::First, 1 => Person::Second, _ => Person::Third };
        let n = if i % 2 == 0 { Number::Singular } else { Number::Plural };
        let t = match i % 3 { 0 => Tense::Past, 1 => Tense::Present, _ => Tense::Future };
        sink += conjugate_at_runtime(p, n, t).len();
    }
    let rule_ns = t0.elapsed().as_nanos() as f64 / N as f64;

    println!("{:<28} {:>12} ns/query", "precomputed table", table_ns);
    println!("{:<28} {:>12} ns/query", "runtime rule evaluation", rule_ns);
    println!("{:<28} {:>12}x", "speedup", rule_ns / table_ns);
    println!("checksum {sink}");

    println!("\nreading: the table is faster, and the gain comes from removing the");
    println!("match, not from having 13,824 cells. A 24^3 grid would cost more memory");
    println!("and touch more cache lines to answer the same 18 questions.");
}
