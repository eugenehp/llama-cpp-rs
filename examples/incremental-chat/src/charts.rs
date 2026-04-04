//! Generate SVG benchmark charts from hardcoded benchmark data.
//!
//! ```console
//! cargo run --release -p incremental-chat --bin incremental-charts
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::write_literal
)]

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Benchmark data (Qwen2.5-0.5B Q4_K_M, Apple Silicon, latest run)
// ---------------------------------------------------------------------------

struct LatencyRow { label: &'static str, normal_ms: f64, flush_ms: f64, ttft_ms: f64 }
const LATENCY: &[LatencyRow] = &[
    LatencyRow { label: "Short (7t)",  normal_ms: 27.87, flush_ms: 13.63, ttft_ms: 14.39 },
    LatencyRow { label: "Med A (14t)", normal_ms: 51.63, flush_ms: 10.52, ttft_ms: 10.61 },
    LatencyRow { label: "Med B (14t)", normal_ms: 52.06, flush_ms: 10.42, ttft_ms: 10.50 },
    LatencyRow { label: "Med C (14t)", normal_ms: 52.15, flush_ms: 10.32, ttft_ms: 10.41 },
];

struct LoadRow { label: &'static str, normal_ms: f64, margin_ms: f64, naive_ms: f64 }
const LOAD: &[LoadRow] = &[
    LoadRow { label: "7 tokens",  normal_ms: 28.32, margin_ms: 44.47, naive_ms: 72.81  },
    LoadRow { label: "14 tok A",  normal_ms: 52.07, margin_ms: 81.70, naive_ms: 181.08 },
    LoadRow { label: "14 tok B",  normal_ms: 54.10, margin_ms: 88.60, naive_ms: 215.83 },
    LoadRow { label: "14 tok C",  normal_ms: 52.03, margin_ms: 81.54, naive_ms: 198.05 },
];

struct UxRow { label: &'static str, time_ms: f64, tokens: u32 }
const UX: &[UxRow] = &[
    UxRow { label: "Delete tail",      time_ms: 5.97,  tokens: 1  },
    UxRow { label: "Retype tail",      time_ms: 5.80,  tokens: 1  },
    UxRow { label: "Mid-line replace", time_ms: 28.69, tokens: 7  },
    UxRow { label: "Prefix insert",    time_ms: 42.38, tokens: 11 },
];

struct KvRow { label: &'static str, prefill_ms: f64, gen64_ms: f64, diverge_char: i32 }
// diverge_char: character position where output diverges from F16 (-1 = identical)
const KV: &[KvRow] = &[
    KvRow { label: "F16",           prefill_ms: 51.65, gen64_ms: 386.33, diverge_char: -1  },
    KvRow { label: "Q8+turbo",      prefill_ms: 52.76, gen64_ms: 398.55, diverge_char: 195 },
    KvRow { label: "Q5+turbo",      prefill_ms: 53.02, gen64_ms: 411.92, diverge_char: 24  },
    KvRow { label: "Q4+turbo",      prefill_ms: 53.24, gen64_ms: 404.23, diverge_char: 24  },
    KvRow { label: "Q5 no-t",       prefill_ms: 58.44, gen64_ms: 387.46, diverge_char: 2   },
    KvRow { label: "Q4 no-t",       prefill_ms: 52.23, gen64_ms: 386.76, diverge_char: 2   },
];

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------
const C_BLUE:   &str = "#4C78A8";
const C_ORANGE: &str = "#F58518";
const C_RED:    &str = "#E45756";
const C_TEAL:   &str = "#72B7B2";
const C_GREEN:  &str = "#54A24B";
const C_YELLOW: &str = "#EECA3B";
const C_BG:     &str = "#1a1a2e";

// ---------------------------------------------------------------------------
// SVG builder
// ---------------------------------------------------------------------------

fn svg_header(w: f64, h: f64, title: &str, subtitle: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" \
         font-family=\"system-ui,-apple-system,sans-serif\">\n\
         <rect width=\"{}\" height=\"{}\" fill=\"{}\" rx=\"12\"/>\n\
         <text x=\"{}\" y=\"32\" fill=\"#e0e0e0\" font-size=\"16\" font-weight=\"bold\" \
         text-anchor=\"middle\">{}</text>\n\
         <text x=\"{}\" y=\"50\" fill=\"#888\" font-size=\"11\" \
         text-anchor=\"middle\">{}</text>\n",
        w, h, w, h, C_BG, w/2.0, title, w/2.0, subtitle
    )
}

fn svg_footer() -> &'static str { "</svg>\n" }

fn bar(svg: &mut String, x: f64, y: f64, w: f64, h: f64, color: &str, label: &str) {
    write!(
        svg,
        "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
         fill=\"{color}\" rx=\"3\" opacity=\"0.9\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#ccc\" font-size=\"9\" \
         text-anchor=\"middle\">{label}</text>\n",
        x + w / 2.0, y - 4.0
    ).unwrap();
}

fn gridline(svg: &mut String, x1: f64, x2: f64, y: f64, label: &str) {
    write!(
        svg,
        "<line x1=\"{x1:.0}\" y1=\"{y:.0}\" x2=\"{x2:.0}\" y2=\"{y:.0}\" \
         stroke=\"#333\" stroke-width=\"0.5\"/>\n\
         <text x=\"{:.0}\" y=\"{:.0}\" fill=\"#666\" font-size=\"9\" \
         text-anchor=\"end\">{label}</text>\n",
        x1 - 5.0, y + 3.0
    ).unwrap();
}

fn legend_item(svg: &mut String, x: f64, y: f64, color: &str, label: &str) {
    write!(
        svg,
        "<rect x=\"{x:.0}\" y=\"{:.0}\" width=\"12\" height=\"12\" fill=\"{color}\" rx=\"2\"/>\n\
         <text x=\"{:.0}\" y=\"{:.0}\" fill=\"#ccc\" font-size=\"11\">{label}</text>\n",
        y - 10.0, x + 16.0, y
    ).unwrap();
}

fn group_label(svg: &mut String, x: f64, y: f64, label: &str) {
    write!(
        svg,
        "<text x=\"{x:.1}\" y=\"{y:.1}\" fill=\"#aaa\" font-size=\"10\" \
         text-anchor=\"middle\">{label}</text>\n"
    ).unwrap();
}

fn hbar(svg: &mut String, y: f64, val: f64, max: f64, cw: f64, lx: f64, color: &str, label: &str, vlabel: &str) {
    let bw = (val / max) * cw;
    let bh = 22.0;
    write!(
        svg,
        "<text x=\"{:.0}\" y=\"{:.1}\" fill=\"#aaa\" font-size=\"11\" \
         text-anchor=\"end\">{label}</text>\n\
         <rect x=\"{lx:.0}\" y=\"{y:.0}\" width=\"{bw:.1}\" height=\"{bh}\" \
         fill=\"{color}\" rx=\"3\" opacity=\"0.85\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#eee\" font-size=\"10\" \
         font-weight=\"bold\">{vlabel}</text>\n",
        lx - 8.0, y + bh/2.0 + 4.0,
        lx + bw + 6.0, y + bh/2.0 + 4.0,
    ).unwrap();
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

fn chart_latency() -> String {
    let (w, h) = (700.0, 320.0);
    let mut s = svg_header(w, h, "1. Latency — Normal vs Incremental Prefill",
                           "Qwen2.5-0.5B Q4_K_M · Apple Silicon");
    let (max_v, base_y, ch) = (60.0, 260.0, 180.0);
    for i in 0..=3 {
        let v = i as f64 * 20.0;
        gridline(&mut s, 80.0, 620.0, base_y - (v/max_v)*ch, &format!("{v:.0}ms"));
    }
    let xpos = [170.0, 310.0, 450.0, 580.0];
    let bw = 35.0;
    let gap = 3.0;
    for (i, row) in LATENCY.iter().enumerate() {
        let cx = xpos[i];
        let sx = cx - (3.0 * bw + 2.0 * gap) / 2.0;
        for (j, (val, col)) in [(row.normal_ms, C_RED), (row.flush_ms, C_BLUE), (row.ttft_ms, C_TEAL)].iter().enumerate() {
            let bx = sx + j as f64 * (bw + gap);
            let bh = (val / max_v) * ch;
            bar(&mut s, bx, base_y - bh, bw, bh, col, &format!("{val:.1}"));
        }
        let spd = row.normal_ms / row.flush_ms;
        group_label(&mut s, cx, base_y + 16.0, &format!("{} ({spd:.1}×)", row.label));
    }
    legend_item(&mut s, 90.0, 280.0, C_RED,  "Normal prefill");
    legend_item(&mut s, 90.0, 298.0, C_BLUE, "Incremental flush");
    legend_item(&mut s, 250.0, 280.0, C_TEAL, "TTFT (incremental)");
    s.push_str(svg_footer());
    s
}

fn chart_load() -> String {
    let (w, h) = (700.0, 320.0);
    let mut s = svg_header(w, h, "3. GPU Load — Normal vs Margin vs Naive",
                           "Total compute time (lower = less GPU work)");
    let (max_v, base_y, ch) = (230.0, 260.0, 180.0);
    for i in 0..=4 {
        let v = i as f64 * 50.0;
        gridline(&mut s, 80.0, 620.0, base_y - (v/max_v)*ch, &format!("{v:.0}ms"));
    }
    let xpos = [170.0, 310.0, 450.0, 580.0];
    let bw = 30.0;
    let gap = 3.0;
    for (i, row) in LOAD.iter().enumerate() {
        let cx = xpos[i];
        let sx = cx - (3.0 * bw + 2.0 * gap) / 2.0;
        for (j, (val, col)) in [(row.normal_ms, C_BLUE), (row.margin_ms, C_GREEN), (row.naive_ms, C_RED)].iter().enumerate() {
            let bx = sx + j as f64 * (bw + gap);
            let bh = (val / max_v) * ch;
            bar(&mut s, bx, base_y - bh, bw, bh, col, &format!("{val:.0}"));
        }
        let sav = (1.0 - row.margin_ms / row.naive_ms) * 100.0;
        group_label(&mut s, cx, base_y + 16.0, &format!("{} (saves {sav:.0}%)", row.label));
    }
    legend_item(&mut s, 90.0, 280.0, C_BLUE,  "Normal");
    legend_item(&mut s, 90.0, 298.0, C_GREEN, "Margin (BPE-stable)");
    legend_item(&mut s, 260.0, 280.0, C_RED,   "Naive (no margin)");
    s.push_str(svg_footer());
    s
}

fn chart_ux() -> String {
    let (w, h) = (600.0, 260.0);
    let mut s = svg_header(w, h, "5. UX — Edit Recovery Cost",
                           "Time to re-sync KV cache after mid-line edits");
    let colors = [C_BLUE, C_TEAL, C_ORANGE, C_RED];
    for (i, row) in UX.iter().enumerate() {
        let y = 80.0 + i as f64 * 40.0;
        hbar(&mut s, y, row.time_ms, 50.0, 380.0, 160.0, colors[i],
             row.label, &format!("{:.1}ms ({} tok)", row.time_ms, row.tokens));
    }
    s.push_str(svg_footer());
    s
}

fn chart_kv() -> String {
    let (w, h) = (720.0, 400.0);
    let mut s = svg_header(w, h, "7. KV Cache Quantization + TurboQuant",
                           "Gen 64 tokens — divergence point from F16 baseline (higher = better quality)");

    // Top half: generation time bars
    let (max_v, base_y, ch) = (450.0, 200.0, 120.0);
    for i in 0..=4 {
        let v = i as f64 * 100.0;
        gridline(&mut s, 60.0, 680.0, base_y - (v/max_v)*ch, &format!("{v:.0}ms"));
    }
    let n = KV.len();
    let gw = 600.0 / n as f64;
    let bw = 30.0;
    let gap = 3.0;
    for (i, row) in KV.iter().enumerate() {
        let cx = 80.0 + i as f64 * gw + gw / 2.0;
        let sx = cx - (2.0 * bw + gap) / 2.0;
        // Prefill bar
        let bh = (row.prefill_ms / max_v) * ch;
        bar(&mut s, sx, base_y - bh, bw, bh, C_BLUE, &format!("{:.0}", row.prefill_ms));
        // Gen64 bar
        let bh2 = (row.gen64_ms / max_v) * ch;
        bar(&mut s, sx + bw + gap, base_y - bh2, bw, bh2, C_ORANGE, &format!("{:.0}", row.gen64_ms));
        group_label(&mut s, cx, base_y + 16.0, row.label);
    }
    legend_item(&mut s, 80.0, 220.0, C_BLUE,   "Prefill");
    legend_item(&mut s, 80.0, 238.0, C_ORANGE, "Gen 64 tokens");

    // Bottom half: quality divergence bars
    let qual_y = 270.0;
    write!(
        s,
        "<text x=\"360\" y=\"{qual_y}\" fill=\"#e0e0e0\" font-size=\"13\" \
         font-weight=\"bold\" text-anchor=\"middle\">Output quality: chars before divergence from F16</text>\n"
    ).unwrap();

    let max_div = 220.0;
    let bar_left = 140.0;
    let bar_area = 520.0;
    let qual_colors = [C_TEAL, C_GREEN, C_GREEN, C_YELLOW, C_RED, C_RED];
    for (i, row) in KV.iter().enumerate() {
        if row.diverge_char < 0 { continue; } // skip baseline
        let y = qual_y + 12.0 + i as f64 * 26.0;
        let val = row.diverge_char as f64;
        let quality = if val >= 195.0 { "near-identical" }
                      else if val >= 20.0 { "coherent" }
                      else { "DEGRADED" };
        hbar(&mut s, y, val.min(max_div), max_div, bar_area, bar_left, qual_colors[i],
             row.label, &format!("char {} ({quality})", row.diverge_char));
    }

    s.push_str(svg_footer());
    s
}

fn chart_speedup() -> String {
    let (w, h) = (500.0, 280.0);
    let mut s = svg_header(w, h, "Perceived Speedup at Enter",
                           "Incremental flush vs normal prefill (higher = better)");
    let colors = [C_BLUE, C_GREEN, C_TEAL, C_ORANGE];
    let max_v = 6.0;
    let lx = 140.0;
    let cw = 300.0;
    let mut sum = 0.0;
    for (i, row) in LATENCY.iter().enumerate() {
        let spd = row.normal_ms / row.flush_ms;
        sum += spd;
        let y = 80.0 + i as f64 * 42.0;
        hbar(&mut s, y, spd, max_v, cw, lx, colors[i],
             row.label, &format!("{spd:.1}× faster"));
    }
    let avg = sum / LATENCY.len() as f64;
    let avg_x = lx + (avg / max_v) * cw;
    write!(
        s,
        "<line x1=\"{avg_x:.0}\" y1=\"70\" x2=\"{avg_x:.0}\" y2=\"250\" \
         stroke=\"{C_YELLOW}\" stroke-width=\"2\" stroke-dasharray=\"6 3\"/>\n\
         <text x=\"{avg_x:.0}\" y=\"265\" fill=\"{C_YELLOW}\" font-size=\"11\" \
         text-anchor=\"middle\">avg {avg:.1}\u{00d7}</text>\n"
    ).unwrap();
    s.push_str(svg_footer());
    s
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn chart_samplers() -> String {
    let (w, h) = (750.0, 420.0);
    let mut s = svg_header(w, h, "8. Samplers & Temperature",
                           "48 tokens generated from \"Write a haiku about the ocean\" (seed=42)");

    struct Row { label: &'static str, gen_ms: f64, tps: f64 }
    let rows: &[Row] = &[
        Row { label: "greedy (t=0)",       gen_ms: 306.1, tps: 156.8 },
        Row { label: "t=0.1 top_k=40",    gen_ms: 306.4, tps: 156.7 },
        Row { label: "t=0.4 top_p=0.9",   gen_ms: 330.3, tps: 145.3 },
        Row { label: "t=0.7 top_p=0.9",   gen_ms: 335.2, tps: 143.2 },
        Row { label: "t=1.0 top_p=0.95",  gen_ms: 413.3, tps: 116.1 },
        Row { label: "t=1.5 top_k=50",    gen_ms: 307.1, tps: 156.3 },
        Row { label: "min_p=0.05 t=0.7",  gen_ms: 309.7, tps: 155.0 },
        Row { label: "top_n_sigma=1.0",   gen_ms: 643.0, tps: 74.7  },
        Row { label: "mirostat_v2",       gen_ms: 564.5, tps: 85.0  },
    ];

    let colors = [C_BLUE, C_BLUE, C_TEAL, C_GREEN, C_YELLOW, C_ORANGE, C_TEAL, C_RED, C_RED];
    let max_ms = 700.0;
    let lx = 170.0;
    let cw = 500.0;

    for (i, row) in rows.iter().enumerate() {
        let y = 70.0 + i as f64 * 36.0;
        hbar(&mut s, y, row.gen_ms, max_ms, cw, lx, colors[i],
             row.label, &format!("{:.0}ms ({:.0} tok/s)", row.gen_ms, row.tps));
    }

    // footer note
    write!(
        s,
        "<text x=\"375\" y=\"405\" fill=\"#666\" font-size=\"10\" \
         text-anchor=\"middle\">Same seed (42) produces deterministic output across runs</text>\n"
    ).unwrap();

    s.push_str(svg_footer());
    s
}

fn main() {
    let dir = Path::new("charts");
    fs::create_dir_all(dir).expect("create charts dir");

    let charts: Vec<(&str, String)> = vec![
        ("latency.svg",  chart_latency()),
        ("load.svg",     chart_load()),
        ("ux.svg",       chart_ux()),
        ("kv_quant.svg", chart_kv()),
        ("speedup.svg",  chart_speedup()),
        ("samplers.svg", chart_samplers()),
    ];

    for (name, svg) in &charts {
        let path = dir.join(name);
        fs::write(&path, svg).expect("write svg");
        println!("  \u{2713} {}", path.display());
    }
    println!("\n  Done — {} charts written to {}/", charts.len(), dir.display());
}
