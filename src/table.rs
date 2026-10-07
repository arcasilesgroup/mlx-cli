//! Table rendering for `list` and `status`.

use crate::discover::Model;
use crate::server::ServerState;

/// Box-drawing table.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    /// Render. No rows -> prints headers only.
    pub fn render(&self) -> String {
        let cols = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.chars().count()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate().take(cols) {
                widths[i] = widths[i].max(cell.chars().count());
            }
        }
        let sep = format!(
            "+{}+",
            widths
                .iter()
                .map(|w| "-".repeat(w + 2))
                .collect::<Vec<_>>()
                .join("+")
        );
        let line = |cells: &[String]| -> String {
            let mut out = String::from("|");
            for (i, cell) in cells.iter().enumerate().take(cols) {
                let pad = widths[i] - cell.chars().count();
                out.push_str(&format!(" {}{} |", cell, " ".repeat(pad)));
            }
            out
        };
        let mut out = vec![sep.clone(), line(&self.headers), sep.clone()];
        for row in &self.rows {
            out.push(line(row));
        }
        out.push(sep);
        out.join("\n")
    }
}

impl std::fmt::Display for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

/// Human-readable byte size: B / KiB / MiB / GiB.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// `mlx list`: one row per discovered model; state from pidfiles, URL from
/// pinned port (or port_base). `ports` = short-name -> pinned port.
pub fn list_table(
    models: &[Model],
    running: &[ServerState],
    pinned: &dyn Fn(&Model) -> u16,
) -> String {
    let mut t = Table::new(&[
        "MODEL", "PARAMS", "SIZE", "STATE", "PORT", "URL", "LOCATION",
    ]);
    for m in models {
        let state = running
            .iter()
            .find(|s| s.model == m.name || m.name.ends_with(&s.model) || s.model.ends_with(&m.name))
            .map_or("stopped".into(), |s| format!("running ({})", s.port));
        let port = pinned(m);
        t.row(vec![
            m.name.clone(),
            m.params.clone().unwrap_or_else(|| "-".into()),
            human_bytes(m.size_bytes),
            state,
            port.to_string(),
            format!("http://localhost:{port}/v1"),
            m.location.clone(),
        ]);
    }
    t.render()
}
