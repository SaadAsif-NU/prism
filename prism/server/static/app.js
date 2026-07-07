"use strict";

const editor = document.getElementById("editor");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const planEl = document.getElementById("plan");

const NUMERIC = new Set(["INTEGER", "FLOAT"]);

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function showTab(which) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === which)
  );
  document
    .getElementById("panel-result")
    .classList.toggle("active", which === "result");
  document.getElementById("panel-plan").classList.toggle("active", which === "plan");
}

function renderCell(value, type) {
  const td = document.createElement("td");
  if (value === null) {
    td.textContent = "NULL";
    td.className = "null";
    return td;
  }
  if (type === "BOOLEAN") {
    td.textContent = value ? "true" : "false";
    return td;
  }
  td.textContent = String(value);
  if (NUMERIC.has(type)) td.className = "num";
  return td;
}

function renderResult(payload) {
  resultEl.innerHTML = "";
  if (payload.columns.length === 0) {
    resultEl.innerHTML = '<p class="hint">Query returned no columns.</p>';
    return;
  }
  const table = document.createElement("table");
  table.className = "grid";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  payload.columns.forEach((name, i) => {
    const th = document.createElement("th");
    th.innerHTML = `${escapeHtml(name)}<br /><span class="type">${payload.types[i]}</span>`;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  payload.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value, i) => tr.appendChild(renderCell(value, payload.types[i])));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  resultEl.appendChild(table);
}

function renderError(message) {
  resultEl.innerHTML = "";
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = message;
  resultEl.appendChild(div);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function runQuery() {
  const sql = editor.value.trim();
  if (!sql) return;
  setStatus("running...");
  showTab("result");
  const data = await api("/api/query", { sql });
  if (!data.ok) {
    renderError(data.error);
    setStatus("error", "err");
    return;
  }
  renderResult(data);
  const noun = data.row_count === 1 ? "row" : "rows";
  setStatus(`${data.row_count} ${noun} in ${data.elapsed_ms.toFixed(1)} ms`, "ok");
}

async function explainQuery() {
  const sql = editor.value.trim();
  if (!sql) return;
  setStatus("planning...");
  showTab("plan");
  const data = await api("/api/explain", { sql });
  if (!data.ok) {
    planEl.innerHTML = "";
    const div = document.createElement("div");
    div.className = "error";
    div.textContent = data.error;
    planEl.appendChild(div);
    setStatus("error", "err");
    return;
  }
  planEl.innerHTML = `
    <div class="plan-block optimized">
      <h3>Optimized plan</h3>
      <pre>${escapeHtml(data.optimized)}</pre>
    </div>
    <div class="plan-block">
      <h3>Original plan (before optimization)</h3>
      <pre>${escapeHtml(data.original)}</pre>
    </div>`;
  setStatus("plan ready", "ok");
}

async function loadTables() {
  const data = await (await fetch("/api/tables")).json();
  const container = document.getElementById("tables");
  container.innerHTML = "";
  data.tables.forEach((table) => {
    const card = document.createElement("div");
    card.className = "table-card";

    const head = document.createElement("div");
    head.className = "table-head";
    head.innerHTML = `<span>${escapeHtml(table.name)}</span><span class="rows">${table.rows} rows</span>`;
    head.addEventListener("click", (event) => {
      // A plain click expands the schema; the chevron double-click runs a preview.
      card.classList.toggle("open");
      if (event.detail === 2) {
        editor.value = `SELECT * FROM ${table.name} LIMIT 20;`;
        runQuery();
      }
    });

    const cols = document.createElement("div");
    cols.className = "table-cols";
    table.columns.forEach((c) => {
      const row = document.createElement("div");
      row.className = "col-row";
      row.innerHTML = `<span>${escapeHtml(c.name)}</span><span class="ctype">${c.type}</span>`;
      cols.appendChild(row);
    });

    card.appendChild(head);
    card.appendChild(cols);
    container.appendChild(card);
  });
}

async function loadSamples() {
  const data = await (await fetch("/api/samples")).json();
  const container = document.getElementById("samples");
  container.innerHTML = "";
  data.samples.forEach((sample) => {
    const btn = document.createElement("button");
    btn.className = "sample";
    btn.textContent = sample.label;
    btn.addEventListener("click", () => {
      editor.value = sample.sql;
      runQuery();
    });
    container.appendChild(btn);
  });
}

document.getElementById("run").addEventListener("click", runQuery);
document.getElementById("explain").addEventListener("click", explainQuery);
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => showTab(tab.dataset.tab))
);
editor.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    runQuery();
  }
});

loadTables();
loadSamples();
editor.value =
  "SELECT name, department, salary\nFROM employees\nORDER BY salary DESC NULLS LAST\nLIMIT 5;";
