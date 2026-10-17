"use strict";

/* ---------- tiny helpers ---------- */
const $ = (sel) => document.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { "Content-Type": "application/json" };
    init.method = opts.method || "POST";
  }
  const res = await fetch("/api" + path, init);
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) throw new Error((body && body.detail) || res.statusText);
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 3500);
}

const pct = (v, d = 0) => (v === null || v === undefined ? "–" : (v * 100).toFixed(d) + "%");
const fmtNum = (v, d = 0) => (v === null || v === undefined ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const kb = (bytes) => (bytes === null || bytes === undefined ? "–" : (bytes / 1024).toFixed(bytes < 10240 ? 1 : 0) + " KB");

function tagged(n, render) {
  if (!n || n.value === null || n.value === undefined) return h("span", { class: "muted" }, "–");
  return h("span", { title: n.source || "" }, render(n.value), h("span", { class: `tag ${n.tag}` }, n.tag));
}

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

/* ---------- state and routing ---------- */
const state = { projects: [], data: null, pid: null, doctor: null, settings: null, recipes: [], vendors: [] };

const STEPS = [
  { key: "describe", title: "Describe", hint: "what to detect" },
  { key: "vendor", title: "Vendor", hint: "and requirements" },
  { key: "data", title: "Data", hint: "upload and check" },
  { key: "algorithm", title: "Algorithm", hint: "simplest that works" },
  { key: "rank", title: "Rank chips", hint: "measured, not guessed" },
  { key: "build", title: "Build", hint: "verified package" },
];

function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "p" && parts[1]) return { page: "project", pid: parts[1], step: (parts[2] || "describe").split("?")[0] };
  if (["decisions", "toolchains", "settings"].includes(parts[0])) return { page: parts[0] };
  return { page: "project", pid: null, step: "describe" };
}

function go(pid, step) { location.hash = `#/p/${pid}/${step}`; }

function stepState(key) {
  const d = state.data;
  if (!d) return key === "describe" ? "current" : "locked";
  const status = d.project.status;
  const has = {
    describe: !!d.project.spec,
    vendor: !!d.project.spec,
    data: !!d.dataset,
    algorithm: d.candidates.some((c) => c.selected),
    rank: d.chip_results.some((r) => r.rank),
    build: d.builds.some((b) => b.passed),
  };
  const unlocked = {
    describe: true, vendor: has.describe, data: has.describe, algorithm: has.data,
    rank: has.algorithm, build: has.algorithm,
  };
  if (!unlocked[key]) return "locked";
  return has[key] || (key === "vendor" && has.data) ? "done" : status === "failed" ? "" : "";
}

function renderSteps(route) {
  const nav = $("#steps");
  nav.replaceChildren(...STEPS.map((st, i) => {
    const cls = [stepState(st.key), route.page === "project" && route.step === st.key ? "current" : ""].join(" ");
    const href = state.pid ? `#/p/${state.pid}/${st.key}` : "#/";
    return h("a", { href, class: cls }, h("span", { class: "n" }, i + 1), h("span", {}, st.title), h("small", {}, st.hint));
  }));
}

async function loadProjects() {
  state.projects = await api("/projects");
  const sel = $("#project-select");
  sel.replaceChildren(
    h("option", { value: "" }, "+ New project"),
    ...state.projects.map((p) => h("option", { value: p.id, selected: p.id === state.pid }, `${p.name}  (${p.status})`)),
  );
}

async function loadProject(pid) {
  state.pid = pid;
  state.data = pid ? await api(`/projects/${pid}`) : null;
}

async function render() {
  const route = parseRoute();
  document.querySelectorAll(".topnav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#/" + route.page));
  if (route.page === "project" && route.pid !== state.pid) await loadProject(route.pid);
  if (route.page === "project" && !route.pid) { state.pid = null; state.data = null; }
  await loadProjects();
  renderSteps(route);
  const main = $("#main");
  let view;
  try {
    if (route.page === "decisions") view = await decisionsView();
    else if (route.page === "toolchains") view = await toolchainsView();
    else if (route.page === "settings") view = await settingsView();
    else view = await ({ describe: describeView, vendor: vendorView, data: dataView, algorithm: algorithmView,
      rank: rankView, build: buildView }[route.step] || describeView)();
  } catch (err) {
    view = h("div", { class: "panel error" }, err.message);
  }
  main.replaceChildren(view);
  resumeRunningJob();
}

async function refresh() {
  if (state.pid) state.data = await api(`/projects/${state.pid}`);
  await render();
}

/* ---------- jobs ---------- */
function jobBox() { return h("div", { id: "job-box" }); }

async function runJob(path, body, label) {
  const job = await api(path, { json: body || {} });
  watchJob(job.id, label);
}

function resumeRunningJob() {
  const jobs = (state.data && state.data.jobs) || [];
  const running = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (running && $("#job-box") && !$("#job-box").dataset.watching) watchJob(running.id, `Working on ${running.kind}`);
}

function watchJob(id, label) {
  const box = $("#job-box");
  if (!box) return;
  box.dataset.watching = id;
  const status = h("div", { class: "status-line" }, h("span", { class: "spinner" }), h("span", {}, label));
  const log = h("div", { class: "log" });
  box.replaceChildren(h("div", { class: "panel" }, status, h("div", { style: "height:10px" }), log));
  const tick = async () => {
    let job;
    try { job = await api(`/jobs/${id}`); } catch { return setTimeout(tick, 1500); }
    log.textContent = job.log;
    log.scrollTop = log.scrollHeight;
    if (job.status === "done") { toast("Done"); return refresh(); }
    if (job.status === "failed") {
      status.replaceChildren(h("strong", { class: "error" }, "Stopped: "), h("span", {}, job.error || "failed"));
      if (state.pid) state.data = await api(`/projects/${state.pid}`);
      return;
    }
    setTimeout(tick, 1000);
  };
  tick();
}

/* ---------- charts ---------- */
function lineChart(series, { height = 160, seconds = null } = {}) {
  const W = 640, H = height, pad = 24;
  const all = series.flatMap((x) => x.values);
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const n = Math.max(...series.map((x) => x.values.length));
  const sx = (i) => pad + (i / Math.max(n - 1, 1)) * (W - pad - 6);
  const sy = (v) => H - pad - ((v - lo) / span) * (H - pad - 8);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Signal preview" },
    s("line", { x1: pad, y1: H - pad, x2: W - 6, y2: H - pad, class: "axis" }),
    s("text", { x: pad, y: H - 6 }, "0 s"),
    seconds ? s("text", { x: W - 40, y: H - 6 }, `${seconds.toFixed(1)} s`) : null,
    s("text", { x: 0, y: 12 }, fmtNum(hi, 2)),
    s("text", { x: 0, y: H - pad }, fmtNum(lo, 2)),
    ...series.map((x, k) => s("polyline", {
      points: x.values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" "),
      fill: "none", stroke: x.color || SERIES[k % 4], "stroke-width": 1.2, "stroke-linejoin": "round",
    })));
  return h("div", {}, svg, h("div", { class: "legend" }, series.map((x, k) => h("span", {}, h("i", { style: `background:${x.color || SERIES[k % 4]}` }), x.name))));
}

function histogram(viz) {
  const W = 640, H = 200, pad = 26;
  const names = Object.keys(viz.hist);
  const bins = viz.bins;
  const max = Math.max(1, ...names.flatMap((c) => viz.hist[c]));
  const x0 = bins[0], x1 = bins[bins.length - 1];
  const sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - pad - 10);
  const sy = (c) => H - pad - (c / max) * (H - pad - 10);
  const step = (W - pad - 10) / (bins.length - 1);
  const shapes = names.map((name, k) => {
    const pts = viz.hist[name].map((c, i) => `${(pad + i * step).toFixed(1)},${sy(c).toFixed(1)} ${(pad + (i + 1) * step).toFixed(1)},${sy(c).toFixed(1)}`).join(" ");
    return s("polyline", { points: `${pad},${H - pad} ${pts} ${W - 10},${H - pad}`, fill: SERIES[k % 4], "fill-opacity": 0.18, stroke: SERIES[k % 4], "stroke-width": 1.4 });
  });
  const lines = (viz.thresholds || []).map((t) => [
    s("line", { x1: sx(t), x2: sx(t), y1: 8, y2: H - pad, stroke: "var(--block)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }),
    s("text", { x: Math.min(sx(t) + 4, W - 80), y: 18 }, `alert above ${fmtNum(t, 4)}`),
  ]);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": `Distribution of ${viz.feature_label}` },
    s("line", { x1: pad, y1: H - pad, x2: W - 10, y2: H - pad, class: "axis" }),
    s("text", { x: pad, y: H - 8 }, fmtNum(x0, 3)), s("text", { x: W - 60, y: H - 8 }, fmtNum(x1, 3)),
    ...shapes, ...lines.flat());
  return h("div", {}, svg, h("div", { class: "legend" }, names.map((n, k) => h("span", {}, h("i", { style: `background:${SERIES[k % 4]}` }), n))));
}

function confusion(metrics, classes) {
  if (!metrics.confusion) return null;
  const max = Math.max(1, ...metrics.confusion.flat());
  return h("table", { class: "confusion" },
    h("tr", {}, h("th", {}, "true \\ predicted"), classes.map((c) => h("th", { class: "num" }, c))),
    metrics.confusion.map((row, i) => h("tr", {}, h("th", {}, classes[i]),
      row.map((v, j) => h("td", { class: "num", style: `background: color-mix(in srgb, ${i === j ? "var(--measured)" : "var(--block)"} ${Math.round((v / max) * 35)}%, transparent)` }, v)))));
}

/* ---------- step 1: describe ---------- */
const EXAMPLES = [
  "Alert me when my pump vibration looks abnormal. It runs on battery and reports over Wi-Fi.",
  "Detect harsh braking, harsh acceleration and sharp turns in delivery vans. Bluetooth to the driver's phone, budget $4.",
  "Tell whether a worker is resting, walking, running or on stairs from a wrist sensor. Coin cell, long battery life.",
];

async function describeView() {
  const d = state.data;
  const area = h("textarea", { id: "desc", placeholder: "For example: alert me when my pump vibration looks abnormal…" },
    d ? d.project.description : "");
  const submit = async () => {
    if (!area.value.trim()) return toast("Describe what the device should detect");
    const res = await api("/projects", { json: { description: area.value } });
    state.pid = res.project.id;
    state.data = await api(`/projects/${state.pid}`);
    go(state.pid, "vendor");
  };
  const view = h("div", {},
    h("h1", {}, "What should your device notice?"),
    h("p", { class: "lede" }, "Write it the way you would tell a colleague. No machine-learning words needed; the platform picks windows, features and models for you."),
    h("div", { class: "panel" }, area,
      h("div", { class: "chips", style: "margin-top:12px" }, EXAMPLES.map((ex) => h("button", { class: "chip", onclick: () => { area.value = ex; } }, ex.split(".")[0]))),
      h("div", { class: "row", style: "margin-top:16px" }, h("button", { class: "primary", onclick: submit }, d ? "Start a new project from this" : "Continue"))));
  if (d && d.project.spec) {
    const sp = d.project.spec;
    view.append(h("div", { class: "panel" },
      h("p", { class: "voice small" }, `Understood as ${recipeTitle(sp.recipe).toLowerCase()}: ${sp.task === "anomaly" ? "flag anything unusual" : "name which event happened"}.`),
      pendingQuestions(["recipe", "task", "connectivity", "power", "priority"])));
  }
  return view;
}

function recipeTitle(id) {
  const r = state.recipes.find((x) => x.id === id);
  return r ? r.title : id;
}

function pendingQuestions(types) {
  const pend = ((state.data && state.data.pending_decisions) || []).filter((p) => types.includes(p.decision_type));
  if (!pend.length) return null;
  return h("div", {}, pend.slice(0, 3).map((p) => decisionPrompt(p)));
}

function decisionPrompt(p) {
  return h("div", { class: "row", style: "margin:10px 0" },
    h("span", {}, p.follow_up || p.question),
    h("div", { class: "chips" }, p.options.map((o) => h("button", {
      class: "chip" + (o === p.answer ? " on" : ""),
      onclick: async () => {
        await api(`/decisions/${p.id}`, { method: "PATCH", json: { choice: o } });
        if (["recipe", "task", "connectivity", "power", "priority"].includes(p.decision_type)) {
          await api(`/projects/${state.pid}/spec`, { method: "PATCH", json: { [p.decision_type]: o } });
        }
        toast(`Saved: ${o}`);
        refresh();
      },
    }, o))));
}

/* ---------- step 2: vendor ---------- */
async function vendorView() {
  const sp = state.data.project.spec;
  const vendors = [{ id: "any", name: "Any vendor", note: "neutral comparison" }, ...state.vendors.map((v) => ({ ...v, note: "" }))];
  const form = { vendor: sp.vendor, connectivity: sp.connectivity, power: sp.power, budget_usd_per_unit: sp.budget_usd_per_unit, priority: sp.priority, max_latency_ms: sp.max_latency_ms };
  const vendorButtons = h("div", { class: "choice-grid" });
  const drawVendors = () => vendorButtons.replaceChildren(...vendors.map((v) => h("button", {
    class: "choice" + (form.vendor === v.id ? " on" : ""), onclick: () => { form.vendor = v.id; drawVendors(); },
  }, h("strong", {}, v.name), h("span", {}, v.id === "any" ? "compare every vendor"
    : (state.chips || []).filter((c) => c.vendor === v.id).map((c) => c.family).join(", ")))));
  drawVendors();
  const select = (key, options) => h("select", { onchange: (e) => { form[key] = e.target.value; } },
    options.map(([v, label]) => h("option", { value: v, selected: form[key] === v }, label)));
  const save = async () => {
    await api(`/projects/${state.pid}/spec`, { method: "PATCH", json: form });
    toast("Saved");
    state.data = await api(`/projects/${state.pid}`);
    go(state.pid, "data");
  };
  return h("div", {},
    h("h1", {}, "Pick a vendor, or let the platform compare"),
    h("p", { class: "lede" }, "If you already use a chip maker, choose it. Otherwise pick any vendor and get a neutral ranking. The answers below come from your description; change anything that is wrong."),
    h("div", { class: "panel" }, vendorButtons),
    h("div", { class: "panel" }, h("div", { class: "grid3" },
      h("label", { class: "field" }, "How it talks", select("connectivity", [["wifi", "Wi-Fi"], ["ble", "Bluetooth LE"], ["cellular", "Cellular (external modem)"], ["none", "No radio"], ["any", "Not decided"]])),
      h("label", { class: "field" }, "Power", select("power", [["battery", "Battery"], ["plugged", "Plugged in"]])),
      h("label", { class: "field" }, "Budget per chip (USD)", h("input", { type: "number", step: "0.1", min: "0", value: form.budget_usd_per_unit ?? "", placeholder: "no limit",
        oninput: (e) => { form.budget_usd_per_unit = e.target.value === "" ? null : Number(e.target.value); } })),
      h("label", { class: "field" }, "What matters most", select("priority", [["battery", "Battery life"], ["cost", "Low cost"], ["speed", "Speed"]])),
      h("label", { class: "field" }, "Answer within (ms)", h("input", { type: "number", min: "1", value: form.max_latency_ms, oninput: (e) => { form.max_latency_ms = Number(e.target.value); } })),
    ), h("div", { class: "row", style: "margin-top:18px" }, h("button", { class: "primary", onclick: save }, "Save and continue"))));
}

/* ---------- step 3: data ---------- */
async function dataView() {
  const d = state.data;
  const drop = h("label", { class: "drop" }, h("strong", {}, "Drop CSV files here"), h("br"), "or click to choose. One file per recording works best; a column that names each recording also works.",
    h("input", { type: "file", accept: ".csv", multiple: true, class: "sr-only", onchange: (e) => upload(e.target.files) }));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); });
  const busy = h("div", { class: "status-line" });
  async function upload(files) {
    if (!files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    busy.replaceChildren(h("span", { class: "spinner" }), "Reading and checking…");
    try {
      const res = await fetch(`/api/projects/${state.pid}/datasets`, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail);
      await refresh();
    } catch (err) { busy.replaceChildren(h("span", { class: "error" }, err.message)); }
  }
  const demo = async (name) => {
    busy.replaceChildren(h("span", { class: "spinner" }), "Generating demo data and checking it…");
    try { await api(`/projects/${state.pid}/demo-data`, { json: { dataset: name } }); await refresh(); }
    catch (err) { busy.replaceChildren(h("span", { class: "error" }, err.message)); }
  };
  const view = h("div", {},
    h("h1", {}, "Upload sensor recordings"),
    h("p", { class: "lede" }, "Each file needs one column per sensor axis, and ideally a time column and a label column. The platform works out which is which and checks the data before anything is trained."),
    h("div", { class: "panel" }, drop, h("div", { class: "row", style: "margin-top:12px" }, h("span", { class: "muted" }, "No data yet? Try a demo:"),
      ["pump", "driving", "activity"].map((n) => h("button", { class: "chip", onclick: () => demo(n) }, n))), h("div", { style: "margin-top:10px" }, busy)));
  if (!d.dataset) return view;
  const ds = d.dataset;
  const pendingRoles = new Set(d.pending_decisions.filter((p) => p.decision_type === "column_role").map((p) => p.decision_key));
  const roles = { ...ds.roles };
  const roleRows = ds.summary.columns.map((c) => h("tr", { class: pendingRoles.has(c.name.toLowerCase()) ? "unsure" : "" },
    h("td", {}, h("strong", {}, c.name)),
    h("td", {}, h("select", { onchange: (e) => { roles[c.name] = e.target.value; } },
      ["time", "sensor_axis", "label", "recording_id", "ignore"].map((r) => h("option", { value: r, selected: roles[c.name] === r }, r.replace("_", " "))))),
    h("td", {}, c.dtype), h("td", { class: "num" }, c.min !== null && c.min !== undefined ? fmtNum(c.min, 3) : "–"),
    h("td", { class: "num" }, c.max !== null && c.max !== undefined ? fmtNum(c.max, 3) : "–"),
    h("td", {}, c.sample.join(", "))));
  const labels = (ds.label_map && ds.label_map.labels) || [];
  let normal = ds.label_map && ds.label_map.normal_label;
  const normalPicker = d.project.spec.task === "anomaly" && labels.length
    ? h("label", { class: "field", style: "max-width:280px;margin-top:12px" }, "Which label means normal operation?",
      h("select", { onchange: (e) => { normal = e.target.value; } }, labels.map((l) => h("option", { value: l, selected: l === normal }, l)))) : null;
  const applyRoles = async () => {
    busy.replaceChildren(h("span", { class: "spinner" }), "Re-checking with your choices…");
    await api(`/projects/${state.pid}/datasets/roles`, { method: "PUT", json: { roles, normal_label: normal } });
    await refresh();
  };
  view.append(
    h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "What each column is"),
      pendingRoles.size ? h("p", { class: "muted" }, "Highlighted rows are guesses the platform was unsure about. Check them.") : null,
      h("table", {}, h("tr", {}, ["Column", "Role", "Type", "Min", "Max", "Sample"].map((x, i) => h("th", { class: i === 3 || i === 4 ? "num" : "" }, x))), roleRows),
      normalPicker, h("div", { class: "row", style: "margin-top:14px" }, h("button", { onclick: applyRoles }, "Apply and re-check"))),
    h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Checks"), statsLine(ds.stats),
      ds.issues.length ? h("ul", { class: "issues" }, ds.issues.map((i) => h("li", {}, h("span", { class: `sev ${i.severity}` }, i.severity), h("span", {}, i.message))))
        : h("p", {}, "No problems found."),
      h("div", { class: "row", style: "margin-top:16px" },
        h("button", { class: "primary", disabled: ds.issues.some((i) => i.severity === "block"), onclick: () => go(state.pid, "algorithm") }, "Continue to the algorithm"))));
  if (d.preview && d.preview.samples.length) {
    view.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "A look at the signal"),
      h("div", { class: "grid2" }, d.preview.samples.map((smp) => h("div", {},
        h("p", { class: "muted", style: "margin:0 0 4px" }, `${smp.label} — ${smp.recording}`),
        lineChart(Object.entries(smp.series).map(([name, values]) => ({ name, values })), { height: 150, seconds: Object.values(smp.series)[0].length / smp.rate }))))));
  }
  return view;
}

function statsLine(st) {
  if (!st || !st.sample_rate_hz) return null;
  const classes = Object.entries(st.class_seconds || {}).map(([k, v]) => `${k} ${(v / 60).toFixed(1)} min`).join(", ");
  return h("p", {}, `${st.recordings} recordings, ${(st.duration_s / 60).toFixed(1)} minutes at ${fmtNum(st.sample_rate_hz)} readings per second. `, classes ? `Per kind: ${classes}.` : "");
}

/* ---------- step 4: algorithm ---------- */
async function algorithmView() {
  const d = state.data;
  const tr = d.training;
  const chosen = d.candidates.find((c) => c.selected);
  const view = h("div", {},
    h("h1", {}, "Find the simplest thing that works"),
    h("p", { class: "lede" }, "The platform tries plain rules first, then classic machine learning, then a tiny neural network, and stops at the first one that meets the target on recordings it never trained on."),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => runJob(`/projects/${state.pid}/train`, {}, "Training") }, chosen ? "Train again" : "Train")),
    jobBox());
  if (!tr) return view;
  if (tr.explanation) view.append(...explanationBlock(tr.explanation));
  if (tr.data_request) view.append(h("div", { class: "panel" }, h("p", { class: "voice small error" }, tr.data_request), h("button", { onclick: () => go(state.pid, "data") }, "Go back to the data")));
  const byRung = { 1: [], 2: [], 3: [] };
  d.candidates.forEach((c) => byRung[c.rung].push(c));
  const metricText = (c) => {
    const m = c.metrics || {};
    if ("accuracy" in m) return `accuracy ${pct(m.accuracy, 1)}, per event ${pct(m.balanced_accuracy, 1)}`;
    if ("false_alarm_rate" in m) return `faults caught ${m.detection_rate === null ? "n/a" : pct(m.detection_rate, 1)}, false alarms ${pct(m.false_alarm_rate, 1)}`;
    return c.note;
  };
  view.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "The ladder"),
    h("div", { class: "ladder" }, [1, 2, 3].map((r) => h("div", { class: "rung" },
      h("h3", {}, `Rung ${r}`, h("small", {}, (tr.rung_names || {})[r])),
      h("div", {}, byRung[r].length ? byRung[r].map((c) => h("div", { class: "cand" + (c.selected ? " chosen" : "") },
        h("span", {}, h("strong", {}, c.algorithm.replaceAll("_", " ")), " ", h("span", { class: "muted" }, c.note || "")),
        h("span", { class: "metrics" }, metricText(c), " ", c.passed ? "✓ meets target" : "✗"))) : h("div", { class: "cand untried" }, chosen && chosen.rung < r ? "not needed" : "not tried")))))));
  const viz = tr.viz || {};
  const grid = h("div", { class: "grid2" });
  if (viz.hist && Object.keys(viz.hist).length) grid.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "The most useful measurement"), h("p", { class: "muted", style: "margin-top:0" }, `${viz.feature_label[0].toUpperCase()}${viz.feature_label.slice(1)}, for each kind of data`), histogram(viz)));
  if (chosen && chosen.metrics.confusion) grid.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Test results"), confusion(chosen.metrics, tr.classes)));
  if (chosen && "false_alarm_rate" in chosen.metrics) {
    const m = chosen.metrics;
    grid.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Test results"),
      h("div", { class: "stat" }, h("span", {}, "Fault windows caught"), h("strong", {}, m.detection_rate === null ? "no fault data" : pct(m.detection_rate, 1))),
      h("div", { class: "stat" }, h("span", {}, "False alarms on normal windows"), h("strong", {}, pct(m.false_alarm_rate, 1))),
      h("div", { class: "stat" }, h("span", {}, "Test windows"), h("strong", {}, `${m.test_normal_windows} normal, ${m.test_fault_windows} fault`)),
      h("p", { class: "muted" }, `Split: ${tr.split}.`)));
  }
  view.append(grid);
  if (chosen) view.append(h("div", { class: "row" }, h("button", { class: "primary", onclick: () => go(state.pid, "rank") }, "Continue to chip ranking")));
  return view;
}

function explanationBlock(text) {
  const rules = text.match(/^(Rules were enough): (if .*)\.$/);
  if (!rules || !rules[2].includes("; ")) return [h("p", { class: "voice" }, text)];
  return [h("p", { class: "voice" }, `${rules[1]}. The device follows these rules:`),
    h("ul", { class: "rules" }, rules[2].split("; ").map((r) => h("li", {}, r)))];
}

/* ---------- step 5: rank ---------- */
async function rankView() {
  const d = state.data;
  const results = d.chip_results;
  const ranked = results.filter((r) => r.rank);
  const filtered = results.filter((r) => !r.rank);
  const view = h("div", {},
    h("h1", {}, "Which chip should it run on?"),
    h("p", { class: "lede" }, "Every chip that passes your filters gets a real build with its own compiler and runs in an emulator. Each number is tagged: measured, emulated or estimated."),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => runJob(`/projects/${state.pid}/rank`, {}, "Building for each chip") }, ranked.length ? "Rank again" : "Rank chips")),
    jobBox());
  if (!results.length) return view;
  view.append(h("div", { class: "grid3", style: "margin-top:18px" }, ranked.slice(0, 3).map((r) => {
    const m = r.measurement;
    return h("div", { class: "panel chipcard" },
      h("div", { class: "rank" }, r.rank),
      h("h3", {}, chipName(r.chip_id)),
      h("div", { class: "stat" }, h("span", {}, "Time per prediction"), tagged(m.latency_ms, (v) => `${fmtNum(v, 3)} ms`)),
      h("div", { class: "stat" }, h("span", {}, "Flash"), tagged(m.flash_bytes, kb)),
      h("div", { class: "stat" }, h("span", {}, "RAM"), tagged(m.ram_bytes, kb)),
      h("div", { class: "stat" }, h("span", {}, "Battery (MCU + sensor)"), tagged(m.battery_days, (v) => `${fmtNum(v)} days`)),
      h("div", { class: "stat" }, h("span", {}, "Price"), tagged(m.price_usd, (v) => `$${fmtNum(v, 2)}`)),
      h("ul", {}, r.reasons.map((x) => h("li", {}, x)), r.tradeoffs.map((x) => h("li", { class: "trade" }, x))),
      h("div", { class: "row", style: "margin-top:12px" }, h("button", { class: r.rank === 1 ? "primary" : "", onclick: () => go(state.pid, `build?chip=${r.chip_id}`) }, "Build for this chip")));
  })));
  view.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "All chips"),
    h("table", {}, h("tr", {}, ["#", "Chip", "Score", "Instructions", "Time", "Energy", "Flash", "RAM", "Parity"].map((x) => h("th", {}, x))),
      ranked.map((r) => {
        const m = r.measurement;
        return h("tr", {}, h("td", {}, r.rank), h("td", {}, chipName(r.chip_id)), h("td", { class: "num" }, fmtNum(r.score)),
          h("td", { class: "num" }, tagged(m.instructions, (v) => fmtNum(v))), h("td", { class: "num" }, tagged(m.latency_ms, (v) => `${fmtNum(v, 3)} ms`)),
          h("td", { class: "num" }, tagged(m.energy_uj, (v) => `${fmtNum(v, 2)} µJ`)), h("td", { class: "num" }, tagged(m.flash_bytes, kb)),
          h("td", { class: "num" }, tagged(m.ram_bytes, kb)), h("td", {}, m.parity_ok ? `✓ ${m.parity_detail.label_matches}/${m.parity_detail.windows}` : "✗"));
      }))),
  );
  if (ranked.length) view.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Time per prediction"), latencyBars(ranked)));
  if (filtered.length) {
    view.append(h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Left out"),
      h("table", {}, filtered.map((r) => h("tr", {}, h("td", {}, chipName(r.chip_id)), h("td", { class: "muted" }, r.measurement.filtered_reason || r.measurement.error || r.measurement.status))))));
  }
  return view;
}

function latencyBars(ranked) {
  const vals = ranked.map((r) => (r.measurement.latency_ms && r.measurement.latency_ms.value) || 0);
  const max = Math.max(...vals, 1e-9);
  const W = 640, row = 26, H = ranked.length * row + 10, left = 150;
  return s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Time per prediction by chip" },
    ...ranked.map((r, i) => {
      const w = Math.max(2, (vals[i] / max) * (W - left - 90));
      return [s("text", { x: 0, y: i * row + 17 }, chipName(r.chip_id).split(" (")[0]),
        s("rect", { x: left, y: i * row + 6, width: w, height: 14, rx: 3, fill: i === 0 ? "var(--trace)" : "var(--rule)" }),
        s("text", { x: left + w + 6, y: i * row + 17 }, `${fmtNum(vals[i], 3)} ms`)];
    }).flat());
}

function chipName(id) {
  const c = (state.chips || []).find((x) => x.id === id);
  return c ? `${c.family} (${c.part_number})` : id;
}

/* ---------- step 6: build ---------- */
async function buildView() {
  const d = state.data;
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const ranked = d.chip_results.filter((r) => r.rank);
  let chip = params.get("chip") || (ranked[0] && ranked[0].chip_id) || "esp32s3";
  const picker = h("select", { onchange: (e) => { chip = e.target.value; } },
    (state.chips || []).map((c) => h("option", { value: c.id, selected: c.id === chip }, `${c.family} (${c.part_number})`)));
  const view = h("div", {},
    h("h1", {}, "Build, prove, download"),
    h("p", { class: "lede" }, "The final build uses the chip's pinned toolchain and a fixed seed. Golden test windows run through the compiled firmware in the emulator; the package is only offered if every answer matches Python."),
    h("div", { class: "panel row" }, h("label", { class: "field" }, "Chip", picker),
      h("button", { class: "primary", style: "align-self:end", onclick: () => runJob(`/projects/${state.pid}/build`, { chip_id: chip }, `Building for ${chip}`) }, "Build package")),
    jobBox());
  if (!d.builds.length) return view;
  const b = d.builds[0];
  const parity = b.parity;
  const golden = parity.windows || 0;
  const failedIds = new Set((parity.failures || []).map((f) => f.id));
  view.append(h("div", { class: "panel" },
    h("p", { class: "voice small" }, b.passed
      ? `${parity.label_matches} of ${golden} answers from the compiled firmware match Python on ${chipName(b.chip_id)}. Package ${b.version} is signed and ready.`
      : `Blocked on ${chipName(b.chip_id)}: ${parity.blocked_reason || "the firmware's answers did not match Python."}`),
    h("div", { class: "parity-grid", "aria-label": "Parity per golden window" }, Array.from({ length: golden }, (_, i) =>
      h("span", { class: failedIds.has(i) ? "bad" : "", title: `window ${i}` }, i))),
    h("p", { class: "muted" }, `Emulator: ${parity.measurement.emulator}. Worst feature difference used ${(100 * (parity.worst_tolerance_used || 0)).toFixed(2)}% of its allowed tolerance (${parity.worst_feature || "–"}).`),
    b.package_path ? h("div", { class: "row" }, h("a", { href: `/api/builds/${b.id}/download` }, h("button", { class: "primary" }, "Download package (.zip)"))) : null));
  if (b.package_path) {
    const reportBox = h("div", { class: "panel report" }, "Loading report…");
    api(`/builds/${b.id}/report`).then((md) => reportBox.replaceChildren(markdown(md))).catch((e) => { reportBox.textContent = e.message; });
    const code = h("pre", { class: "code" });
    const codeTabs = h("div", { class: "chips" }, ["tinyml_model.h", "main.c", "TinyMLModel.hpp", "tinyml_model.c"].map((f) => h("button", {
      class: "chip", onclick: async (e) => { codeTabs.querySelectorAll(".chip").forEach((c) => c.classList.remove("on")); e.target.classList.add("on"); code.textContent = await api(`/builds/${b.id}/files/${f}`); },
    }, f)));
    view.append(reportBox, h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Code"), codeTabs, code), askBox(b));
    api(`/builds/${b.id}/files/tinyml_model.h`).then((t) => { code.textContent = t; codeTabs.firstChild.classList.add("on"); });
  }
  return view;
}

function askBox(build) {
  const q = h("input", { type: "text", placeholder: "For example: how do I call this from an ESP-IDF task at 12 kHz?", style: "flex:1;min-width:260px" });
  const out = h("div", { style: "margin-top:12px" });
  const ask = async () => {
    if (!q.value.trim()) return;
    out.replaceChildren(h("div", { class: "status-line" }, h("span", { class: "spinner" }), "Asking…"));
    try {
      const res = await api(`/projects/${state.pid}/ask`, { json: { question: q.value, build_id: build.id } });
      if (res.answer) out.replaceChildren(markdown(res.answer));
      else {
        const pre = h("pre", { class: "code" }, res.prompt);
        out.replaceChildren(h("p", { class: "muted" }, res.error ? `Claude Code was not available (${res.error}). ` : "", "Copy this into Claude chat:"),
          h("button", { onclick: () => { navigator.clipboard.writeText(res.prompt); toast("Copied"); } }, "Copy prompt"), pre);
      }
    } catch (err) { out.replaceChildren(h("p", { class: "error" }, err.message)); }
  };
  return h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Ask about this package"),
    h("p", { class: "muted" }, "Answered by Claude Code on this machine when the assistant is set to Claude Code; otherwise you get a prompt to paste into Claude chat."),
    h("div", { class: "row" }, q, h("button", { onclick: ask }, "Ask")), out);
}

function markdown(md) {
  const root = h("div", {});
  let list = null;
  const inline = (text) => {
    const span = h("span");
    text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach((part) => {
      if (part.startsWith("**") && part.endsWith("**")) span.append(h("strong", {}, part.slice(2, -2)));
      else if (part.startsWith("`") && part.endsWith("`")) span.append(h("code", {}, part.slice(1, -1)));
      else span.append(part);
    });
    return span;
  };
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    const item = line.match(/^(\s*)[-*] (.*)/) || line.match(/^(\s*)\d+\. (.*)/);
    if (item) { if (!list) { list = h("ul"); root.append(list); } list.append(h("li", {}, inline(item[2]))); continue; }
    list = null;
    if (!line.trim()) continue;
    if (line.startsWith("## ")) root.append(h("h2", {}, line.slice(3)));
    else if (line.startsWith("# ")) root.append(h("h1", {}, line.slice(2)));
    else if (line.startsWith("> ")) root.append(h("blockquote", {}, line.slice(2)));
    else root.append(h("p", {}, inline(line)));
  }
  return root;
}

/* ---------- side pages ---------- */
async function decisionsView() {
  const rows = await api(`/decisions${state.pid ? "?project_id=" + state.pid : ""}`);
  const rules = await api("/learned-rules");
  return h("div", {},
    h("h1", {}, "Decision log"),
    h("p", { class: "lede" }, "Every judgement call: what the rules, the assistant or you decided. Corrections you repeat three times become fixed rules. Raw sensor data is never stored here."),
    h("div", { class: "panel" }, rows.length ? h("table", {},
      h("tr", {}, ["Type", "About", "Answer", "Confidence", "Source", "Final", "Build"].map((x) => h("th", {}, x))),
      rows.map((r) => h("tr", { class: r.final_choice ? "" : "unsure" },
        h("td", {}, r.decision_type), h("td", {}, r.decision_key), h("td", {}, r.answer), h("td", { class: "num" }, pct(r.confidence)),
        h("td", {}, r.source), h("td", {}, h("select", {
          onchange: async (e) => { await api(`/decisions/${r.id}`, { method: "PATCH", json: { choice: e.target.value } }); toast("Recorded"); },
        }, h("option", { value: "" }, r.final_choice ? "" : "choose…"), r.options.map((o) => h("option", { value: o, selected: o === r.final_choice }, o)))),
        h("td", {}, r.build_passed === null ? "–" : r.build_passed ? "passed" : "failed")))) : h("p", { class: "empty" }, "No decisions yet.")),
    h("div", { class: "panel" }, h("h2", { style: "margin-top:0" }, "Learned rules"), rules.length
      ? h("table", {}, rules.map((r) => h("tr", {}, h("td", {}, r.decision_type), h("td", {}, r.decision_key), h("td", {}, `→ ${r.choice}`), h("td", { class: "muted" }, `${r.support} corrections`))))
      : h("p", { class: "muted" }, "None yet.")));
}

async function toolchainsView() {
  const rep = await api("/doctor");
  state.doctor = rep;
  return h("div", {},
    h("h1", {}, "Toolchains on this machine"),
    h("p", { class: "lede" }, "A chip can be ranked only when its compiler and emulator are installed. Everything runs locally."),
    h("div", { class: "panel" }, h("p", {}, "Host C compiler: ", rep.host_compiler.ok ? rep.host_compiler.version : h("span", { class: "error" }, rep.host_compiler.why)),
      h("table", {}, h("tr", {}, ["Chip", "Ready", "Backend", "Toolchain or what to install"].map((x) => h("th", {}, x))),
        rep.chips.map((c) => {
          const usable = c.options.find((o) => o.ok);
          return h("tr", {}, h("td", {}, c.name), h("td", {}, h("span", { class: "dot", style: `background:${c.ready ? "var(--measured)" : "var(--block)"}` }), c.ready ? "yes" : "no"),
            h("td", {}, c.backend || "–"), h("td", { class: usable ? "" : "muted" }, usable ? usable.toolchain : c.options.map((o) => `${o.why}. ${o.hint}`).join(" | ")));
        }))));
}

async function settingsView() {
  const st = await api("/settings");
  const rep = state.doctor || await api("/doctor");
  const choose = async (provider) => { await api("/settings", { method: "PUT", json: { provider } }); toast("Saved"); render(); };
  const opt = (id, title, text) => h("button", { class: "choice" + (st.assistant.provider === id ? " on" : ""), onclick: () => choose(id) }, h("strong", {}, title), h("span", {}, text));
  return h("div", {},
    h("h1", {}, "Who answers when the rules are unsure"),
    h("p", { class: "lede" }, "Rules decide first. Only when they are unsure does the platform ask an assistant, and you always have the final say. Jev plugs in here later."),
    h("div", { class: "panel" }, h("div", { class: "choice-grid" },
      opt("off", "Ask me", "No AI. Unsure points are shown to you as questions."),
      opt("claude_code", "Claude Code", rep.assistant.claude_code_found ? `Found: ${rep.assistant.claude_version}. Runs headless with your login.` : "Not found on PATH. Install Claude Code to use it."),
      opt("claude_chat", "Claude chat", "Shows a ready prompt to paste into Claude chat, then you paste the answer back."))),
    h("p", { class: "muted" }, `Confidence cut-offs: ${Object.entries(st.assistant.cutoffs).map(([k, v]) => `${k} ${v}`).join(", ")}.`));
}

/* ---------- boot ---------- */
async function boot() {
  [state.recipes, state.vendors, state.chips] = await Promise.all([api("/recipes"), api("/vendors"), api("/chips")]);
  $("#project-select").addEventListener("change", (e) => { location.hash = e.target.value ? `#/p/${e.target.value}/describe` : "#/"; });
  window.addEventListener("hashchange", render);
  api("/doctor").then((rep) => {
    state.doctor = rep;
    const ready = rep.chips.filter((c) => c.ready).length;
    $("#doctor-link").replaceChildren(h("span", { class: "dot", style: `background:${ready === rep.chips.length ? "var(--measured)" : "var(--estimated)"}` }), `Toolchains ${ready}/${rep.chips.length}`);
  });
  render();
}

boot();
