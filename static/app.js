const state = {
  models: [],
  favorites: [],
  modelsExpanded: false,
  welcomeSlide: 0,
  welcomeTimer: null,
  activeTab: "load",
  previousTab: "load",
  hfModels: [],
  hfRepo: null,
  hfJobTimer: null,
  hfLimit: 20,
  hfHasMore: false,
  hfSelectedFiles: new Set(),
  gpus: [],
  servers: [],
  settings: {},
  selectedModel: null,
  chatServerId: null,
  chatMessages: [],
  analyzerServerId: null,
  analyzerTimer: null,
  analyzerBusy: false,
  azHist: [],
  azChartsAt: 0,
  azForceChart: false,
  azResWindowS: 1800,
  azTlWindowS: 1800,
  ioTraceId: null,
  ioFollowLive: true,
  ioQueryTs: null,
  ioTimer: null,
  ioBusy: false,
  ioNeedsRefresh: false,
  ioFetchGen: 0,
  ioLastStamp: "",
};

const $ = (sel) => document.querySelector(sel);

function formatApiError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === "string" ? d : d.msg || JSON.stringify(d)))
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return "Request failed";
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...opts.headers },
      ...opts,
    });
  } catch {
    throw new Error(
      "Cannot reach Lemur backend (is the server running?). Close this window and reopen from the desktop icon."
    );
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(formatApiError(err.detail) || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

function formatSize(gb) {
  return `${gb} GB`;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function initTooltips() {
  let el = null;
  let active = null;

  function ensure() {
    if (el) return el;
    el = document.createElement("div");
    el.className = "hub-tip";
    el.setAttribute("role", "tooltip");
    document.body.appendChild(el);
    return el;
  }

  function hide() {
    active = null;
    if (!el) return;
    el.classList.remove("visible");
  }

  function place(target) {
    const tip = ensure();
    const text = target.getAttribute("data-tip");
    if (!text) {
      hide();
      return;
    }
    tip.textContent = text;
    tip.classList.add("visible");
    const r = target.getBoundingClientRect();
    const pad = 8;
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let left = r.left;
    let top = r.bottom + pad;
    if (top + th > window.innerHeight - pad) top = r.top - th - pad;
    if (left + tw > window.innerWidth - pad) left = window.innerWidth - tw - pad;
    if (left < pad) left = pad;
    if (top < pad) top = pad;
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
  }

  function findTip(node) {
    if (!(node instanceof Element)) return null;
    return node.closest("[data-tip]");
  }

  document.addEventListener(
    "pointerover",
    (e) => {
      const t = findTip(e.target);
      if (!t || t === active) return;
      active = t;
      place(t);
    },
    true
  );
  document.addEventListener(
    "pointerout",
    (e) => {
      const t = findTip(e.target);
      if (!t || t !== active) return;
      const next = findTip(e.relatedTarget);
      if (next === active) return;
      hide();
    },
    true
  );
  document.addEventListener(
    "focusin",
    (e) => {
      const t = findTip(e.target);
      if (!t) return;
      active = t;
      place(t);
    },
    true
  );
  document.addEventListener(
    "focusout",
    (e) => {
      if (active && findTip(e.target) === active) hide();
    },
    true
  );
  document.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
}

function favoriteMatchesServer(favorite, server) {
  return (
    favorite.model_path === server.model_path &&
    Number(favorite.gpu) === Number(server.gpu) &&
    Number(favorite.ctx) === Number(server.ctx) &&
    Number(favorite.ngl) === Number(server.ngl) &&
    (favorite.spill || "none") === (server.spill || "none") &&
    Boolean(favorite.mtp) === Boolean(server.mtp) &&
    Number(favorite.mtp_draft_n || 2) === Number(server.mtp_draft_n || 2) &&
    Boolean(favorite.vision) === Boolean(server.vision)
  );
}

function favoriteMeta(favorite) {
  const parts = [
    `GPU ${favorite.gpu}`,
    `ctx ${formatCtx(Number(favorite.ctx))}`,
    `spill ${favorite.spill || "none"}`,
  ];
  if (favorite.mtp) parts.push(`mtp n=${favorite.mtp_draft_n || 2}`);
  if (favorite.vision) parts.push("vision");
  if (Number(favorite.ngl) !== 999) parts.push(`ngl ${favorite.ngl}`);
  return parts.join(" · ");
}

function renderFavorites() {
  const list = $("#favorite-list");
  const empty = $("#favorite-empty");
  if (!list || !empty) return;

  empty.classList.toggle("hidden", state.favorites.length > 0);
  list.classList.toggle("hidden", state.favorites.length === 0);
  if (!state.favorites.length) {
    list.innerHTML = "";
    return;
  }

  list.innerHTML = state.favorites
    .map((favorite) => {
      const running = state.servers.some(
        (server) =>
          ["running", "starting", "converting"].includes(server.status) &&
          favoriteMatchesServer(favorite, server)
      );
      return `
      <li class="favorite-item ${running ? "is-running" : ""}">
        <div class="favorite-copy">
          <span class="favorite-name">${escapeHtml(favorite.model_name || favorite.alias || "model")}</span>
          <span class="favorite-meta">${escapeHtml(favoriteMeta(favorite))}</span>
        </div>
        <div class="favorite-actions">
          <button type="button" class="btn btn-primary btn-favorite-start tip" data-id="${escapeHtml(favorite.id)}" data-tip="Start this exact saved configuration" ${running ? "disabled" : ""}>${running ? "running" : "start()"}</button>
          <button type="button" class="btn btn-ghost btn-favorite-load tip" data-id="${escapeHtml(favorite.id)}" data-tip="Load this preset into the launch form">load</button>
          <button type="button" class="btn btn-danger btn-favorite-delete tip" data-id="${escapeHtml(favorite.id)}" data-tip="Delete this saved preset" aria-label="Delete favorite">×</button>
        </div>
      </li>`;
    })
    .join("");

  list.querySelectorAll(".btn-favorite-start").forEach((button) => {
    button.addEventListener("click", () => startFavorite(button.dataset.id, button));
  });
  list.querySelectorAll(".btn-favorite-load").forEach((button) => {
    button.addEventListener("click", () => loadFavoriteIntoForm(button.dataset.id));
  });
  list.querySelectorAll(".btn-favorite-delete").forEach((button) => {
    button.addEventListener("click", () => deleteFavorite(button.dataset.id));
  });
}

function setModelsExpanded(expanded) {
  state.modelsExpanded = Boolean(expanded);
  const browser = $("#model-browser");
  const button = $("#btn-toggle-models");
  const panel = document.querySelector(".panel-models");
  if (!browser || !button) return;
  panel?.classList.toggle("models-expanded", state.modelsExpanded);
  browser.classList.toggle("hidden", !state.modelsExpanded);
  browser.setAttribute("aria-hidden", String(!state.modelsExpanded));
  button.setAttribute("aria-expanded", String(state.modelsExpanded));
  button.textContent = state.modelsExpanded ? "[hide models]" : "[all models]";
  if (state.modelsExpanded) $("#model-search")?.focus();
}

let modelsPanelDefaultApplied = false;

function applyModelsPanelDefault(prevFavoriteCount) {
  const count = state.favorites.length;
  if (!modelsPanelDefaultApplied) {
    setModelsExpanded(count === 0);
    modelsPanelDefaultApplied = true;
    return;
  }
  if (prevFavoriteCount === 0 && count > 0) {
    setModelsExpanded(false);
  } else if (prevFavoriteCount > 0 && count === 0) {
    setModelsExpanded(true);
  }
}

function setTabSelection(tab) {
  document.querySelectorAll(".app-tab").forEach((button) => {
    const active = button.dataset.tab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

function showMainTab(tab) {
  const next = tab === "load" ? "load" : "download";
  state.activeTab = next;
  state.previousTab = next;
  $("#tab-download").classList.toggle("hidden", next !== "download");
  $("#tab-load").classList.toggle("hidden", next !== "load");
  $("#analyzer-overlay").classList.add("hidden");
  $("#analyzer-overlay").setAttribute("aria-hidden", "true");
  document.body.classList.remove("analyzer-open");
  setTabSelection(next);
  closeAnalyzerIo();
  if (state.analyzerTimer) {
    clearInterval(state.analyzerTimer);
    state.analyzerTimer = null;
  }
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (!value) return "size unknown";
  const gib = value / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(gib >= 10 ? 1 : 2)} GB` : `${(value / 1024 ** 2).toFixed(0)} MB`;
}

function compactNumber(value) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(
    Number(value) || 0
  );
}

function shortDate(value) {
  if (!value) return "date unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "date unknown" : date.toLocaleDateString();
}

function renderHuggingFaceModels() {
  const results = $("#hf-results");
  if (!state.hfModels.length) {
    results.innerHTML = '<p class="muted">No GGUF models found.</p>';
    return;
  }
  let models = [...state.hfModels];
  const sort = $("#hf-sort")?.value || "downloads";
  const direction = Number($("#btn-hf-direction")?.dataset.direction || -1);
  if (sort === "name") {
    models.sort((a, b) => direction * a.id.localeCompare(b.id));
  }
  results.innerHTML = models
    .map((model) => `<article class="hf-model ${state.hfRepo?.id === model.id ? "selected" : ""}">
      <img class="hf-avatar" src="${escapeHtml(model.avatar_url)}" alt="${escapeHtml(model.author)} profile picture" loading="lazy">
      <button type="button" class="hf-model-select" data-repo="${escapeHtml(model.id)}">
        <span class="hf-model-name">${escapeHtml(model.id)}</span>
        <span class="hf-model-meta">↓ ${compactNumber(model.downloads)} · ♥ ${compactNumber(model.likes)} · ${model.gguf_count} GGUF · updated ${shortDate(model.updated)}</span>
        <span class="hf-badges"><span class="hf-badge">${escapeHtml(model.license || "license unknown")}</span>${model.gated ? '<span class="hf-badge gated">gated</span>' : ""}</span>
      </button>
      <a class="hf-author-link" href="${escapeHtml(model.author_url)}" target="_blank" rel="noreferrer">${escapeHtml(model.author)} ↗</a>
    </article>`)
    .join("");
  results.querySelectorAll(".hf-model-select").forEach((button) => {
    button.addEventListener("click", () => loadHuggingFaceModel(button.dataset.repo));
  });
}

async function searchHuggingFace(e, more = false) {
  if (e) e.preventDefault();
  const query = $("#hf-search-input").value.trim();
  if (query.length < 2) return;
  state.hfLimit = more ? Math.min(99, state.hfLimit + 20) : 20;
  $("#btn-hf-search").disabled = true;
  $("#hf-status").textContent = "Searching Hugging Face…";
  try {
    const sort = $("#hf-sort").value;
    const apiSort = sort === "name" ? "downloads" : sort;
    const direction = $("#btn-hf-direction").dataset.direction;
    const data = await api(
      `/api/huggingface/models?q=${encodeURIComponent(query)}&limit=${state.hfLimit}&sort=${encodeURIComponent(apiSort)}&direction=${direction}`
    );
    state.hfModels = data.models || [];
    state.hfHasMore = Boolean(data.has_more);
    state.hfRepo = null;
    renderHuggingFaceModels();
    $("#hf-detail").innerHTML = '<p class="muted">Select a model to see its GGUF files.</p>';
    $("#hf-result-count").textContent = `${state.hfModels.length}${state.hfHasMore ? "+" : ""} results`;
    $("#hf-status").textContent = ` · Search: ${query}`;
    $("#btn-hf-more").classList.toggle("hidden", !state.hfHasMore);
  } catch (err) {
    $("#hf-status").textContent = err.message;
  } finally {
    $("#btn-hf-search").disabled = false;
  }
}

async function loadHuggingFaceModel(repoId) {
  if (!repoId) return;
  $("#hf-status").textContent = `Loading ${repoId}…`;
  try {
    const path = repoId.split("/").map(encodeURIComponent).join("/");
    state.hfRepo = await api(`/api/huggingface/models/${path}`);
    state.hfSelectedFiles = new Set();
    renderHuggingFaceModels();
    renderHuggingFaceFiles();
    $("#hf-status").textContent = `${state.hfRepo.files.length} GGUF files available.`;
  } catch (err) {
    $("#hf-status").textContent = err.message;
  }
}

function quantizationLabel(name) {
  const match = name.toUpperCase().match(/(?:^|[-_/])(UD-)?(IQ\d_[A-Z0-9]+|Q\d_[A-Z0-9]+|BF16|F16|NVFP4)(?:[-_.]|$)/);
  return match ? `${match[1] || ""}${match[2]}` : "GGUF";
}

function isRecommendedQuant(name) {
  const upper = name.toUpperCase();
  return upper.includes("Q4_K_M") || upper.includes("Q4_0");
}

function updateHuggingFaceSelection() {
  const selected = state.hfRepo?.files.filter((file) => state.hfSelectedFiles.has(file.name)) || [];
  const total = selected.reduce((sum, file) => sum + Number(file.size || 0), 0);
  const summary = $("#hf-selection-summary");
  if (summary) summary.textContent = `${selected.length} selected · ${formatBytes(total)}`;
  const button = $("#btn-hf-download");
  if (button) {
    button.disabled = !selected.length;
    button.textContent = selected.length ? `download ${selected.length} file${selected.length === 1 ? "" : "s"}` : "download selected";
  }
}

function renderHuggingFaceFiles() {
  const repo = state.hfRepo;
  if (!repo) return;
  const rawQuery = $("#hf-file-filter")?.value || "";
  const query = rawQuery.toLowerCase();
  const visibleFiles = repo.files.filter((file) => file.name.toLowerCase().includes(query));
  const files = visibleFiles
    .map((file) => `<label class="hf-file">
      <input type="checkbox" class="hf-file-check" value="${escapeHtml(file.name)}" ${state.hfSelectedFiles.has(file.name) ? "checked" : ""}>
      <span><span class="hf-model-name">${escapeHtml(file.name)}</span><span class="hf-file-meta">${formatBytes(file.size)} · ${quantizationLabel(file.name)}</span></span>
      ${isRecommendedQuant(file.name) ? '<span class="hf-badge recommended">recommended</span>' : ""}
    </label>`)
    .join("");
  $("#hf-detail").innerHTML = `<div class="hf-detail-head">
      <h2>${escapeHtml(repo.id)}</h2>
      <span class="hf-model-meta">${repo.gated ? "HF_TOKEN is required · " : ""}<a href="${escapeHtml(repo.url)}" target="_blank" rel="noreferrer">open model card</a></span>
    </div>
    <div class="hf-file-toolbar">
      <input type="search" id="hf-file-filter" value="${escapeHtml(rawQuery)}" placeholder="Filter GGUF files">
      <button type="button" id="btn-hf-select-all" class="btn btn-ghost btn-tiny">select all</button>
      <button type="button" id="btn-hf-select-none" class="btn btn-ghost btn-tiny">clear</button>
      <span id="hf-selection-summary" class="hf-selection-summary"></span>
    </div>
    <div>${files || '<p class="muted">No matching GGUF files.</p>'}</div>
    <div class="hf-download-actions">
      <button type="button" id="btn-hf-download" class="btn btn-primary" disabled>download selected</button>
    </div>
    <div id="hf-download-progress" class="hf-download-progress hidden"></div>`;
  $("#hf-file-filter").addEventListener("input", (event) => {
    const position = event.target.selectionStart;
    renderHuggingFaceFiles();
    const next = $("#hf-file-filter");
    next.focus();
    next.setSelectionRange(position, position);
  });
  $("#btn-hf-select-all").addEventListener("click", () => {
    visibleFiles.forEach((file) => state.hfSelectedFiles.add(file.name));
    renderHuggingFaceFiles();
  });
  $("#btn-hf-select-none").addEventListener("click", () => {
    state.hfSelectedFiles.clear();
    renderHuggingFaceFiles();
  });
  document.querySelectorAll(".hf-file-check").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.hfSelectedFiles.add(input.value);
      else state.hfSelectedFiles.delete(input.value);
      updateHuggingFaceSelection();
    });
  });
  $("#btn-hf-download")?.addEventListener("click", startHuggingFaceDownload);
  updateHuggingFaceSelection();
}

async function startHuggingFaceDownload() {
  const files = Array.from(state.hfSelectedFiles);
  if (!files.length) {
    $("#hf-status").textContent = "Select at least one GGUF file.";
    return;
  }
  const button = $("#btn-hf-download");
  button.disabled = true;
  try {
    const job = await api("/api/huggingface/downloads", {
      method: "POST",
      body: JSON.stringify({ repo_id: state.hfRepo.id, files }),
    });
    watchHuggingFaceDownload(job.id);
  } catch (err) {
    button.disabled = false;
    $("#hf-status").textContent = err.message;
  }
}

async function watchHuggingFaceDownload(jobId) {
  if (state.hfJobTimer) clearTimeout(state.hfJobTimer);
  try {
    const job = await api(`/api/huggingface/downloads/${encodeURIComponent(jobId)}`);
    const progress = $("#hf-download-progress");
    const pct = job.total_bytes
      ? Math.min(100, (100 * job.downloaded_bytes) / job.total_bytes)
      : 0;
    progress.classList.remove("hidden");
    const message = job.status === "complete"
      ? `Complete: ${job.path}`
      : job.status === "failed"
        ? `Download failed: ${job.error}`
        : `${pct.toFixed(1)}% complete`;
    const detail = job.status === "downloading"
      ? `file ${job.file_index}/${job.file_count} · ${job.file || "starting"} · destination ~/models/HuggingFace/${job.repo_id}`
      : job.status;
    progress.innerHTML = `<strong>${escapeHtml(message)}</strong><div class="hf-progress-track"><div class="hf-progress-bar" style="width:${pct.toFixed(1)}%"></div></div><div class="hf-progress-meta">${escapeHtml(detail)}</div>`;
    if (job.status === "complete") {
      $("#hf-status").textContent = "Download complete. The local model scan is running.";
      $("#btn-hf-download").disabled = false;
      await refreshModels();
      return;
    }
    if (job.status === "failed") {
      $("#hf-status").textContent = job.error;
      $("#btn-hf-download").disabled = false;
      return;
    }
    state.hfJobTimer = setTimeout(() => watchHuggingFaceDownload(jobId), 1000);
  } catch (err) {
    $("#hf-status").textContent = err.message;
    $("#btn-hf-download").disabled = false;
  }
}

function renderModels() {
  const list = $("#model-list");
  const q = ($("#model-search").value || "").toLowerCase();
  const filtered = state.models.filter(
    (m) =>
      m.name.toLowerCase().includes(q) ||
      m.folder.toLowerCase().includes(q) ||
      (m.alias || "").toLowerCase().includes(q) ||
      (m.path || "").toLowerCase().includes(q)
  );

  if (!filtered.length) {
    list.innerHTML = `<li class="muted" style="padding:1rem">${
      state.models.length ? "No matches" : "No models found — try Refresh"
    }</li>`;
    return;
  }

  list.innerHTML = filtered
    .map(
      (m) => {
        const badges = [
          `<span class="badge">${escapeHtml(m.format || "gguf")}</span>`,
          m.shards > 1 ? `<span class="badge">${m.shards} shards</span>` : "",
        ].join("");
        return `
    <li>
      <div class="model-item tip ${
        state.selectedModel?.path === m.path ? "selected" : ""
      }" data-path="${escapeHtml(m.path)}" data-tip="${escapeHtml(
          `${m.format || "gguf"} · ${formatSize(m.size_gb)} · ${m.path}`
        )}" role="button" tabindex="0">
        <span class="model-name-line">
          <span class="model-name">${badges}${escapeHtml(m.name)}</span>
          <button type="button" class="btn-model-copy tip" data-name="${escapeHtml(
            m.name
          )}" data-tip="Copy model name" aria-label="Copy model name">⎘</button>
        </span>
        <span class="model-meta">${formatSize(m.size_gb)} · ${escapeHtml(
          m.folder
        )}</span>
      </div>
    </li>`;
      }
    )
    .join("");

  list.querySelectorAll(".model-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".btn-model-copy")) return;
      selectModel(el.dataset.path);
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectModel(el.dataset.path);
      }
    });
  });
  list.querySelectorAll(".btn-model-copy").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const name = btn.dataset.name || "";
      if (!name) return;
      try {
        await copyText(name);
        const prev = btn.textContent;
        btn.textContent = "✓";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = prev;
          btn.classList.remove("copied");
        }, 1000);
      } catch (err) {
        alert(err.message || "Copy failed");
      }
    });
  });
}

function modelLooksMtp(m) {
  if (!m) return false;
  if (m.mtp_capable) return true;
  const hay = `${m.name || ""} ${m.path || ""}`.toLowerCase();
  return hay.includes("mtp");
}

function syncMtpControls() {
  const on = Boolean($("#mtp-check")?.checked);
  const draft = $("#mtp-draft-n");
  if (draft) draft.disabled = !on;
  updateGpuWarning();
}

function syncVisionControls() {
  const el = $("#vision-mmproj");
  if (!el) return;
  const m = state.selectedModel;
  if (m?.has_mmproj && m.mmproj_name) {
    el.textContent = m.mmproj_name;
    el.classList.remove("muted");
  } else {
    el.textContent = "// none";
    el.classList.add("muted");
  }
}

function fmtGb(n) {
  if (n == null || Number.isNaN(n)) return "?";
  const v = Number(n);
  return (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)) + " GB";
}

function analyzerTps(n) {
  const value = Number(n);
  return Number.isFinite(value) && value > 0 && value <= 1000 ? value : null;
}

function renderVramEstimate(est, extraMsgs) {
  const box = $("#vram-estimate");
  if (!box) return;
  const verdict = est?.verdict || "ok";
  box.classList.remove(
    "hidden",
    "verdict-ok",
    "verdict-tight",
    "verdict-oom",
    "verdict-oom_vram"
  );
  box.classList.add(`verdict-${verdict}`);

  const label =
    verdict === "ok" ? "fits" : verdict === "tight" ? "tight" : "oom";
  $("#vram-est-verdict").textContent = label;

  const mtpBit = est.mtp ? ` · mtp+${fmtGb(est.mtp_gb)}` : "";
  const poolBit =
    est.devices && est.devices.length > 1
      ? `pool ${fmtGb(est.pool_gb)} free`
      : `GPU ${est.gpu} ${fmtGb(est.free_gb)} free`;
  const ramBit = est.uses_ram ? ` · RAM ~${fmtGb(est.estimated_ram_gb)}` : "";
  $("#vram-est-summary").textContent = `need ${fmtGb(est.need_gb)} · ${poolBit}${ramBit}${mtpBit}`;

  const parts = [
    `weights ${fmtGb(est.weights_gb)}`,
    `KV ${fmtGb(est.kv_gb)}`,
    `compute ${fmtGb(est.scratch_gb)}`,
  ];
  if (est.mtp && est.mtp_gb > 0) parts.push(`MTP ${fmtGb(est.mtp_gb)}`);
  if (est.confidence === "low") parts.push("low-confidence arch");
  if (est.arch?.architecture) parts.push(String(est.arch.architecture));
  $("#vram-est-breakdown").textContent = parts.join(" · ");

  const gpusEl = $("#vram-est-gpus");
  const rows = est.per_gpu || [];
  if (rows.length) {
    gpusEl.innerHTML = rows
      .map((g) => {
        const usedPct = Math.max(
          0,
          Math.min(100, g.free_gb > 0 ? (g.need_gb / g.free_gb) * 100 : 100)
        );
        const fallback = !g.fits && est.uses_ram;
        const cls =
          !g.fits && !fallback
            ? "bad"
            : fallback || g.headroom_gb < 2
              ? "warn"
              : "";
        const mark = fallback
          ? "RAM fallback"
          : !g.fits
            ? "OOM"
            : g.headroom_gb < 2
              ? "tight"
              : "ok";
        return `<div class="vram-est-gpu ${cls}" title="tensor-split share ${g.share_pct}%">
          <span>GPU ${g.gpu}</span>
          <div class="bar"><span style="width:${usedPct.toFixed(1)}%"></span></div>
          <span>${fmtGb(g.need_gb)} / ${fmtGb(g.free_gb)} free · ${mark}</span>
        </div>`;
      })
      .join("");
  } else {
    gpusEl.innerHTML = "";
  }

  const tipEl = $("#vram-est-tip");
  const tips = [];
  if (verdict === "oom") {
    tips.push("Likely out of VRAM at these settings");
  } else if (verdict === "tight") {
    tips.push("Low headroom — may still OOM under load");
  }
  if (extraMsgs?.length) tips.push(...extraMsgs);
  if (est.tip) tips.push(est.tip);
  if (est.mtp_capable && est.mtp) {
    tips.push(`draft_n=${est.mtp_draft_n} (--spec-type draft-mtp)`);
  }
  if (tips.length) {
    tipEl.textContent = tips.join(" · ");
    tipEl.classList.remove("hidden");
  } else {
    tipEl.textContent = "";
    tipEl.classList.add("hidden");
  }
}

function selectModel(path) {
  state.selectedModel = state.models.find((m) => m.path === path) || null;
  const el = $("#selected-model");
  const launchHint = $("#launch-hint");
  if (state.selectedModel) {
    let tag = "";
    if (state.selectedModel.format === "hf") {
      tag = "[hf→gguf] ";
    } else if (state.selectedModel.shards > 1) {
      tag = `[${state.selectedModel.shards} shards] `;
    }
    el.textContent = tag + state.selectedModel.name;
    el.classList.remove("muted");
    $("#btn-start").disabled = false;
    if (launchHint) launchHint.classList.add("hidden");
    const mtpCheck = $("#mtp-check");
    if (mtpCheck) {
      mtpCheck.checked = modelLooksMtp(state.selectedModel);
      const draft = $("#mtp-draft-n");
      if (draft) {
        draft.value = String(state.selectedModel.recommended_mtp_draft_n || 2);
      }
      syncMtpControls();
    }
    const visionCheck = $("#vision-check");
    if (visionCheck) {
      visionCheck.checked = Boolean(state.selectedModel.has_mmproj);
      syncVisionControls();
    }
  } else {
    el.textContent = "Choose a model from the list";
    el.classList.add("muted");
    $("#btn-start").disabled = true;
    if (launchHint) launchHint.classList.remove("hidden");
    syncVisionControls();
  }
  renderModels();
  updateGpuWarning();
}

const SPILL_OPTIONS = [
  {
    value: "none",
    label: "Keep on GPU",
    hint: "Fastest option — run entirely on the selected graphics card.",
  },
  {
    value: "ram",
    label: "Use system RAM if needed",
    hint: "Lets the model spill into memory when VRAM is full. Slower, but fits larger models.",
  },
  {
    value: "gpu",
    label: "Spread across other GPUs",
    hint: "Split the model across your other graphics cards when one card is not enough.",
    multiGpu: true,
  },
  {
    value: "both",
    label: "Other GPUs, then RAM",
    hint: "Use every graphics card first, then system RAM for anything left over.",
    multiGpu: true,
  },
];

function normalizeSpillValue(value) {
  const spill = value || "none";
  if (state.gpus.length < 2) {
    if (spill === "both") return "ram";
    if (spill === "gpu") return "none";
  }
  return spill;
}

function formatGpuSummary(g) {
  const free =
    g.memory_free_mib != null
      ? ` · ${Math.round(g.memory_free_mib / 1024)} GB free`
      : "";
  return `${g.name} (${Math.round(g.memory_total_mib / 1024)} GB${free})`;
}

function updateSpillHint() {
  const sel = $("#spill-select");
  const hint = $("#spill-hint");
  if (!sel || !hint) return;
  const multi = state.gpus.length >= 2;
  const selected = SPILL_OPTIONS.filter((option) => !option.multiGpu || multi).find(
    (option) => option.value === sel.value
  );
  hint.textContent = selected?.hint || "";
}

function renderSpillOptions(preferred) {
  const sel = $("#spill-select");
  if (!sel) return;
  const multi = state.gpus.length >= 2;
  const options = SPILL_OPTIONS.filter((option) => !option.multiGpu || multi);
  const current = normalizeSpillValue(preferred ?? sel.value);
  sel.innerHTML = options
    .map(
      (option) =>
        `<option value="${option.value}">${escapeHtml(option.label)}</option>`
    )
    .join("");
  sel.value = options.some((option) => option.value === current)
    ? current
    : "none";
  updateSpillHint();
}

function renderGpus() {
  const sel = $("#gpu-select");
  const field = $("#gpu-field");
  const singleWrap = $("#gpu-single-wrap");
  const single = $("#gpu-single");
  const deviceRow = $("#launch-device-row");
  if (!sel) return;

  if (!state.gpus.length) {
    field?.classList.remove("hidden");
    singleWrap?.classList.add("hidden");
    deviceRow?.classList.remove("launch-device-row--single");
    sel.innerHTML = '<option value="0">No GPU detected</option>';
    renderSpillOptions();
    return;
  }

  sel.innerHTML = state.gpus
    .map(
      (g) =>
        `<option value="${g.index}">GPU ${g.index}: ${escapeHtml(formatGpuSummary(g))}</option>`
    )
    .join("");

  if (state.gpus.length === 1) {
    const g = state.gpus[0];
    field?.classList.add("hidden");
    singleWrap?.classList.remove("hidden");
    deviceRow?.classList.add("launch-device-row--single");
    if (single) single.textContent = formatGpuSummary(g);
    sel.value = String(g.index);
  } else {
    field?.classList.remove("hidden");
    singleWrap?.classList.add("hidden");
    deviceRow?.classList.remove("launch-device-row--single");
  }

  renderSpillOptions();
}

function updateGpuWarning() {
  const gpu = parseInt($("#gpu-select").value, 10);
  const spill = ($("#spill-select") && $("#spill-select").value) || "none";
  const g = state.gpus.find((x) => x.index === gpu);
  const onGpu = state.servers.filter(
    (s) =>
      (s.gpu === gpu || String(s.devices || "").split(",").includes(String(gpu))) &&
      ["running", "starting", "converting"].includes(s.status)
  );
  const box = $("#vram-estimate");
  const msgs = [];
  if (onGpu.length) {
    msgs.push(`GPU ${gpu} already has ${onGpu.length} server(s) running`);
  }

  const nglInput = $("#ngl-input");
  if (nglInput) nglInput.disabled = spill !== "none";

  const reqId = (state._vramReqId = (state._vramReqId || 0) + 1);
  if (g && state.selectedModel) {
    const ctx = typeof getCtxValue === "function" ? getCtxValue() : 8192;
    const mtp = Boolean($("#mtp-check")?.checked);
    const mtpDraftN = Math.max(
      1,
      Math.min(6, parseInt($("#mtp-draft-n")?.value || "2", 10) || 2)
    );
    api("/api/vram-estimate", {
      method: "POST",
      body: JSON.stringify({
        model: state.selectedModel.path,
        ctx,
        spill,
        gpu,
        mtp,
        mtp_draft_n: mtpDraftN,
      }),
    })
      .then((est) => {
        if (reqId !== state._vramReqId) return;
        renderVramEstimate(est, msgs);
      })
      .catch(() => {
        if (reqId !== state._vramReqId) return;
        if (box) {
          if (msgs.length) {
            box.classList.remove("hidden");
            box.classList.add("verdict-tight");
            $("#vram-est-verdict").textContent = "warn";
            $("#vram-est-summary").textContent = msgs.join(" · ");
            $("#vram-est-breakdown").textContent = "";
            $("#vram-est-gpus").innerHTML = "";
            $("#vram-est-tip").classList.add("hidden");
          } else {
            box.classList.add("hidden");
          }
        }
      });
    return;
  }

  if (box) box.classList.add("hidden");
}

function renderServers() {
  const container = $("#server-list");
  const active = state.servers.filter((s) => s.status !== "stopped");

  if (!active.length) {
    container.innerHTML = '<p class="muted empty-hint">No servers running</p>';
    updateChatServerSelect();
    const unloadBtn = $("#btn-unload-all");
    if (unloadBtn) unloadBtn.disabled = true;
    renderFavorites();
    return;
  }

  container.innerHTML = active
    .map(
      (s) => {
        const cmd =
          s.codex_cmd ||
          `codex --profile lemur -m ${s.alias || s.model_name}`;
        const saved = state.favorites.some((favorite) =>
          favoriteMatchesServer(favorite, s)
        );
        const canFavorite = ["running", "starting", "converting"].includes(
          s.status
        );
        return `
    <div class="server-card status-${s.status}">
      <div class="server-card-head">
        <span class="server-card-name tip" data-tip="Model name for this llama-server instance">${escapeHtml(s.model_name)}</span>
        <span class="server-status ${s.status} tip" data-tip="Process state: starting (loading), running (ready), or error">${s.status}</span>
      </div>
      <div class="server-meta tip" data-tip="Primary GPU, CUDA device list, spill mode, MTP, vision, listen port, and context size (-c)">GPU ${s.gpu}${s.devices && s.devices !== String(s.gpu) ? `→[${escapeHtml(s.devices)}]` : ""} · spill ${escapeHtml(s.spill || "none")}${s.mtp ? ` · mtp n=${s.mtp_draft_n || 2}` : ""}${s.vision ? " · vision" : ""} · :${s.port} · ctx ${s.ctx}</div>
      <div class="server-actions">
        ${canFavorite ? `<button type="button" class="btn btn-ghost btn-favorite-server tip ${saved ? "is-saved" : ""}" data-id="${s.id}" data-tip="Save this model and its current launch settings as a one-click favorite" ${saved ? "disabled" : ""}>${saved ? "★ saved" : "☆ favorite"}</button>` : ""}
        <button type="button" class="btn btn-ghost btn-chat tip" data-id="${s.id}" data-tip="Open the quick chat playground against this server">chat</button>
        <button type="button" class="btn btn-ghost btn-analyzer tip" data-id="${s.id}" data-tip="Open process analyzer for this server">analyzer</button>
        <button type="button" class="btn btn-ghost btn-copy-codex tip" data-cmd="${escapeHtml(cmd)}" data-tip="Copy the Codex CLI command for this model alias">copy codex</button>
        <button type="button" class="btn btn-ghost btn-logs tip" data-id="${s.id}" data-tip="Show recent llama-server log lines">logs</button>
        <button type="button" class="btn btn-danger btn-stop tip" data-id="${s.id}" data-tip="Stop this server and free its GPU memory">stop</button>
      </div>
    </div>`;
      }
    )
    .join("");

  container.querySelectorAll(".btn-stop").forEach((btn) => {
    btn.addEventListener("click", () => stopServer(btn.dataset.id));
  });
  container.querySelectorAll(".btn-favorite-server").forEach((btn) => {
    btn.addEventListener("click", () => saveFavorite(btn.dataset.id, btn));
  });
  container.querySelectorAll(".btn-logs").forEach((btn) => {
    btn.addEventListener("click", () => showLogs(btn.dataset.id));
  });
  container.querySelectorAll(".btn-chat").forEach((btn) => {
    btn.addEventListener("click", () => openChat(btn.dataset.id));
  });
  container.querySelectorAll(".btn-analyzer").forEach((btn) => {
    btn.addEventListener("click", () => openAnalyzer(btn.dataset.id));
  });
  container.querySelectorAll(".btn-copy-codex").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cmd = btn.dataset.cmd || "";
      try {
        await copyText(cmd);
        const prev = btn.textContent;
        btn.textContent = "copied";
        setTimeout(() => {
          btn.textContent = prev;
        }, 1200);
      } catch (err) {
        alert(err.message || "Copy failed");
      }
    });
  });

  updateChatServerSelect();
  updateGpuWarning();
  const unloadBtn = $("#btn-unload-all");
  if (unloadBtn) unloadBtn.disabled = !active.length;
  renderFavorites();
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(ta);
  if (!ok) throw new Error("Clipboard unavailable");
}

function updateChatServerSelect() {
  const sel = $("#chat-server-select");
  const runnable = state.servers.filter((s) =>
    ["running", "starting"].includes(s.status)
  );
  sel.innerHTML = runnable
    .map(
      (s) =>
        `<option value="${s.id}">${escapeHtml(s.model_name)} (:${s.port})</option>`
    )
    .join("");
  if (state.chatServerId && runnable.some((s) => s.id === state.chatServerId)) {
    sel.value = state.chatServerId;
  } else if (runnable.length) {
    state.chatServerId = runnable[0].id;
    sel.value = state.chatServerId;
  }
  updateChatSubtitle();
}

function fmtToks(n) {
  if (n == null || Number.isNaN(n)) return null;
  const v = Number(n);
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}

function chatMsgMeta(m) {
  if (m.role !== "assistant") return "";
  const parts = [];
  if (m.tps != null) parts.push(`${fmtToks(m.tps)} tok/s`);
  else if (m.streaming) parts.push("…");
  if (m.promptTps != null) parts.push(`prompt ${fmtToks(m.promptTps)}`);
  if (m.tokens != null) parts.push(`${m.tokens} tok`);
  if (!parts.length) return "";
  return `<span class="chat-msg-meta">${escapeHtml(parts.join(" · "))}</span>`;
}

function renderChat() {
  const box = $("#chat-messages");
  const empty = $("#chat-empty");
  if (!state.chatMessages.length) {
    box.innerHTML = `
      <div id="chat-empty" class="chat-empty">
        <p class="accent">›_</p>
        <p>Send a message to test the running model.</p>
        <p class="muted">This is a quick playground — codex is for real work.</p>
      </div>`;
    return;
  }
  if (empty) empty.remove();
  box.innerHTML = state.chatMessages
    .map(
      (m) =>
        `<div class="chat-msg ${m.role}"><span class="chat-msg-role">${
          m.role === "user" ? "you" : "model"
        }</span>${escapeHtml(m.content)}${chatMsgMeta(m)}</div>`
    )
    .join("");
  box.scrollTop = box.scrollHeight;
}

function updateChatSubtitle() {
  const sub = $("#chat-subtitle");
  if (!sub) return;
  const s = state.servers.find((x) => x.id === state.chatServerId);
  sub.textContent = s
    ? `${s.alias || s.model_name} · :${s.port} · GPU ${s.gpu}`
    : "pick a running server";
}

function openChat(serverId) {
  state.chatServerId = serverId;
  state.chatMessages = [];
  const overlay = $("#chat-overlay");
  overlay.classList.remove("hidden");
  overlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("chat-open");
  updateChatServerSelect();
  updateChatSubtitle();
  renderChat();
  const input = $("#chat-input");
  input.focus();
}

function closeChat() {
  const overlay = $("#chat-overlay");
  overlay.classList.add("hidden");
  overlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("chat-open");
}

async function loadModels() {
  const status = $("#scan-status");
  status.classList.remove("hidden");
  try {
    const data = await api("/api/models");
    state.models = data.models;
    if (data.scanning) {
      setTimeout(loadModels, 2000);
    } else {
      status.classList.add("hidden");
    }
    renderModels();
  } catch (e) {
    status.textContent = `Scan failed: ${e.message}`;
  }
}

async function refreshModels() {
  $("#scan-status").classList.remove("hidden");
  $("#scan-status").textContent = "Scanning home directory…";
  try {
    const data = await api("/api/models/refresh", { method: "POST" });
    state.models = data.models;
    $("#scan-status").classList.add("hidden");
    renderModels();
  } catch (e) {
    $("#scan-status").textContent = `Scan failed: ${e.message}`;
  }
}

async function loadGpus() {
  const data = await api("/api/gpus");
  state.gpus = data.gpus;
  renderGpus();
  updateGpuWarning();
}

async function loadServers() {
  const data = await api("/api/servers");
  state.servers = data.servers;
  renderServers();
}

async function loadFavorites() {
  const prevCount = state.favorites.length;
  const data = await api("/api/favorites");
  state.favorites = data.favorites || [];
  applyModelsPanelDefault(prevCount);
  renderFavorites();
  renderServers();
}

async function saveFavorite(serverId, button) {
  if (!serverId) return;
  if (button) button.disabled = true;
  try {
    await api("/api/favorites", {
      method: "POST",
      body: JSON.stringify({ server_id: serverId }),
    });
    await loadFavorites();
  } catch (err) {
    if (button) button.disabled = false;
    alert(err.message);
  }
}

async function deleteFavorite(favoriteId) {
  if (!favoriteId) return;
  try {
    await api(`/api/favorites/${encodeURIComponent(favoriteId)}`, {
      method: "DELETE",
    });
    await loadFavorites();
  } catch (err) {
    alert(err.message);
  }
}

async function startFavorite(favoriteId, button) {
  if (!favoriteId) return;
  if (button) {
    button.disabled = true;
    button.textContent = "starting…";
  }
  try {
    await api(`/api/favorites/${encodeURIComponent(favoriteId)}/start`, {
      method: "POST",
    });
    await loadServers();
  } catch (err) {
    if (button) {
      button.disabled = false;
      button.textContent = "start()";
    }
    alert(err.message);
  }
}

function loadFavoriteIntoForm(favoriteId) {
  const favorite = state.favorites.find((item) => item.id === favoriteId);
  if (!favorite) return;
  const model = state.models.find((item) => item.path === favorite.model_path);
  if (!model) {
    alert("This favorite's model is not in the current model scan.");
    return;
  }

  selectModel(model.path);
  renderGpus();
  $("#gpu-select").value = String(favorite.gpu);
  renderSpillOptions(normalizeSpillValue(favorite.spill));
  setCtxPreset(Number(favorite.ctx));
  $("#ngl-input").value = String(favorite.ngl);
  $("#mtp-check").checked = Boolean(favorite.mtp);
  $("#mtp-draft-n").value = String(favorite.mtp_draft_n || 2);
  $("#vision-check").checked = Boolean(favorite.vision);
  $("#port-input").value = "";
  syncMtpControls();
  syncVisionControls();
  updateGpuWarning();
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  renderLanAccess();
  setCtxPreset(state.settings.default_ctx);
  $("#ngl-input").value = state.settings.default_ngl;
  $("#settings-binary").value = state.settings.llama_server_path;
  $("#settings-scan-root").value = state.settings.scan_root;
  $("#settings-min-size").value = state.settings.min_model_size_mb;
  $("#settings-default-ctx").value = state.settings.default_ctx;
  const fontSize = clampFontSize(state.settings.ui_font_size);
  const fontInput = $("#settings-font-size");
  if (fontInput) fontInput.value = String(fontSize);
  const settingsSplash = $("#settings-show-splash");
  if (settingsSplash) {
    settingsSplash.checked = state.settings.show_splash_on_startup !== false;
  }
  const welcomeSplash = $("#welcome-show-startup");
  if (welcomeSplash) {
    welcomeSplash.checked = state.settings.show_splash_on_startup !== false;
  }
  updateFontSizeLabel(fontSize);
  applyFontSize(fontSize);
  const bits = [
    state.settings.binary_exists ? "llama-server ok" : "llama-server missing",
  ];
  $("#settings-binary-status").textContent = bits.join(" · ");
}

function renderLanAccess() {
  const btn = $("#btn-lan-access");
  if (!btn) return;
  const enabled = state.settings.default_host === "0.0.0.0";
  btn.textContent = enabled ? "lan:on" : "lan:off";
  btn.setAttribute("aria-pressed", String(enabled));
  btn.classList.toggle("lan-enabled", enabled);
}

async function toggleLanAccess() {
  const btn = $("#btn-lan-access");
  const enabled = state.settings.default_host === "0.0.0.0";
  if (
    !enabled &&
    !confirm(
      "Enable unauthenticated access to loaded models from your local network? Only do this on a trusted network."
    )
  ) {
    return;
  }

  btn.disabled = true;
  try {
    await api("/api/network-access", {
      method: "PUT",
      body: JSON.stringify({ lan_enabled: !enabled }),
    });
    state.settings.default_host = enabled ? "127.0.0.1" : "0.0.0.0";
    renderLanAccess();
    await loadServers();
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
  }
}

function clampFontSize(n) {
  const v = parseInt(n, 10);
  if (!Number.isFinite(v)) return 15;
  return Math.min(20, Math.max(12, v));
}

function updateFontSizeLabel(n) {
  const el = $("#settings-font-size-val");
  if (el) el.textContent = `${clampFontSize(n)}px`;
}

function applyFontSize(n) {
  document.documentElement.style.setProperty(
    "--font-size",
    `${clampFontSize(n)}px`
  );
}

const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const CTX_MIN = 512;
const CTX_MAX = 1048576;
const CTX_STEP = 512;

function formatCtx(n) {
  n = Math.round(n);
  if (n >= 1048576) return `${(n / 1048576).toFixed(n % 1048576 === 0 ? 0 : 2)}M`;
  if (n >= 1024) {
    const k = n / 1024;
    return Number.isInteger(k) ? `${k}k` : `${k.toFixed(1)}k`;
  }
  return String(n);
}

function snapCtx(n) {
  n = Math.min(CTX_MAX, Math.max(CTX_MIN, Math.round(n) || 8192));
  if (n === CTX_MAX || n === CTX_MIN) return n;
  return Math.round(n / CTX_STEP) * CTX_STEP;
}

function syncCtxSliderLabel() {
  const custom = $("#ctx-custom");
  const label = $("#ctx-custom-val");
  if (!custom || !label) return;
  label.textContent = formatCtx(parseInt(custom.value, 10) || 8192);
}

function setCtxPreset(value) {
  const n = parseInt(value, 10);
  const sel = $("#ctx-select");
  const wrap = $("#ctx-custom-wrap");
  const custom = $("#ctx-custom");
  if (!sel) return;
  if (CTX_PRESETS.includes(n)) {
    sel.value = String(n);
    wrap.classList.add("hidden");
  } else {
    sel.value = "custom";
    wrap.classList.remove("hidden");
    custom.value = String(snapCtx(n));
    syncCtxSliderLabel();
  }
}

function getCtxValue() {
  const sel = $("#ctx-select");
  if (sel.value === "custom") {
    return snapCtx(parseInt($("#ctx-custom").value, 10));
  }
  return parseInt(sel.value, 10);
}

function onCtxSelectChange() {
  const wrap = $("#ctx-custom-wrap");
  if ($("#ctx-select").value === "custom") {
    wrap.classList.remove("hidden");
    syncCtxSliderLabel();
    $("#ctx-custom").focus();
  } else {
    wrap.classList.add("hidden");
  }
  updateGpuWarning();
}

function onCtxSliderInput() {
  syncCtxSliderLabel();
  updateGpuWarning();
}

function stepNumber(inputId, dir) {
  const el = document.getElementById(inputId);
  if (!el) return;
  const step = parseFloat(el.step) || 1;
  const min = el.min !== "" ? parseFloat(el.min) : -Infinity;
  const max = el.max !== "" ? parseFloat(el.max) : Infinity;
  let val = parseFloat(el.value);
  if (!Number.isFinite(val)) val = min !== -Infinity ? min : 0;
  val = Math.min(max, Math.max(min, val + dir * step));
  el.value = Number.isInteger(step) ? String(Math.round(val)) : String(val);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

async function startServer(e) {
  e.preventDefault();
  if (!state.selectedModel) return;

  const portVal = $("#port-input").value;
  const body = {
    model: state.selectedModel.path,
    gpu: parseInt($("#gpu-select").value, 10),
    ctx: getCtxValue(),
    ngl: parseInt($("#ngl-input").value, 10),
    spill: ($("#spill-select") && $("#spill-select").value) || "none",
    mtp: Boolean($("#mtp-check")?.checked),
    mtp_draft_n: Math.max(1, Math.min(6, parseInt($("#mtp-draft-n")?.value || "2", 10) || 2)),
    vision: Boolean($("#vision-check")?.checked),
    sync_codex: true,
    set_codex_default: true,
  };
  if (portVal) body.port = parseInt(portVal, 10);

  $("#btn-start").disabled = true;
  try {
    await api("/api/servers", { method: "POST", body: JSON.stringify(body) });
    await loadServers();
  } catch (err) {
    alert(err.message);
  } finally {
    $("#btn-start").disabled = !state.selectedModel;
  }
}

async function stopServer(id) {
  try {
    await api(`/api/servers/${id}`, { method: "DELETE" });
    await loadServers();
  } catch (err) {
    alert(err.message);
  }
}

async function unloadAll() {
  const active = state.servers.filter((s) => s.status !== "stopped");
  if (!active.length) return;
  try {
    await api("/api/servers", { method: "DELETE" });
    await loadServers();
  } catch (err) {
    alert(err.message);
  }
}

function fmtNum(n, digits = 0) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function fmtMib(n) {
  if (n == null) return "—";
  if (n >= 1024) return `${(n / 1024).toFixed(1)}G`;
  return `${Math.round(n)}M`;
}

function barClass(pct) {
  if (pct >= 90) return "fill hot";
  if (pct >= 75) return "fill warn";
  return "fill";
}

function openAnalyzer(serverId) {
  const runnable = state.servers.filter((s) =>
    ["running", "starting"].includes(s.status)
  );
  if (state.activeTab !== "analyzer") state.previousTab = state.activeTab;
  state.activeTab = "analyzer";
  state.analyzerServerId =
    runnable.length && serverId && runnable.some((s) => s.id === serverId)
      ? serverId
      : runnable[0]?.id || null;
  const sel = $("#az-server-select");
  sel.innerHTML = runnable.length
    ? runnable
        .map(
          (s) =>
            `<option value="${s.id}">${escapeHtml(s.alias || s.model_name)} · :${s.port}</option>`
        )
        .join("")
    : '<option value="">No running models</option>';
  sel.disabled = !runnable.length;
  sel.value = state.analyzerServerId || "";
  sel.dataset.ids = runnable.map((s) => s.id).join(",");
  $("#analyzer-overlay").classList.remove("hidden");
  $("#analyzer-overlay").setAttribute("aria-hidden", "false");
  document.body.classList.add("analyzer-open");
  setTabSelection("analyzer");
  _bindAzWindowSliders();
  state.azForceChart = true;
  refreshAnalyzer();
  if (state.analyzerTimer) clearInterval(state.analyzerTimer);
  state.analyzerTimer = setInterval(refreshAnalyzer, 1000);
}

function closeAnalyzer() {
  showMainTab(state.previousTab);
}

async function refreshAnalyzer() {
  if ($("#analyzer-overlay").classList.contains("hidden")) return;
  if (state.analyzerBusy) return;
  state.analyzerBusy = true;
  try {
    const id = state.analyzerServerId || "";
    const data = await api(
      `/api/analyzer${id ? `?server_id=${encodeURIComponent(id)}` : ""}`
    );
    renderAnalyzer(data);
  } catch (err) {
    $("#az-subtitle").textContent = err.message || "analyzer error";
  } finally {
    state.analyzerBusy = false;
  }
}

function renderAnalyzer(data) {
  const active = data.active;
  if (!active) {
    $("#az-subtitle").textContent = data.error || "no running servers";
    $("#az-kpis").innerHTML = "";
    $("#az-devices-body").innerHTML = `<p class="az-empty">${escapeHtml(data.error || "none")}</p>`;
    $("#az-resources-body").innerHTML = "";
    $("#az-timeline-body").innerHTML = "";
    return;
  }
  const s = active.server || {};
  const k = active.kpis || {};
  const p = active.props || {};
  $("#az-subtitle").textContent = `${p.model_alias || s.alias || "model"} · ${p.model_ftype || ""} · spill ${s.spill || "none"} · devices ${active.devices_claimed || s.gpu}`;
  if (data.servers) {
    const sel = $("#az-server-select");
    const ids = data.servers.map((x) => x.id).join(",");
    if (sel.dataset.ids !== ids) {
      sel.dataset.ids = ids;
      const cur = sel.value;
      sel.innerHTML = data.servers
        .map(
          (x) =>
            `<option value="${x.id}">${escapeHtml(x.alias || x.model_name)} · :${x.port}</option>`
        )
        .join("");
      sel.value = s.id || cur;
    }
    state.analyzerServerId = sel.value || s.id;
  }

  const hist = (active.history || []).map((h) => ({
    ...h,
    gen_tps: analyzerTps(h.gen_tps),
    prompt_tps: analyzerTps(h.prompt_tps),
  }));
  state.azHist = hist;
  const latest = hist.length ? hist[hist.length - 1] : null;
  const liveGen =
    k.busy && latest?.gen_tps != null
      ? latest.gen_tps
      : analyzerTps(k.gen_tps_avg);
  const livePrompt =
    k.busy && latest?.prompt_tps != null
      ? latest.prompt_tps
      : analyzerTps(k.prompt_tps_avg);

  let genTip =
    "Token generation speed — live while busy, else recent average. Matches llama-server predicted_per_second (accepted tokens / gen time; MTP counts accepted drafts).";
  if (k.mtp && k.draft_n) {
    const acc =
      k.draft_n_accepted != null
        ? ` · draft accept ${fmtNum(k.draft_n_accepted)}/${fmtNum(k.draft_n)}`
        : "";
    genTip += acc;
  }

  const kpis = [
    { k: "ctx fill", v: `${fmtNum(k.ctx_pct, 0)}%`, u: `${fmtNum(k.ctx_used)}/${fmtNum(k.n_ctx)}`, tip: "Context tokens in use vs n_ctx. High % means the window is nearly full." },
    { k: "prompt t/s", v: livePrompt != null ? fmtNum(livePrompt, 0) : "—", u: k.busy ? "live" : "avg", tip: "Prompt-eval (prefill) speed — live while busy, else recent average." },
    { k: "gen t/s", v: liveGen != null ? fmtNum(liveGen, 1) : "—", u: k.busy ? "live" : "avg", tip: genTip },
    { k: "vram", v: fmtMib(k.vram_mib), u: "process", tip: "GPU memory used by this llama-server PID (sum across GPUs)." },
    { k: "ram rss", v: fmtMib(k.rss_mib), u: "host", tip: "Host RAM resident set size for the process (CPU/RAM tensors, mmap overhead)." },
    { k: "weights", v: fmtMib(k.model_size_mib), u: "file", tip: "On-disk GGUF size — approximate weight footprint when fully offloaded." },
    { k: "gpu util", v: k.gpu_util != null ? `${fmtNum(k.gpu_util, 0)}%` : "—", u: "claimed", tip: "Average GPU utilization across claimed devices." },
    { k: "state", v: k.busy ? "busy" : "idle", u: `${k.slots_busy || 0}/${k.slots_total || 0}`, tip: "Whether a slot is actively processing, and busy/total slots." },
  ];
  $("#az-kpis").innerHTML = kpis
    .map(
      (x) =>
        `<div class="az-kpi tip" data-tip="${escapeHtml(x.tip)}"><div class="k">${x.k}</div><div class="v">${x.v} <span class="u">${x.u}</span></div></div>`
    )
    .join("");

  const gpus = active.gpus || [];
  const rssPct =
    k.rss_mib && k.model_size_mib
      ? Math.min(100, (k.rss_mib / Math.max(k.model_size_mib * 2, 1)) * 100)
      : null;
  let deviceHtml = `<p class="az-meta-line">claimed CUDA_VISIBLE_DEVICES=${escapeHtml(String(active.devices_claimed || ""))} · spill=${escapeHtml(String(active.spill || "none"))}</p>`;
  for (const g of gpus) {
    const pct = g.total_mib ? (100 * (g.used_mib || 0)) / g.total_mib : 0;
    deviceHtml += `<div class="az-bar">
      <div class="lbl">GPU${g.gpu}${g.claimed ? "*" : ""}</div>
      <div class="track"><div class="${barClass(pct)}" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="val">${fmtMib(g.used_mib)}/${fmtMib(g.total_mib)}${g.util_gpu != null ? ` · ${fmtNum(g.util_gpu, 0)}%` : ""}</div>
    </div>`;
  }
  if (k.rss_mib != null) {
    const rp = rssPct != null ? rssPct : Math.min(100, (k.rss_mib / 65536) * 100);
    deviceHtml += `<div class="az-bar">
      <div class="lbl">RAM</div>
      <div class="track"><div class="fill ram" style="width:${rp.toFixed(1)}%"></div></div>
      <div class="val">${fmtMib(k.rss_mib)} rss</div>
    </div>`;
  }
  $("#az-devices-body").innerHTML = deviceHtml;

  const now = performance.now();
  if (state.azForceChart || now - state.azChartsAt >= 250) {
    state.azChartsAt = now;
    state.azForceChart = false;
    renderAnalyzerResources(hist);
    renderAnalyzerTimeline(hist);
  }
}

function _azFmtWindow(s) {
  s = Math.max(30, Math.min(1800, Number(s) || 1800));
  if (s < 60) return `${s}s`;
  if (s % 60 === 0) return `${s / 60}m`;
  return `${(s / 60).toFixed(1)}m`;
}

function _azCompressGaps(samples, maxGapS = 1.5) {
  if (samples.length < 2) return samples;
  const out = [];
  let t = samples[0].ts;
  let prev = samples[0].ts;
  out.push({ ...samples[0], ts: t });
  for (let i = 1; i < samples.length; i++) {
    const dt = samples[i].ts - prev;
    t += dt > maxGapS ? 0.05 : Math.max(0, dt);
    out.push({ ...samples[i], ts: t });
    prev = samples[i].ts;
  }
  return out;
}

function _azWindow(hist, windowS, { activeOnly = false } = {}) {
  windowS = Math.max(30, Math.min(1800, Number(windowS) || 1800));
  if (!hist.length) return { hist: [], t0: 0, t1: 1, span: 1, empty: true };
  const tEnd = hist[hist.length - 1].ts;
  let windowed = hist.filter((h) => h.ts >= tEnd - windowS);
  if (activeOnly) {
    windowed = windowed.filter((h) => h.busy);
    let lastGen = null;
    windowed = windowed.map((h) => {
      if (h.gen_tps != null) lastGen = h.gen_tps;
      return lastGen == null ? h : { ...h, gen_tps: lastGen };
    });
    // Drop idle gaps so gen/ctx lines stay continuous across the plot.
    windowed = _azCompressGaps(windowed);
  }
  if (!windowed.length) return { hist: [], t0: 0, t1: 1, span: 1, empty: true };
  const dataT0 = windowed[0].ts;
  const dataT1 = windowed[windowed.length - 1].ts;
  const dataSpan = Math.max(dataT1 - dataT0, 0.001);
  // Always fill the plot with available (possibly compressed) samples.
  return {
    hist: _azDownsample(windowed, 900),
    t0: dataT0,
    t1: dataT1,
    span: dataSpan,
    empty: false,
  };
}

function _azDownsample(hist, maxPts) {
  if (hist.length <= maxPts) return hist;
  const out = [];
  const step = (hist.length - 1) / (maxPts - 1);
  for (let i = 0; i < maxPts; i++) out.push(hist[Math.round(i * step)]);
  return out;
}

function _azAgeLabel(ageS) {
  if (ageS <= 0.05) return "now";
  if (ageS < 60) return `-${fmtNum(ageS, 0)}s`;
  if (ageS < 3600) return `-${fmtNum(ageS / 60, ageS < 600 ? 1 : 0)}m`;
  return `-${fmtNum(ageS / 3600, 1)}h`;
}

function _azXLabels(t0, t1, span, xOf, y, fractions = [0, 0.5, 1]) {
  return fractions
    .map((f) => {
      const ts = t0 + span * f;
      const x = xOf(ts).toFixed(1);
      return `<text class="az-tick" x="${x}" y="${y}" text-anchor="middle">${_azAgeLabel(t1 - ts)}</text>`;
    })
    .join("");
}

function _azLinePaths(hist, xOf, yOf, key) {
  const paths = [];
  let seg = [];
  for (const h of hist) {
    const v = h[key];
    if (v != null && Number.isFinite(v)) {
      seg.push(`${xOf(h.ts).toFixed(1)},${yOf(v).toFixed(1)}`);
    } else if (seg.length) {
      paths.push(seg.join(" "));
      seg = [];
    }
  }
  if (seg.length) paths.push(seg.join(" "));
  return paths;
}

function _azPlotFrame({ W, H, pad, plot, yLeft, yRight, xLabels, titleL, titleR, titleLFill, titleRFill }) {
  // Outer SVG keeps text crisp; nested plot SVG stretches with preserveAspectRatio=none.
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  return `<div class="az-chart-wrap">
    <svg class="az-chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">
      <text class="az-axis-title" x="${pad.l}" y="9" fill="${titleLFill}">${titleL}</text>
      <text class="az-axis-title" x="${W - pad.r}" y="9" text-anchor="end" fill="${titleRFill}">${titleR}</text>
      ${yLeft}
      ${yRight}
      ${xLabels}
      <svg x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" viewBox="0 0 ${pw} ${ph}" preserveAspectRatio="none">
        ${plot}
      </svg>
    </svg>
  </div>`;
}

function renderAnalyzerResources(rawHist) {
  const body = $("#az-resources-body");
  if (!body) return;
  if (!rawHist.length) {
    body.innerHTML = `<p class="az-empty">// no samples yet</p>`;
    return;
  }

  const { hist, t0, t1, span, empty } = _azWindow(rawHist, state.azResWindowS);
  if (empty) {
    body.innerHTML = `<p class="az-empty">// no samples in window</p>`;
    return;
  }
  const W = 720;
  const H = 180;
  const pad = { l: 36, r: 28, t: 14, b: 16 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const xOf = (ts) => ((ts - t0) / span) * pw;
  const xOfOuter = (ts) => pad.l + xOf(ts);

  const memVals = hist.flatMap((h) =>
    [h.vram_mib, h.rss_mib].filter((v) => v != null && v > 0)
  );
  const memMax = Math.max(1024, ...(memVals.length ? memVals : [0])) * 1.1;
  const yMem = (mib) => ph - (Math.min(memMax, Math.max(0, mib || 0)) / memMax) * ph;
  const yUtil = (pct) => ph - (Math.min(100, Math.max(0, pct || 0)) / 100) * ph;
  const yMemOuter = (mib) => pad.t + yMem(mib);
  const yUtilOuter = (pct) => pad.t + yUtil(pct);

  const vramPts = _azLinePaths(hist, xOf, yMem, "vram_mib");
  const rssPts = _azLinePaths(hist, xOf, yMem, "rss_mib");
  const utilPts = _azLinePaths(hist, xOf, yUtil, "gpu_util");

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => {
      const y = (ph * (1 - f)).toFixed(1);
      return `<line class="az-grid" x1="0" y1="${y}" x2="${pw}" y2="${y}" />`;
    })
    .join("");

  const yLeft = [0, 0.5, 1]
    .map((f) => {
      const v = memMax * f;
      const y = yMemOuter(v).toFixed(1);
      return `<text class="az-tick" x="${pad.l - 3}" y="${Number(y) + 2}" text-anchor="end">${fmtMib(v)}</text>`;
    })
    .join("");
  const yRight = [0, 50, 100]
    .map((pct) => {
      const y = yUtilOuter(pct).toFixed(1);
      return `<text class="az-tick az-tick-util" x="${W - pad.r + 3}" y="${Number(y) + 2}">${pct}</text>`;
    })
    .join("");

  const plot = `${grid}
    ${vramPts.map((pts) => `<polyline class="az-line az-line-vram" points="${pts}" />`).join("")}
    ${rssPts.map((pts) => `<polyline class="az-line az-line-rss" points="${pts}" />`).join("")}
    ${utilPts.map((pts) => `<polyline class="az-line az-line-util" points="${pts}" />`).join("")}`;

  body.innerHTML = _azPlotFrame({
    W,
    H,
    pad,
    plot,
    yLeft,
    yRight,
    xLabels: _azXLabels(t0, t1, span, xOfOuter, H - 3),
    titleL: "mib",
    titleR: "util%",
    titleLFill: "#6a8ab0",
    titleRFill: "#e06a74",
  });
}

function renderAnalyzerTimeline(rawHist) {
  const body = $("#az-timeline-body");
  if (!body) return;
  if (!rawHist.length) {
    body.innerHTML = `<p class="az-empty">// waiting for samples</p>`;
    return;
  }

  const { hist, t0, t1, span, empty } = _azWindow(rawHist, state.azTlWindowS, {
    activeOnly: true,
  });
  if (empty) {
    body.innerHTML = `<p class="az-empty">// idle — no busy samples in window</p>`;
    return;
  }

  const W = 960;
  const H = 200;
  const pad = { l: 32, r: 32, t: 14, b: 16 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const genVals = hist.map((h) => h.gen_tps).filter((v) => v != null && v > 0);
  const genMax = Math.max(10, ...(genVals.length ? genVals : [0])) * 1.15;

  const xOf = (ts) => ((ts - t0) / span) * pw;
  const xOfOuter = (ts) => pad.l + xOf(ts);
  const yCtx = (pct) => ph - (Math.min(100, Math.max(0, pct || 0)) / 100) * ph;
  const yGen = (tps) => ph - (Math.min(genMax, Math.max(0, tps || 0)) / genMax) * ph;
  const yCtxOuter = (pct) => pad.t + yCtx(pct);
  const yGenOuter = (tps) => pad.t + yGen(tps);

  const ctxPts = hist.map((h) => `${xOf(h.ts).toFixed(1)},${yCtx(h.ctx_pct).toFixed(1)}`).join(" ");
  // Continuous gen line over busy-only samples (idle omitted upstream).
  const genPts = hist
    .filter((h) => h.gen_tps != null)
    .map((h) => `${xOf(h.ts).toFixed(1)},${yGen(h.gen_tps).toFixed(1)}`)
    .join(" ");

  const grid = [0, 25, 50, 75, 100]
    .map((pct) => {
      const y = yCtx(pct).toFixed(1);
      return `<line class="az-grid" x1="0" y1="${y}" x2="${pw}" y2="${y}" />`;
    })
    .join("");

  const yLeft = [0, 25, 50, 75, 100]
    .map((pct) => {
      const y = yCtxOuter(pct).toFixed(1);
      return `<text class="az-tick" x="${pad.l - 3}" y="${Number(y) + 2}" text-anchor="end">${pct}</text>`;
    })
    .join("");
  // Six labels make changes in generation speed easier to read without
  // crowding the 200 px chart.
  const yRight = [0, 0.2, 0.4, 0.6, 0.8, 1]
    .map((f) => {
      const v = genMax * f;
      const y = yGenOuter(v).toFixed(1);
      return `<text class="az-tick az-tick-gen" x="${W - pad.r + 3}" y="${Number(y) + 2}">${fmtNum(v, v >= 100 ? 0 : 1)}</text>`;
    })
    .join("");

  const plot = `${grid}
    <polyline class="az-line az-line-ctx" points="${ctxPts}" />
    ${genPts ? `<polyline class="az-line az-line-gen" points="${genPts}" />` : ""}`;

  body.innerHTML = _azPlotFrame({
    W,
    H,
    pad,
    plot,
    yLeft,
    yRight,
    xLabels: _azXLabels(t0, t1, span, xOfOuter, H - 3),
    titleL: "ctx%",
    titleR: "gen/s",
    titleLFill: "#7a9a86",
    titleRFill: "#c9b06a",
  });
}

function _bindAzWindowSliders() {
  const res = $("#az-res-win");
  const tl = $("#az-tl-win");
  const resLbl = $("#az-res-win-lbl");
  const tlLbl = $("#az-tl-win-lbl");
  if (res && !res.dataset.bound) {
    res.dataset.bound = "1";
    res.value = String(state.azResWindowS);
    resLbl.textContent = _azFmtWindow(state.azResWindowS);
    res.addEventListener("input", () => {
      state.azResWindowS = parseInt(res.value, 10);
      resLbl.textContent = _azFmtWindow(state.azResWindowS);
      state.azForceChart = true;
      renderAnalyzerResources(state.azHist);
    });
  }
  if (tl && !tl.dataset.bound) {
    tl.dataset.bound = "1";
    tl.value = String(state.azTlWindowS);
    tlLbl.textContent = _azFmtWindow(state.azTlWindowS);
    tl.addEventListener("input", () => {
      state.azTlWindowS = parseInt(tl.value, 10);
      tlLbl.textContent = _azFmtWindow(state.azTlWindowS);
      state.azForceChart = true;
      renderAnalyzerTimeline(state.azHist);
    });
  }
}

function openAnalyzerIo({ followLive = true, traceId = null, ts = null } = {}) {
  const root = document.querySelector(".analyzer");
  const pane = $("#az-io");
  if (!root || !pane) return;
  state.ioFollowLive = followLive;
  state.ioTraceId = followLive ? null : traceId;
  state.ioQueryTs = followLive ? null : ts;
  state.ioLastStamp = "";
  state.ioNeedsRefresh = false;
  root.classList.add("io-open");
  pane.classList.remove("hidden");
  pane.setAttribute("aria-hidden", "false");
  if (state.ioTimer) clearInterval(state.ioTimer);
  refreshAnalyzerIo();
  state.ioTimer = setInterval(refreshAnalyzerIo, 200);
}

function closeAnalyzerIo() {
  const root = document.querySelector(".analyzer");
  const pane = $("#az-io");
  if (root) root.classList.remove("io-open");
  if (pane) {
    pane.classList.add("hidden");
    pane.setAttribute("aria-hidden", "true");
  }
  state.ioFollowLive = true;
  state.ioTraceId = null;
  state.ioQueryTs = null;
  state.ioLastStamp = "";
  state.ioNeedsRefresh = false;
  if (state.ioTimer) {
    clearInterval(state.ioTimer);
    state.ioTimer = null;
  }
}

function ioScrollSnap(el) {
  if (!el) return { stick: true, top: 0 };
  const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
  return { stick: gap <= 48, top: el.scrollTop };
}

function ioRestoreScroll(el, snap) {
  if (!el || !snap) return;
  if (snap.stick) el.scrollTop = el.scrollHeight;
  else el.scrollTop = snap.top;
}

async function refreshAnalyzerIo() {
  if (!$("#az-io") || $("#az-io").classList.contains("hidden")) return;
  if (state.ioBusy) {
    state.ioNeedsRefresh = true;
    return;
  }
  state.ioBusy = true;
  const sid = state.analyzerServerId || "";
  const params = new URLSearchParams();
  if (sid) params.set("server_id", sid);
  // Follow-live always asks for the current in-flight/latest trace.
  if (!state.ioFollowLive && state.ioTraceId) params.set("trace_id", state.ioTraceId);
  if (!state.ioFollowLive && state.ioQueryTs != null) params.set("ts", String(state.ioQueryTs));
  const fetchGen = (state.ioFetchGen = (state.ioFetchGen || 0) + 1);
  try {
    const data = await api(`/api/analyzer/io?${params.toString()}`);
    if (fetchGen !== state.ioFetchGen) return;
    const t = data.trace;
    const sub = $("#az-io-subtitle");
    const inputEl = $("#az-io-input");
    const thinkEl = $("#az-io-thinking");
    const ansEl = $("#az-io-answer");
    if (!t) {
      sub.textContent = "no hub-proxied I/O yet — use Codex/chat via hub (:9000/v1)";
      inputEl.textContent = "// waiting for traffic through Lemur proxy";
      thinkEl.textContent = "";
      ansEl.textContent = "";
      state.ioLastStamp = "";
      return;
    }
    if (state.ioFollowLive) state.ioTraceId = t.id;
    const thinkLen = (t.thinking || "").length;
    const outLen = (t.output || "").length;
    const inLen = (t.input || "").length;
    const stamp = `${t.id}|${t.updated_at}|${thinkLen}|${outLen}|${inLen}|${t.done}`;
    const age = t.updated_at ? ((Date.now() / 1000) - t.updated_at).toFixed(1) : "?";
    const live = data.live || !t.done;
    sub.textContent = `${live ? "streaming" : "done"} · ${t.kind || "req"} · think ${thinkLen} · out ${outLen} · ${age}s ago`;

    if (stamp === state.ioLastStamp) return;
    state.ioLastStamp = stamp;

    const thinkSnap = ioScrollSnap(thinkEl);
    const ansSnap = ioScrollSnap(ansEl);
    const inputSnap = ioScrollSnap(inputEl);

    inputEl.textContent = t.input || "// empty input";
    thinkEl.textContent = t.thinking
      ? `⟦ thinking ⟧\n${t.thinking}`
      : (t.done ? "// no thinking" : "// thinking…");
    ansEl.textContent = t.output
      ? `⟦ answer ⟧\n${t.output}`
      : (t.done ? "// no answer text" : "// generating…");

    // Auto-follow only if the pane was already near the bottom.
    ioRestoreScroll(thinkEl, thinkSnap);
    ioRestoreScroll(ansEl, ansSnap);
    ioRestoreScroll(inputEl, inputSnap);
  } catch (err) {
    if (fetchGen === state.ioFetchGen) {
      $("#az-io-subtitle").textContent = err.message || "io error";
    }
  } finally {
    state.ioBusy = false;
    if (state.ioNeedsRefresh) {
      state.ioNeedsRefresh = false;
      refreshAnalyzerIo();
    }
  }
}

async function showLogs(id) {
  const data = await api(`/api/servers/${id}/logs?tail=200`);
  $("#logs-content").textContent = data.logs.join("\n") || "(no logs yet)";
  $("#logs-dialog").showModal();
}

async function sendChat(e) {
  e.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text || !state.chatServerId) return;

  state.chatMessages.push({ role: "user", content: text });
  state.chatMessages.push({
    role: "assistant",
    content: "",
    streaming: true,
  });
  renderChat();
  input.value = "";
  input.disabled = true;

  const assistantIdx = state.chatMessages.length - 1;
  const msg = state.chatMessages[assistantIdx];
  let firstTokenAt = 0;
  let chunkTokens = 0;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        server_id: state.chatServerId,
        messages: state.chatMessages.slice(0, -1).map((m) => ({
          role: m.role,
          content: m.content,
        })),
        stream: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {
          const json = JSON.parse(payload);
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) {
            if (!firstTokenAt) firstTokenAt = performance.now();
            chunkTokens += 1;
            msg.content += delta;
            if (!msg.timingsLocked) {
              const elapsed = (performance.now() - firstTokenAt) / 1000;
              if (elapsed > 0.05) msg.tps = chunkTokens / elapsed;
            }
            renderChat();
          }
          const timings = json.timings;
          if (timings && typeof timings === "object") {
            if (timings.predicted_per_second != null) {
              msg.tps = Number(timings.predicted_per_second);
              msg.timingsLocked = true;
            }
            if (timings.prompt_per_second != null) {
              msg.promptTps = Number(timings.prompt_per_second);
            }
            if (timings.predicted_n != null) {
              msg.tokens = Number(timings.predicted_n);
            }
            renderChat();
          }
        } catch {
          /* skip malformed chunks */
        }
      }
    }

    if (!msg.content) {
      msg.content = "(empty response)";
    }
    msg.streaming = false;
    renderChat();
  } catch (err) {
    msg.content = `Error: ${err.message}`;
    msg.streaming = false;
    renderChat();
  } finally {
    input.disabled = false;
    input.focus();
  }
}

function openSettings() {
  loadSettings().then(() => $("#settings-dialog").showModal());
}

function showWelcomeIfEnabled() {
  if (state.settings.show_splash_on_startup !== false) {
    renderWelcomeSlide(0);
    $("#welcome-dialog").showModal();
    startWelcomeAutoplay();
  }
}

function stopWelcomeAutoplay() {
  if (state.welcomeTimer) clearInterval(state.welcomeTimer);
  state.welcomeTimer = null;
}

function startWelcomeAutoplay() {
  stopWelcomeAutoplay();
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  state.welcomeTimer = setInterval(() => {
    renderWelcomeSlide(state.welcomeSlide + 1);
  }, 5000);
}

function restartWelcomeAutoplay() {
  if ($("#welcome-dialog").open) startWelcomeAutoplay();
}

function renderWelcomeSlide(index) {
  const slides = [...document.querySelectorAll(".welcome-slide")];
  const dots = [...document.querySelectorAll(".welcome-dot")];
  if (!slides.length) return;
  const previous = state.welcomeSlide;
  const next = (index + slides.length) % slides.length;
  const direction = index < previous ? "previous" : "next";
  state.welcomeSlide = next;
  slides.forEach((slide, slideIndex) => {
    const active = slideIndex === state.welcomeSlide;
    slide.hidden = !active;
    slide.classList.toggle("active", active);
    slide.classList.remove("slide-from-left", "slide-from-right");
    if (active && next !== previous) {
      void slide.offsetWidth;
      slide.classList.add(
        direction === "previous" ? "slide-from-left" : "slide-from-right"
      );
    }
  });
  dots.forEach((dot, dotIndex) => {
    const active = dotIndex === state.welcomeSlide;
    dot.classList.toggle("active", active);
    dot.setAttribute("aria-selected", String(active));
  });
}

async function closeWelcome(tab = null) {
  const showOnStartup = $("#welcome-show-startup").checked;
  if (showOnStartup !== (state.settings.show_splash_on_startup !== false)) {
    await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ show_splash_on_startup: showOnStartup }),
    });
    state.settings.show_splash_on_startup = showOnStartup;
    $("#settings-show-splash").checked = showOnStartup;
  }
  stopWelcomeAutoplay();
  $("#welcome-dialog").close();
  const currentSlide = document.querySelector(".welcome-slide.active");
  tab = tab || currentSlide?.dataset.welcomeTab || "load";
  if (tab === "analyzer") openAnalyzer();
  else showMainTab(tab);
}

async function saveSettings(e) {
  e.preventDefault();
  try {
    const fontSize = clampFontSize($("#settings-font-size").value);
    await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({
        llama_server_path: $("#settings-binary").value,
        scan_root: $("#settings-scan-root").value,
        min_model_size_mb: parseInt($("#settings-min-size").value, 10),
        default_ctx: parseInt($("#settings-default-ctx").value, 10),
        ui_font_size: fontSize,
        show_splash_on_startup: $("#settings-show-splash").checked,
      }),
    });
    applyFontSize(fontSize);
    $("#settings-dialog").close();
    await loadSettings();
    await refreshModels();
  } catch (err) {
    alert(err.message);
  }
}

function init() {
  initTooltips();
  showMainTab("load");
  $("#btn-refresh").addEventListener("click", refreshModels);
  $("#btn-settings").addEventListener("click", openSettings);
  $("#btn-welcome-continue").addEventListener("click", () => closeWelcome());
  $("#btn-welcome-prev").addEventListener("click", () => {
    renderWelcomeSlide(state.welcomeSlide - 1);
    restartWelcomeAutoplay();
  });
  $("#btn-welcome-next").addEventListener("click", () => {
    renderWelcomeSlide(state.welcomeSlide + 1);
    restartWelcomeAutoplay();
  });
  document.querySelectorAll("[data-welcome-slide]").forEach((button) => {
    button.addEventListener("click", () => {
      renderWelcomeSlide(Number(button.dataset.welcomeSlide));
      restartWelcomeAutoplay();
    });
  });
  const welcomeDialog = $("#welcome-dialog");
  welcomeDialog.addEventListener("close", stopWelcomeAutoplay);
  welcomeDialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      renderWelcomeSlide(state.welcomeSlide - 1);
      restartWelcomeAutoplay();
    }
    if (event.key === "ArrowRight") {
      renderWelcomeSlide(state.welcomeSlide + 1);
      restartWelcomeAutoplay();
    }
  });
  $("#hf-search-form").addEventListener("submit", searchHuggingFace);
  $("#btn-hf-clear").addEventListener("click", () => {
    $("#hf-search-input").value = "";
    $("#hf-results").innerHTML = "";
    $("#hf-detail").innerHTML = '<p class="muted">Select a model to see its GGUF files.</p>';
    $("#hf-result-count").textContent = "0 results";
    $("#hf-status").textContent = " · Search Hugging Face to find a model.";
    $("#btn-hf-more").classList.add("hidden");
    state.hfModels = [];
    state.hfRepo = null;
    $("#hf-search-input").focus();
  });
  document.querySelectorAll(".hf-suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      $("#hf-search-input").value = button.dataset.query;
      searchHuggingFace();
    });
  });
  $("#hf-sort").addEventListener("change", () => {
    if ($("#hf-search-input").value.trim()) searchHuggingFace();
  });
  $("#btn-hf-direction").addEventListener("click", (event) => {
    const button = event.currentTarget;
    const descending = button.dataset.direction === "-1";
    button.dataset.direction = descending ? "1" : "-1";
    button.textContent = descending ? "↑" : "↓";
    button.setAttribute("aria-label", descending ? "Sort ascending" : "Sort descending");
    if ($("#hf-search-input").value.trim()) searchHuggingFace();
  });
  $("#btn-hf-more").addEventListener("click", () => searchHuggingFace(null, true));
  $("#btn-toggle-models").addEventListener("click", () =>
    setModelsExpanded(!state.modelsExpanded)
  );
  const fontRange = $("#settings-font-size");
  if (fontRange) {
    fontRange.addEventListener("input", () => {
      const n = clampFontSize(fontRange.value);
      updateFontSizeLabel(n);
      applyFontSize(n);
    });
  }
  $("#model-search").addEventListener("input", renderModels);
  $("#launch-form").addEventListener("submit", startServer);
  $("#gpu-select").addEventListener("change", updateGpuWarning);
  const spillSel = $("#spill-select");
  if (spillSel) {
    spillSel.addEventListener("change", () => {
      updateSpillHint();
      updateGpuWarning();
    });
  }
  const mtpCheck = $("#mtp-check");
  if (mtpCheck) {
    mtpCheck.addEventListener("change", syncMtpControls);
    syncMtpControls();
  }
  const visionCheck = $("#vision-check");
  if (visionCheck) {
    visionCheck.addEventListener("change", syncVisionControls);
    syncVisionControls();
  }
  $("#ctx-select").addEventListener("change", onCtxSelectChange);
  const ctxSlider = $("#ctx-custom");
  if (ctxSlider) {
    ctxSlider.addEventListener("input", onCtxSliderInput);
    syncCtxSliderLabel();
  }
  document.querySelectorAll(".num-step").forEach((btn) => {
    btn.addEventListener("click", () => {
      stepNumber(btn.dataset.target, parseInt(btn.dataset.dir, 10));
      if (btn.dataset.target === "mtp-draft-n") updateGpuWarning();
    });
  });
  const mtpDraft = $("#mtp-draft-n");
  if (mtpDraft) mtpDraft.addEventListener("change", updateGpuWarning);
  $("#chat-form").addEventListener("submit", sendChat);
  $("#btn-close-chat").addEventListener("click", closeChat);
  const backdrop = $("#chat-backdrop");
  if (backdrop) backdrop.addEventListener("click", closeChat);
  const clearBtn = $("#btn-clear-chat");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      state.chatMessages = [];
      renderChat();
      $("#chat-input").focus();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("#analyzer-overlay").classList.contains("hidden")) {
        if ($("#az-io") && !$("#az-io").classList.contains("hidden")) {
          closeAnalyzerIo();
          return;
        }
        closeAnalyzer();
        return;
      }
      if (!$("#chat-overlay").classList.contains("hidden")) {
        closeChat();
      }
    }
  });
  document.querySelectorAll(".app-tab").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.tab === "analyzer") openAnalyzer();
      else showMainTab(button.dataset.tab);
    });
  });
  const azBack = $("#btn-az-back");
  if (azBack) azBack.addEventListener("click", closeAnalyzer);
  const azRef = $("#btn-az-refresh");
  if (azRef) azRef.addEventListener("click", refreshAnalyzer);
  const azLive = $("#btn-az-live");
  if (azLive) azLive.addEventListener("click", () => openAnalyzerIo({ followLive: true }));
  const azIoBack = $("#btn-az-io-back");
  if (azIoBack) azIoBack.addEventListener("click", closeAnalyzerIo);
  const azIoLive = $("#btn-az-io-live");
  if (azIoLive) {
    azIoLive.addEventListener("click", () => openAnalyzerIo({ followLive: true }));
  }
  const azSel = $("#az-server-select");
  if (azSel) {
    azSel.addEventListener("change", (e) => {
      state.analyzerServerId = e.target.value;
      refreshAnalyzer();
      if ($("#az-io") && !$("#az-io").classList.contains("hidden")) {
        openAnalyzerIo({ followLive: true });
      }
    });
  }
  $("#chat-server-select").addEventListener("change", (e) => {
    state.chatServerId = e.target.value;
    state.chatMessages = [];
    updateChatSubtitle();
    renderChat();
  });
  $("#chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#settings-form").addEventListener("submit", saveSettings);
  $("#btn-settings-cancel").addEventListener("click", () =>
    $("#settings-dialog").close()
  );
  $("#btn-close-logs").addEventListener("click", () =>
    $("#logs-dialog").close()
  );
  const unloadBtn = $("#btn-unload-all");
  if (unloadBtn) unloadBtn.addEventListener("click", unloadAll);
  const lanBtn = $("#btn-lan-access");
  if (lanBtn) lanBtn.addEventListener("click", toggleLanAccess);

  const settingsReady = loadSettings().then(showWelcomeIfEnabled);
  Promise.all([
    settingsReady,
    loadGpus(),
    loadModels(),
    loadFavorites(),
    loadServers(),
  ]).then(() => {
    setInterval(loadServers, 3000);
  });
}

init();
