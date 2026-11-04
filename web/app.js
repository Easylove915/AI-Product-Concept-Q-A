/* RAG 知识库答疑系统 · 前端交互（原生 JS，无构建步骤） */
const $ = (id) => document.getElementById(id);
const SAMPLES = [
  "Agent 五要素是哪五个？",
  "RAG 的完整流程分几个阶段？",
  "Agent 的记忆体系怎么设计？",
  "灰度发布和 POC 验证有什么区别？",
  "多 Agent 是怎么协作的？",
  "需求优先级怎么排？",
];

let toastTimer = null;

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast show" + (isError ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = "toast"), 3600);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d{1,2})\]/g, (_, n) => `<button class="cite-ref" data-n="${n}">${n}</button>`);
}

function formatAnswer(text) {
  const lines = String(text).split("\n");
  const out = [];
  let inList = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-•*]\s+(.+)$/);
    if (bullet) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(bullet[1])}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/* ------------------------------ 状态 ------------------------------ */
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    $("modeBadge").textContent = "答案模式：" + s.mode;
    $("indexBadge").textContent = s.loaded
      ? `${s.num_documents || 0} 篇文档 · ${s.num_chunks || 0} 片段`
      : "索引未建立";
    $("statDocs").textContent = s.num_documents || 0;
    $("statChunks").textContent = s.num_chunks || 0;
    const src = s.github_repo ? `${s.github_repo} @ ${s.github_branch}` : (s.docs_dir || "");
    // 界面上只显示末两级目录，避免把本机绝对路径（盘符、用户名）暴露在演示画面上；
    // 完整路径留在 tooltip 里，自己调试时悬停可看。
    const short = s.github_repo
      ? src
      : (src.split(/[\\/]+/).filter(Boolean).slice(-2).join(" / ") || "本地文档");
    const line = $("sourceLine");
    line.textContent = short;
    line.title = src || "";
    $("sourceDot").className = "dot" + (s.loaded ? "" : " dot-off");
    if (s.last_error) toast(s.last_error, true);
  } catch (e) {
    toast("状态读取失败：" + e.message, true);
  }
}

/* ------------------------------ 同步 ------------------------------ */
async function sync() {
  const btn = $("syncBtn");
  btn.disabled = true;
  btn.classList.add("spinning");
  try {
    const r = await api("/api/sync", {});
    toast(`索引已重建：${r.num_documents} 篇文档 → ${r.num_chunks} 个片段`);
    await refreshStatus();
  } catch (e) {
    toast("重建失败：" + e.message, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("spinning");
  }
}

/* ------------------------------ 渲染答案 ------------------------------ */
function renderCitations(citations) {
  $("citeCount").textContent = citations.length;
  const section = $("citationSection");
  if (!citations.length) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  $("citeGrid").innerHTML = citations.map((c) => `
    <div class="cite-card" id="cite-${c.index}" data-n="${c.index}">
      <div class="cite-top">
        <div class="cite-num">${c.index}</div>
        <div class="cite-info">
          <b>${esc(c.doc_title || c.doc_path)}</b>
          <small>${esc(c.doc_path)} › ${esc(c.heading || "")}</small>
        </div>
        <span class="cite-score">${c.relevance}%</span>
      </div>
      <div class="cite-body">
        <pre>${esc(c.content)}</pre>
        <div class="cite-links">
          <span>行 L${c.line_start}–L${c.line_end} · BM25 ${c.bm25} · TF-IDF ${c.tfidf}${c.dense ? ` · Dense ${c.dense}` : ""}</span>
          ${c.github_url ? `<a href="${esc(c.github_url)}" target="_blank" rel="noopener">在 GitHub 查看原文 ↗</a>` : ""}
        </div>
      </div>
    </div>`).join("");
}

function renderHits(hits) {
  $("hitCount").textContent = hits.length;
  const section = $("hitSection");
  if (!hits.length) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  $("hitBody").innerHTML = hits.map((h) => {
    const c = h.chunk;
    return `<tr>
      <td>${h.rank}</td>
      <td><span class="path" title="${esc(c.doc_path)} › ${esc(c.heading)}">${esc(c.doc_path)} › ${esc(c.heading)}</span></td>
      <td class="num"><span class="bar" style="width:${Math.min(60, h.relevance * .6)}px"></span>${h.relevance}%</td>
      <td class="num">${h.bm25}</td>
      <td class="num">${h.tfidf}</td>
      <td class="num">${h.dense || "—"}</td>
      <td><button class="look-btn" data-hit="${h.rank}">查看片段</button></td>
    </tr>
    <tr class="snippet-row" id="snip-${h.rank}" hidden><td colspan="7"><pre style="white-space:pre-wrap;font:12px/1.7 inherit;color:var(--muted);margin:0">${esc(c.content)}</pre></td></tr>`;
  }).join("");
}

function renderResult(r) {
  $("empty").classList.add("hidden");
  $("answerCard").classList.remove("hidden");
  $("answerMode").textContent = r.mode_label || r.mode;
  $("answerMeta").textContent = `${r.citations.length} 条引用 · ${r.hits.length} 个片段 · ${r.elapsed_ms} ms · ${r.model}`;
  $("answerBody").innerHTML = formatAnswer(r.answer);
  $("warnings").innerHTML = (r.warnings || []).map((w) => `<div>⚠ ${esc(w)}</div>`).join("");
  renderCitations(r.citations || []);
  renderHits(r.hits || []);
  const ctx = $("contextSection");
  ctx.hidden = false;
  ctx.open = false;
  $("contextBody").textContent = r.context || "（无检索上下文）";
}

async function ask() {
  const q = $("query").value.trim();
  if (!q) { toast("请输入问题", true); return; }
  const btn = $("askBtn");
  btn.disabled = true;
  btn.textContent = "检索并生成中…";
  try {
    const r = await api("/api/ask", { query: q });
    renderResult(r);
  } catch (e) {
    toast("生成失败：" + e.message, true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '生成答案 <span>→</span>';
  }
}

/* ------------------------------ 事件 ------------------------------ */
$("syncBtn").addEventListener("click", sync);
$("askBtn").addEventListener("click", ask);
$("query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});

$("samples").innerHTML = SAMPLES.map((s) => `<div class="chip">${esc(s)}</div>`).join("");
$("samples").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  $("query").value = e.target.textContent;
  $("query").focus();
  ask();
});

document.addEventListener("click", (e) => {
  const ref = e.target.closest(".cite-ref");
  if (ref) {
    const card = $("cite-" + ref.dataset.n);
    if (card) {
      document.querySelectorAll(".cite-card").forEach((c) => c.classList.remove("open"));
      card.classList.add("open");
      card.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      toast(`引用 [${ref.dataset.n}] 没有对应的检索片段`, true);
    }
    return;
  }
  const top = e.target.closest(".cite-top");
  if (top) { top.parentElement.classList.toggle("open"); return; }
  const look = e.target.closest(".look-btn");
  if (look) {
    const row = $("snip-" + look.dataset.hit);
    if (row) row.hidden = !row.hidden;
    return;
  }
});

refreshStatus();

// 支持 ?q=问题 预置并自动提问：演示时可以直接把「某个问题」的完整效果发出去，
// 不用让对方再手打一遍。例如 http://127.0.0.1:8765/?q=RAG 的完整流程分几个阶段
const preset = new URLSearchParams(location.search).get("q");
if (preset) {
  $("query").value = preset;
  ask();
}
