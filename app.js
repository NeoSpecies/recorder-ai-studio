const API = "";
let state = { projects: [], currentProject: null, theme: localStorage.getItem("recorder-ai-theme") || "auto" };
let activeSpeaker = "all";
let activeViewMode = "paragraph";
let searchText = "";

const scenarioLabels = {
  meeting: "会议",
  sales: "销售",
  interview: "访谈",
  lecture: "课程"
};

function qs(id) { return document.getElementById(id); }

function showError(error) {
  console.error(error);
  updateMetrics("处理失败", 0);
  alert(error?.message || String(error));
}

function normalizeApiError(status, bodyText) {
  try {
    const payload = JSON.parse(bodyText);
    const detail = payload.detail;
    if (detail && typeof detail === "object") {
      const model = detail.status || {};
      const downloaded = model.downloadedMB != null ? `${model.downloadedMB}/${model.estimatedTotalMB || "?"}MB` : "未知进度";
      return new Error(`${status} ${detail.message || "请求失败"}\n模型状态：${model.state || "unknown"} · ${downloaded}\n说明：当前不会使用 mock/fallback，请等待模型下载完成后重试。`);
    }
    return new Error(`${status} ${typeof detail === "string" ? detail : bodyText}`);
  } catch (_) {
    return new Error(`${status} ${bodyText}`);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(`${API}${url}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  if (!response.ok) throw normalizeApiError(response.status, await response.text());
  return response.json();
}

async function loadProjects() {
  const data = await requestJson("/api/projects");
  state.projects = data.projects || [];
  state.currentProject = state.projects[0] || null;
  await refreshAsrStatus();
}

async function refreshAsrStatus() {
  try {
    const data = await requestJson("/api/asr/status");
    const status = data.funasr || {};
    const label = status.ready
      ? `模型就绪 · ${status.downloadedMB || "?"}MB`
      : status.state === "incomplete"
        ? `模型下载中 · ${status.downloadedMB || 0}/${status.estimatedTotalMB || 936}MB`
        : "模型未就绪";
    qs("storageStatus").textContent = label;
  } catch (error) {
    qs("storageStatus").textContent = "模型状态未知";
  }
}

async function refreshProject(projectId = state.currentProject?.id) {
  if (!projectId) return;
  const data = await requestJson(`/api/projects/${projectId}`);
  state.currentProject = data.project;
  const idx = state.projects.findIndex(p => p.id === data.project.id);
  if (idx >= 0) state.projects[idx] = data.project;
  else state.projects.unshift(data.project);
  renderAll();
}

async function saveProject(status = "已保存") {
  if (!state.currentProject) return;
  const data = await requestJson(`/api/projects/${state.currentProject.id}`, {
    method: "PUT",
    body: JSON.stringify({
      title: state.currentProject.title,
      scene: state.currentProject.scene,
      glossary: state.currentProject.glossary,
      segments: state.currentProject.segments,
      tags: state.currentProject.tags,
      todos: state.currentProject.todos,
      insights: state.currentProject.insights,
      duration: state.currentProject.duration
    })
  });
  state.currentProject = data.project;
  updateMetrics(status, 100);
  renderProjectNav();
}

function formatTime(seconds = 0) {
  if (!Number.isFinite(Number(seconds))) return "00:00";
  const total = Math.max(0, Math.floor(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatBytes(bytes = 0) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeTag(value) {
  const tag = String(value || "").trim();
  if (!tag) return "";
  return tag.startsWith("#") ? tag : `#${tag}`;
}

function getSpeakers() {
  const map = new Map();
  (state.currentProject?.segments || []).forEach(segment => {
    if (!map.has(segment.speaker)) map.set(segment.speaker, { id: segment.speaker, name: segment.name || "未命名", count: 0 });
    map.get(segment.speaker).count += 1;
  });
  return Array.from(map.values()).sort((a, b) => String(a.id).localeCompare(String(b.id)));
}

function updateMetrics(status = null, progress = null) {
  const p = state.currentProject;
  if (!p) {
    qs("recordingTitle").textContent = "请新建录音项目";
    qs("fileNameText").textContent = "尚未导入音频";
    qs("fileMetaText").textContent = "当前没有任何项目或转写数据。请先新建项目并上传真实音频。";
    qs("durationText").textContent = "--:--";
    qs("speakerCountText").textContent = "0";
    qs("segmentCountText").textContent = "0 个转写段落";
    qs("progressText").textContent = status || "待新建";
    qs("progressBar").style.width = `${progress ?? 0}%`;
    qs("audioStateText").textContent = "未加载";
    refreshAsrStatus();
    return;
  }
  qs("recordingTitle").textContent = p.title || "未命名录音";
  qs("fileNameText").textContent = p.audio?.name || "尚未导入音频";
  qs("fileMetaText").textContent = p.audio ? `${p.audio.type || "未知格式"} · ${formatBytes(p.audio.size)} · 后端已保存` : "导入后会上传到本机后端 uploads 目录，只允许触发真实 FunASR 转写。";
  qs("durationText").textContent = p.duration ? formatTime(p.duration) : "--:--";
  qs("speakerCountText").textContent = getSpeakers().length;
  qs("segmentCountText").textContent = `${(p.segments || []).length} 个转写段落`;
  qs("progressText").textContent = status || ((p.segments || []).length ? `可校对 · ${p.transcriptionSource || "未知来源"}` : "待真实转写");
  if (progress !== null) qs("progressBar").style.width = `${progress}%`;
  qs("audioStateText").textContent = p.audio ? "本地后端文件" : "未加载";
  refreshAsrStatus();
}

function renderProjectNav() {
  const nav = qs("projectNav");
  nav.innerHTML = "";
  state.projects.forEach(project => {
    const button = document.createElement("button");
    button.className = `nav-item ${project.id === state.currentProject?.id ? "active" : ""}`;
    button.innerHTML = `<span>${escapeHtml(project.title)}</span><small>${project.segments?.length || 0}</small>`;
    button.addEventListener("click", () => refreshProject(project.id));
    nav.appendChild(button);
  });
}

function renderWaveform(duration = 0) {
  const waveform = qs("waveform");
  waveform.innerHTML = "";
  for (let i = 0; i < 120; i++) {
    const bar = document.createElement("i");
    const h = 10 + Math.abs(Math.sin((i + 1) * (duration || 47) * 0.013) * 32) + (i % 7) * 1.7;
    bar.style.height = `${h}px`;
    waveform.appendChild(bar);
  }
}

function renderSpeakers() {
  const row = qs("speakerRow");
  row.innerHTML = "";
  const all = document.createElement("button");
  all.className = `speaker ${activeSpeaker === "all" ? "active" : ""}`;
  all.textContent = "全部说话人";
  all.addEventListener("click", () => { activeSpeaker = "all"; renderSpeakers(); renderTranscript(); });
  row.appendChild(all);
  getSpeakers().forEach(speaker => {
    const button = document.createElement("button");
    button.className = `speaker ${activeSpeaker === speaker.id ? "active" : ""}`;
    button.textContent = `Speaker ${speaker.id} · ${speaker.name} (${speaker.count})`;
    button.addEventListener("click", () => { activeSpeaker = speaker.id; renderSpeakers(); renderTranscript(); });
    row.appendChild(button);
  });
}

function segmentMatches(segment) {
  const keyword = searchText.trim().toLowerCase();
  if (!keyword) return true;
  return [segment.textCorrected, segment.textRaw, segment.speaker, segment.name, ...(segment.tags || [])].join(" ").toLowerCase().includes(keyword);
}

function renderTranscript() {
  const list = qs("transcriptList");
  list.innerHTML = "";
  const segments = (state.currentProject?.segments || [])
    .filter(item => activeSpeaker === "all" || item.speaker === activeSpeaker)
    .filter(segmentMatches);
  if (!segments.length) {
    list.innerHTML = `<div class="empty-state padded">暂无真实转写段落。请上传音频后点击“真实转写音频”；本版本已移除示例/回退数据。</div>`;
    return;
  }
  segments.forEach(segment => {
    const item = document.createElement("article");
    item.className = `transcript-item mode-${activeViewMode}`;
    const changed = segment.textCorrected !== segment.textRaw;
    const body = activeViewMode === "diff" && changed
      ? `<p class="diff-text"><del>${escapeHtml(segment.textRaw)}</del><ins>${escapeHtml(segment.textCorrected)}</ins></p>`
      : `<p class="transcript-text" contenteditable="true" data-field="textCorrected">${escapeHtml(segment.textCorrected || segment.textRaw)}</p>`;
    item.innerHTML = `
      <button class="time" data-action="seek">${formatTime(segment.start)}</button>
      <div class="transcript-body">
        <div class="transcript-meta">
          <span class="badge badge-${String(segment.speaker).toLowerCase()}">Speaker ${escapeHtml(segment.speaker)}</span>
          <input class="speaker-name" data-field="name" value="${escapeHtml(segment.name || "未命名")}" />
          <span class="confidence">置信度 ${segment.confidence ?? "--"}%</span>
        </div>
        ${body}
        <div class="segment-tags">${(segment.tags || []).map(tag => `<button data-tag="${escapeHtml(tag)}">${escapeHtml(tag)} ×</button>`).join("")}</div>
        <div class="inline-tools">
          <button data-action="seek">播放此句</button>
          <button data-action="speaker">改说话人</button>
          <button data-action="tag">加标签</button>
          <button data-action="todo">转待办</button>
          <button data-action="delete">删除</button>
        </div>
      </div>`;
    item.querySelectorAll("[data-action='seek']").forEach(button => button.addEventListener("click", () => seekTo(segment.start)));
    item.querySelector("[data-field='name']").addEventListener("input", event => updateSegment(segment.id, { name: event.target.value }));
    const textNode = item.querySelector("[data-field='textCorrected']");
    if (textNode) textNode.addEventListener("input", event => updateSegment(segment.id, { textCorrected: event.target.textContent }));
    item.querySelector("[data-action='speaker']").addEventListener("click", () => changeSpeaker(segment.id));
    item.querySelector("[data-action='tag']").addEventListener("click", () => addTagToSegment(segment.id));
    item.querySelector("[data-action='todo']").addEventListener("click", () => segmentToTodo(segment.id));
    item.querySelector("[data-action='delete']").addEventListener("click", () => deleteSegment(segment.id));
    item.querySelectorAll("[data-tag]").forEach(button => button.addEventListener("click", () => removeSegmentTag(segment.id, button.dataset.tag)));
    list.appendChild(item);
  });
}

function updateSegment(id, patch) {
  const segment = state.currentProject.segments.find(item => item.id === id);
  if (!segment) return;
  Object.assign(segment, patch);
  renderSpeakers();
  renderInsights();
}

function seekTo(seconds) {
  const audio = qs("audioPlayer");
  if (audio.src) {
    audio.currentTime = Math.max(0, Number(seconds) || 0);
    audio.play().catch(() => {});
  }
}

function changeSpeaker(id) {
  const segment = state.currentProject.segments.find(item => item.id === id);
  if (!segment) return;
  const next = prompt("输入说话人编号，例如 A/B/C/D", segment.speaker);
  if (!next) return;
  segment.speaker = next.trim().toUpperCase().slice(0, 2);
  renderAll();
}

function addTagToSegment(id) {
  const segment = state.currentProject.segments.find(item => item.id === id);
  const tag = normalizeTag(prompt("输入标签", "#重点"));
  if (!segment || !tag) return;
  segment.tags = Array.from(new Set([...(segment.tags || []), tag]));
  state.currentProject.tags = Array.from(new Set([...(state.currentProject.tags || []), tag]));
  renderTranscript();
  renderTags();
}

function removeSegmentTag(id, tag) {
  const segment = state.currentProject.segments.find(item => item.id === id);
  if (!segment) return;
  segment.tags = (segment.tags || []).filter(item => item !== tag);
  renderTranscript();
}

function segmentToTodo(id) {
  const segment = state.currentProject.segments.find(item => item.id === id);
  if (!segment) return;
  state.currentProject.todos.push({ id: crypto.randomUUID(), title: (segment.textCorrected || "待办").slice(0, 30), desc: `来源 ${formatTime(segment.start)} · Speaker ${segment.speaker}`, owner: segment.name || "未分配", done: false });
  renderTodos();
}

function deleteSegment(id) {
  state.currentProject.segments = state.currentProject.segments.filter(item => item.id !== id);
  renderAll();
}

function addSegment() {
  if (!state.currentProject || !(state.currentProject.segments || []).length) {
    showError(new Error("没有真实转写结果前不能手动添加段落，避免混入 mock 内容。"));
    return;
  }
  const last = state.currentProject.segments.at(-1);
  state.currentProject.segments.push({ id: crypto.randomUUID(), start: last ? last.end + 1 : 0, end: last ? last.end + 12 : 12, speaker: "A", name: "未命名", confidence: 100, textRaw: "", textCorrected: "", tags: [] });
  renderAll();
}

function mergeSpeakersByName() {
  const nameToSpeaker = new Map();
  state.currentProject.segments.forEach(segment => {
    const key = String(segment.name || "").trim();
    if (!key) return;
    if (!nameToSpeaker.has(key)) nameToSpeaker.set(key, segment.speaker);
    segment.speaker = nameToSpeaker.get(key);
  });
  renderAll();
}

function renderTodos() {
  const list = qs("todoList");
  list.innerHTML = "";
  (state.currentProject?.todos || []).forEach(todo => {
    const item = document.createElement("label");
    item.className = "todo";
    item.innerHTML = `<input type="checkbox" ${todo.done ? "checked" : ""} /><span><b contenteditable="true" data-field="title">${escapeHtml(todo.title)}</b><p contenteditable="true" data-field="desc">${escapeHtml(todo.desc)}</p></span><span class="owner" contenteditable="true" data-field="owner">${escapeHtml(todo.owner || "未分配")}</span>`;
    item.querySelector("input").addEventListener("change", e => { todo.done = e.target.checked; });
    item.querySelectorAll("[contenteditable]").forEach(node => node.addEventListener("input", () => { todo[node.dataset.field] = node.textContent; }));
    list.appendChild(item);
  });
}

function renderTags() {
  const cloud = qs("tagCloud");
  cloud.innerHTML = "";
  (state.currentProject?.tags || []).forEach(tag => {
    const button = document.createElement("button");
    button.className = "tag" + (tag.includes("敏感") ? " secure" : tag.includes("FunASR") || tag.includes("Plaud") ? " hot" : "");
    button.textContent = `${tag} ×`;
    button.addEventListener("click", () => {
      state.currentProject.tags = state.currentProject.tags.filter(item => item !== tag);
      state.currentProject.segments.forEach(segment => segment.tags = (segment.tags || []).filter(item => item !== tag));
      renderTags();
      renderTranscript();
    });
    cloud.appendChild(button);
  });
}

function renderInsights() {
  const insights = state.currentProject?.insights;
  qs("summaryList").innerHTML = "";
  const summary = insights?.summary || [];
  if (!summary.length) {
    const li = document.createElement("li");
    li.className = "empty-state";
    li.textContent = "暂无真实转写内容，不能生成提要。";
    qs("summaryList").appendChild(li);
  } else {
    summary.forEach(text => {
      const li = document.createElement("li");
      li.textContent = text;
      qs("summaryList").appendChild(li);
    });
  }
  qs("decisionList").innerHTML = insights?.decisions?.length ? `<ul>${insights.decisions.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : `<span class="empty-state">暂无真实转写结论。</span>`;
  qs("riskList").innerHTML = insights?.risks?.length ? `<ul>${insights.risks.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : `<span class="empty-state">暂无真实转写风险。</span>`;
  renderMindmap(insights?.mindmap || []);
  qs("insightStatus").textContent = insights ? "已生成" : "待生成";
}

function renderMindmap(nodes) {
  if (!nodes.length) {
    qs("mindmap").innerHTML = `<div class="empty-state padded">暂无真实转写内容，不能生成脑图。</div>`;
    return;
  }
  const [root, ...branches] = nodes;
  qs("mindmap").innerHTML = `<div class="node root">${escapeHtml(root)}</div>` + branches.slice(0, 4).map((item, index) => `<div class="branch b${index + 1}"><span>${escapeHtml(item)}</span><em>${index === 0 ? "关键词" : index === 1 ? "主题" : index === 2 ? "行动" : "风险"}</em></div>`).join("");
}

async function ensureCurrentProject() {
  if (state.currentProject) return state.currentProject;
  const created = await requestJson("/api/projects", {
    method: "POST",
    body: JSON.stringify({ title: "未命名录音", scene: "meeting", glossary: [] })
  });
  state.currentProject = created.project;
  state.projects.unshift(created.project);
  renderProjectNav();
  return state.currentProject;
}

async function handleFile(file) {
  if (!file) return;
  await ensureCurrentProject();
  const form = new FormData();
  form.append("file", file);
  updateMetrics("上传中", 25);
  const response = await fetch(`/api/projects/${state.currentProject.id}/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  state.currentProject = data.project;
  qs("audioPlayer").src = `/api/projects/${state.currentProject.id}/audio?t=${Date.now()}`;
  updateMetrics("已上传", 60);
  renderProjectNav();
}

async function transcribeAudio() {
  if (!state.currentProject) throw new Error("请先上传真实音频。未上传音频时不会生成任何示例转写。");
  if (!state.currentProject.audio) throw new Error("请先上传真实音频文件。当前没有音频，不能转写。");
  updateMetrics("真实转写中", 75);
  const endpoint = qs("funasrEndpoint").value.trim();
  const data = await requestJson(`/api/projects/${state.currentProject.id}/transcribe`, {
    method: "POST",
    body: JSON.stringify({ endpoint })
  });
  if (data.source !== "local_funasr" && data.source !== "remote_funasr") {
    throw new Error(`转写来源异常：${data.source || "unknown"}。已拒绝展示非真实识别结果。`);
  }
  state.currentProject = data.project;
  const sourceLabel = data.source === "local_funasr" ? "本机 FunASR 完成" : "远程 FunASR 完成";
  updateMetrics(sourceLabel, 100);
  renderAll();
}

async function generateInsights() {
  if (!state.currentProject || !(state.currentProject.segments || []).length) {
    throw new Error("没有真实转写文本，不能生成整理结果。");
  }
  await saveProject("已同步");
  updateMetrics("整理中", 90);
  const data = await requestJson(`/api/projects/${state.currentProject.id}/insights`, { method: "POST", body: JSON.stringify({}) });
  state.currentProject = data.project;
  updateMetrics("已整理", 100);
  renderAll();
}

async function createProject() {
  const title = qs("newTitleInput").value.trim() || "长录音项目";
  const scene = qs("newSceneSelect").value;
  const glossary = qs("newGlossaryInput").value.split(",").map(item => item.trim()).filter(Boolean);
  const data = await requestJson("/api/projects", { method: "POST", body: JSON.stringify({ title, scene, glossary }) });
  state.projects.unshift(data.project);
  state.currentProject = data.project;
  activeSpeaker = "all";
  qs("newRecordingModal").classList.remove("show");
  renderAll();
}

async function applyScenario(scenario) {
  state.currentProject.scene = scenario;
  await requestJson(`/api/projects/${state.currentProject.id}`, { method: "PUT", body: JSON.stringify({ scene }) });
  await refreshProject(state.currentProject.id);
}

function exportJson() {
  const blob = new Blob([JSON.stringify(state.currentProject, null, 2)], { type: "application/json;charset=utf-8" });
  downloadBlob(`${safeFileName(state.currentProject.title)}.json`, blob);
}

function exportMarkdown() {
  window.open(`/api/projects/${state.currentProject.id}/export.md`, "_blank");
}

function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function safeFileName(value) {
  return String(value || "recording").replace(/[\\/:*?"<>|]/g, "-").slice(0, 60) || "recording";
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("recorder-ai-theme", theme);
  document.querySelectorAll(".theme-switch button").forEach(btn => btn.classList.toggle("active", btn.dataset.theme === theme));
}

function renderAll() {
  renderProjectNav();
  if (!state.currentProject) {
    renderWaveform(0);
    qs("speakerRow").innerHTML = "";
    qs("transcriptList").innerHTML = `<div class="empty-state padded">暂无项目和真实转写数据。请新建项目或直接上传音频。</div>`;
    qs("todoList").innerHTML = "";
    qs("tagCloud").innerHTML = "";
    renderInsights();
    updateMetrics(null, 0);
    qs("audioPlayer").removeAttribute("src");
    return;
  }
  renderWaveform(state.currentProject.duration || 0);
  renderSpeakers();
  renderTranscript();
  renderTodos();
  renderTags();
  renderInsights();
  updateMetrics(null, (state.currentProject.segments || []).length ? 100 : 0);
  document.querySelectorAll("#scenarioSwitch button").forEach(btn => btn.classList.toggle("active", btn.dataset.scenario === state.currentProject.scene));
  if (state.currentProject.audio) qs("audioPlayer").src = `/api/projects/${state.currentProject.id}/audio?t=${Date.now()}`;
}

function wireInteractions() {
  qs("newRecordingBtn").addEventListener("click", () => qs("newRecordingModal").classList.add("show"));
  qs("closeModalBtn").addEventListener("click", () => qs("newRecordingModal").classList.remove("show"));
  qs("createProjectBtn").addEventListener("click", () => createProject().catch(showError));
  qs("audioFileInput").addEventListener("change", e => handleFile(e.target.files?.[0]).catch(showError));
  const dropZone = qs("dropZone");
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragging"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
  dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("dragging"); handleFile(e.dataTransfer.files?.[0]).catch(showError); });
  qs("callFunasrBtn").addEventListener("click", () => transcribeAudio().catch(showError));
  qs("saveProjectBtn").addEventListener("click", () => saveProject().catch(showError));
  qs("generateInsightsBtn").addEventListener("click", () => generateInsights().catch(showError));
  qs("exportMarkdownBtn").addEventListener("click", exportMarkdown);
  qs("exportJsonBtn").addEventListener("click", exportJson);
  qs("addSegmentBtn").addEventListener("click", addSegment);
  qs("mergeSpeakersBtn").addEventListener("click", mergeSpeakersByName);
  qs("searchInput").addEventListener("input", e => { searchText = e.target.value; renderTranscript(); });
  qs("recordingTitle").addEventListener("input", e => { state.currentProject.title = e.target.textContent.trim() || "未命名录音"; renderProjectNav(); });
  qs("addTagBtn").addEventListener("click", () => {
    const tag = normalizeTag(qs("tagInput").value);
    if (!tag) return;
    state.currentProject.tags = Array.from(new Set([...(state.currentProject.tags || []), tag]));
    qs("tagInput").value = "";
    renderTags();
  });
  qs("playBtn").addEventListener("click", () => {
    const audio = qs("audioPlayer");
    if (!audio.src) return;
    audio.paused ? audio.play() : audio.pause();
  });
  const audio = qs("audioPlayer");
  audio.addEventListener("play", () => qs("playBtn").textContent = "Ⅱ");
  audio.addEventListener("pause", () => qs("playBtn").textContent = "▶");
  audio.addEventListener("loadedmetadata", () => { state.currentProject.duration = audio.duration || state.currentProject.duration || 0; updateMetrics(); renderWaveform(audio.duration); });
  audio.addEventListener("timeupdate", () => qs("playTimeText").textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration || state.currentProject.duration)}`);
  document.querySelectorAll("#aiTabs button").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll("#aiTabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
    qs(`${btn.dataset.tab}Tab`).classList.add("active");
  }));
  document.querySelectorAll("#scenarioSwitch button").forEach(btn => btn.addEventListener("click", () => applyScenario(btn.dataset.scenario).catch(showError)));
  document.querySelectorAll("#viewModeSwitch button").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll("#viewModeSwitch button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeViewMode = btn.dataset.mode;
    renderTranscript();
  }));
  document.querySelectorAll(".theme-switch button").forEach(btn => btn.addEventListener("click", () => applyTheme(btn.dataset.theme)));
}

async function init() {
  applyTheme(state.theme);
  wireInteractions();
  await loadProjects();
  renderAll();
}

init().catch(showError);
