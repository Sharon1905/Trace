import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiUrl, askProject, deleteProject, fetchProjectFiles, fetchProjects, getJson, renameProject, uploadZip } from "./api";
import type { AskResponse, ProjectFilesResponse, ProjectSession, RetrievalMode, SourceReference } from "./types";

type View = "ask" | "files" | "overview" | "file" | "function" | "usage" | "candidates";
type AskTurn = { question: string; response: AskResponse };
type Health = { status: string; version: string };
type ToolResult = Record<string, unknown>;

const SESSION_KEY = "trace.projects.v2";
const ACTIVE_KEY = "trace.active_project.v2";
const MAX_ZIP_BYTES = 25 * 1024 * 1024;
const views: { id: View; label: string; icon: string }[] = [
  { id: "ask", label: "Ask TRACE", icon: "✳" },
  { id: "files", label: "Uploaded files", icon: "📁" },
  { id: "overview", label: "Repository overview", icon: "⌂" },
  { id: "file", label: "Explain a file", icon: "▤" },
  { id: "function", label: "Explain a function", icon: "ƒ" },
  { id: "usage", label: "Find function usage", icon: "↗" },
  { id: "candidates", label: "Unused candidates", icon: "◌" }
];

function readSavedProjects(): ProjectSession[] {
  try {
    const raw = localStorage.getItem(SESSION_KEY) ?? sessionStorage.getItem(SESSION_KEY);
    const value: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(value) ? value as ProjectSession[] : [];
  } catch {
    return [];
  }
}

function readSavedActive(): string {
  try {
    return localStorage.getItem(ACTIVE_KEY) ?? sessionStorage.getItem(ACTIVE_KEY) ?? readSavedProjects()[0]?.project_id ?? "";
  } catch {
    return "";
  }
}


function displayName(filename: string): string {
  return filename.replace(/\.zip$/i, "").replace(/[-_]+/g, " ");
}

function queryPath(path: string, projectId: string, field?: string, value?: string): string {
  const params = new URLSearchParams({ project_id: projectId });
  if (field && value !== undefined) params.set(field, value);
  return `${path}?${params.toString()}`;
}

function responseReferences(value: ToolResult | null): SourceReference[] {
  if (!value) return [];
  if (Array.isArray(value.references)) return value.references as SourceReference[];
  if (value.reference && typeof value.reference === "object") return [value.reference as SourceReference];
  return [];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Check that the TRACE server is reachable.";
}

function EvidenceCard({ reference }: { reference: SourceReference }) {
  return (
    <article className="evidence-card">
      <div className="evidence-title">
        <span className="file-mark" aria-hidden="true">PY</span>
        <div className="evidence-path">
          <strong title={reference.file_path}>{reference.file_path}</strong>
          <span>{reference.symbol || reference.kind || "source"} · lines {reference.line_start}–{reference.line_end}</span>
        </div>
      </div>
      <pre className="code-block"><code>{reference.code}</code></pre>
    </article>
  );
}

function ToolOutput({ result }: { result: ToolResult }) {
  const refs = responseReferences(result);
  const candidates = Array.isArray(result.unused_functions) ? result.unused_functions as Record<string, unknown>[] : null;
  const usages = Array.isArray(result.usages) ? result.usages as Record<string, unknown>[] : null;
  const explanation = typeof result.explanation === "string" ? result.explanation : null;
  const summary = Array.isArray(result.summary) ? result.summary as string[] : null;
  const limitation = typeof result.limitations === "string" ? result.limitations : null;

  return (
    <div className="tool-output">
      {summary && <div className="summary-list">{summary.map((line, index) => <p key={`${index}-${line}`}>{line}</p>)}</div>}
      {explanation && <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{explanation}</ReactMarkdown></div>}
      {candidates && (candidates.length ? (
        <div className="candidate-list">{candidates.map((item, index) => (
          <div className="candidate-row" key={`${String(item.file)}:${String(item.line)}:${index}`}>
            <code>{String(item.qualified_name ?? item.function)}</code>
            <span>{String(item.file)}:{String(item.line)}</span>
            <small>{String(item.note ?? "Static candidate for review.")}</small>
          </div>
        ))}</div>
      ) : <div className="empty-panel">TRACE found no unused-function candidates.</div>)}
      {usages && (usages.length ? (
        <div className="usage-list">{usages.map((item, index) => (
          <div className="usage-row" key={`${String(item.file)}:${String(item.line)}:${index}`}>
            <code>{String(item.expression)}</code>
            <span>{String(item.file)}:{String(item.line)}</span>
            <small>Called from {String(item.caller)}</small>
          </div>
        ))}</div>
      ) : <div className="empty-panel">No static call sites were found for this function.</div>)}
      {!summary && !explanation && !candidates && !usages && (
        <pre className="json-output">{JSON.stringify(result, null, 2)}</pre>
      )}
      {limitation && <p className="limitation"><span aria-hidden="true">ⓘ</span> {limitation}</p>}
      {refs.length > 0 && <div className="tool-evidence">{refs.map((reference, index) => <EvidenceCard key={`${reference.chunk_id ?? reference.file_path}-${index}`} reference={reference} />)}</div>}
    </div>
  );
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSession[]>(readSavedProjects);
  const [activeProjectId, setActiveProjectId] = useState<string>(readSavedActive);
  const [view, setView] = useState<View>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadNotice, setUploadNotice] = useState("");
  const [autoUploadOnSelect, setAutoUploadOnSelect] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("hybrid");
  const [turns, setTurns] = useState<AskTurn[]>([]);
  const [askBusy, setAskBusy] = useState(false);
  const [askError, setAskError] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [filePath, setFilePath] = useState("");
  const [functionName, setFunctionName] = useState("");
  const [toolResult, setToolResult] = useState<ToolResult | null>(null);
  const [toolBusy, setToolBusy] = useState(false);
  const [toolError, setToolError] = useState("");
  const [openMenuProjectId, setOpenMenuProjectId] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<ProjectSession | null>(null);
  const [renameName, setRenameName] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ProjectSession | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const activeProjectIdRef = useRef(activeProjectId);
  const currentViewRef = useRef(view);
  activeProjectIdRef.current = activeProjectId;
  currentViewRef.current = view;

  const activeProject = useMemo(
    () => projects.find((project) => project.project_id === activeProjectId) ?? null,
    [projects, activeProjectId]
  );
  const evidence = useMemo(() => {
    const latestTurn = turns.at(-1);
    return latestTurn?.response.references ?? responseReferences(toolResult);
  }, [turns, toolResult]);

  useEffect(() => {
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(projects));
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(projects));
    } catch {}
  }, [projects]);

  useEffect(() => {
    if (activeProjectId) {
      try {
        localStorage.setItem(ACTIVE_KEY, activeProjectId);
        sessionStorage.setItem(ACTIVE_KEY, activeProjectId);
      } catch {}
    }
  }, [activeProjectId]);

  useEffect(() => {
    let mounted = true;
    fetchProjects()
      .then((backendProjects) => {
        if (!mounted || !Array.isArray(backendProjects) || backendProjects.length === 0) return;
        setProjects((current) => {
          const currentMap = new Map(current.map((p) => [p.project_id, p]));
          const merged: ProjectSession[] = [];
          for (const bp of backendProjects) {
            const existing = currentMap.get(bp.project_id);
            merged.push(existing ? { ...bp, name: existing.name || bp.name } : bp);
            currentMap.delete(bp.project_id);
          }
          for (const rem of currentMap.values()) {
            merged.push(rem);
          }
          return merged;
        });
        setActiveProjectId((curr) => {
          if (curr && backendProjects.some((p) => p.project_id === curr)) return curr;
          const saved = readSavedActive();
          if (saved && backendProjects.some((p) => p.project_id === saved)) return saved;
          return backendProjects[0].project_id;
        });
      })
      .catch(() => {
        // Safe offline fallback
      });
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    let mounted = true;
    function checkHealth() {
      fetch(apiUrl("/health"), { headers: { Accept: "application/json" } })
        .then((response) => {
          if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
          return response.json() as Promise<Health>;
        })
        .then((data: Health) => {
          if (typeof data?.status !== "string" || typeof data?.version !== "string") throw new Error("TRACE returned an invalid health response.");
          if (mounted) { setHealth(data); setHealthError(""); }
        })
        .catch((error: unknown) => { if (mounted) { setHealth(null); setHealthError(errorMessage(error)); } });
    }
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  useEffect(() => {
    setToolResult(null);
    setToolError("");
    setTurns([]);
    setAskError("");
    setAskBusy(false);
    setPendingQuestion("");
    setToolBusy(false);
  }, [activeProjectId]);

  useEffect(() => {
    if (view !== "overview" && view !== "candidates") {
      setToolResult(null);
      setToolError("");
      setToolBusy(false);
    }
  }, [view]);

  useEffect(() => {
    if (!activeProjectId || (view !== "overview" && view !== "candidates")) return;
    let cancelled = false;
    setToolBusy(true);
    setToolError("");
    const path = view === "overview" ? "/explain_repo" : "/unused_functions";
    getJson<ToolResult>(path, { project_id: activeProjectId })
      .then((result: ToolResult) => { if (!cancelled) setToolResult(result); })
      .catch((error: unknown) => { if (!cancelled) setToolError(errorMessage(error)); })
      .finally(() => { if (!cancelled) setToolBusy(false); });
    return () => { cancelled = true; };
  }, [view, activeProjectId]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (openMenuProjectId) {
          setOpenMenuProjectId(null);
        } else if (renameTarget && !renameBusy) {
          setRenameTarget(null);
          setRenameError("");
        } else if (deleteTarget && !deleteBusy) {
          setDeleteTarget(null);
          setDeleteError("");
        }
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [openMenuProjectId, renameTarget, renameBusy, deleteTarget, deleteBusy]);

  useEffect(() => {
    if (!openMenuProjectId) return;
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as HTMLElement | null;
      if (target && !target.closest(".project-menu-anchor")) {
        setOpenMenuProjectId(null);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [openMenuProjectId]);

  function handleStartRename(project: ProjectSession) {
    setOpenMenuProjectId(null);
    setRenameTarget(project);
    setRenameName(project.name);
    setRenameError("");
  }

  async function handleSaveRename(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!renameTarget || renameBusy) return;
    const trimmed = renameName.trim();
    if (!trimmed) {
      setRenameError("Repository name cannot be empty.");
      return;
    }
    setRenameBusy(true);
    setRenameError("");
    try {
      const updated = await renameProject(renameTarget.project_id, trimmed);
      setProjects((prev) => prev.map((p) => p.project_id === updated.project_id ? { ...p, name: updated.name } : p));
      setRenameTarget(null);
      setUploadNotice(`Repository renamed to "${updated.name}".`);
    } catch (err) {
      setRenameError(errorMessage(err));
    } finally {
      setRenameBusy(false);
    }
  }

  function handleStartDelete(project: ProjectSession) {
    setOpenMenuProjectId(null);
    setDeleteTarget(project);
    setDeleteError("");
  }

  async function handleConfirmDelete() {
    if (!deleteTarget || deleteBusy) return;
    const targetId = deleteTarget.project_id;
    const targetName = deleteTarget.name;
    setDeleteBusy(true);
    setDeleteError("");
    try {
      await deleteProject(targetId);
      setProjects((prev) => {
        const remaining = prev.filter((p) => p.project_id !== targetId);
        if (activeProjectId === targetId) {
          setActiveProjectId(remaining.length > 0 ? remaining[0].project_id : "");
        }
        return remaining;
      });
      setDeleteTarget(null);
      setUploadNotice(`Repository "${targetName}" deleted.`);
    } catch (err) {
      setDeleteError(errorMessage(err));
    } finally {
      setDeleteBusy(false);
    }
  }

  async function performUpload(fileToUpload: File) {
    if (!fileToUpload || uploadBusy) return;
    setUploadError("");
    setUploadNotice("");
    if (!fileToUpload.name.toLowerCase().endsWith(".zip")) {
      setUploadError("Choose a .zip archive. TRACE accepts ZIP uploads only.");
      return;
    }
    if (fileToUpload.size > MAX_ZIP_BYTES) {
      setUploadError("This ZIP exceeds TRACE's 25 MiB compressed upload limit.");
      return;
    }
    setUploadBusy(true);
    setUploadFile(fileToUpload);
    try {
      const result = await uploadZip(fileToUpload);
      const project: ProjectSession = {
        ...result,
        name: displayName(fileToUpload.name),
        added_at: new Date().toISOString()
      };
      setProjects((current) => [project, ...current.filter((item) => item.project_id !== project.project_id)]);
      setActiveProjectId(project.project_id);
      setUploadFile(null);
      setUploadNotice(`${project.name} is ready with ${project.python_files} Python files and ${project.indexed_chunks} indexed chunks.`);
      if (project.parse_errors.length) setUploadNotice(`${project.name} is ready; ${project.parse_errors.length} Python file(s) had parse errors and were retained as source.`);
      setView("ask");
    } catch (error) {
      setUploadError(errorMessage(error));
    } finally {
      setUploadBusy(false);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (uploadFile) {
      await performUpload(uploadFile);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) return;
    setUploadFile(file);
    setUploadError("");
    if (autoUploadOnSelect) {
      setAutoUploadOnSelect(false);
      void performUpload(file);
    }
  }

  function handleDragOver(event: React.DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    if (!uploadBusy) setIsDragging(true);
  }

  function handleDragLeave(event: React.DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }

  function handleDrop(event: React.DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    if (uploadBusy) return;
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void performUpload(file);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    const requestedProjectId = activeProjectId;
    if (!requestedProjectId || !text || askBusy) return;
    setPendingQuestion(text);
    setAskBusy(true);
    setAskError("");
    try {
      const effectiveMode = activeProject?.semantic_index_status === "ready" ? retrievalMode : "lexical";
      const response = await askProject(requestedProjectId, text, effectiveMode);
      if (activeProjectIdRef.current === requestedProjectId) {
        setTurns((current) => [...current, { question: text, response }]);
        setQuestion("");
      }
    } catch (error) {
      if (activeProjectIdRef.current === requestedProjectId) setAskError(errorMessage(error));
    } finally {
      if (activeProjectIdRef.current === requestedProjectId) {
        setPendingQuestion("");
        setAskBusy(false);
      }
    }
  }

  async function handleTool(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeProjectId || toolBusy) return;
    const requestedProjectId = activeProjectId;
    const requestedView = view;
    setToolBusy(true);
    setToolError("");
    setToolResult(null);
    let path = "";
    if (view === "file") {
      if (!filePath.trim()) { setToolBusy(false); return; }
      path = queryPath("/explain_file", requestedProjectId, "filename", filePath.trim());
    } else if (view === "function") {
      if (!functionName.trim()) { setToolBusy(false); return; }
      path = queryPath("/explain_function", requestedProjectId, "function_name", functionName.trim());
    } else {
      if (!functionName.trim()) { setToolBusy(false); return; }
      path = queryPath("/function_usage", requestedProjectId, "function_name", functionName.trim());
    }
    try {
      const response = await fetch(apiUrl(path), { headers: { Accept: "application/json" } });
      const text = await response.text();
      let parsed: unknown;
      try { parsed = text ? JSON.parse(text) : null; }
      catch { throw new Error(`TRACE returned an unreadable response (HTTP ${response.status}).`); }
      if (!response.ok) {
        const detail = (parsed as { detail?: unknown } | null)?.detail;
        const message = Array.isArray(detail)
          ? detail.map((item) => typeof item === "string" ? item : item?.msg ?? "Invalid request").join("; ")
          : typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`;
        throw new Error(message);
      }
      if (!parsed || typeof parsed !== "object") throw new Error("TRACE returned an empty or invalid response.");
      const toolPayload = parsed as ToolResult;
      const references = responseReferences(toolPayload);
      if (toolPayload.project_id !== requestedProjectId || references.some((reference) => !reference.file_path || !Number.isInteger(reference.line_start) || !Number.isInteger(reference.line_end) || reference.line_end < reference.line_start || typeof reference.code !== "string" || (reference.project_id != null && reference.project_id !== requestedProjectId))) {
        throw new Error("TRACE returned a response that does not match the selected repository.");
      }
      if (activeProjectIdRef.current === requestedProjectId && currentViewRef.current === requestedView) setToolResult(toolPayload);
    } catch (error) {
      if (activeProjectIdRef.current === requestedProjectId && currentViewRef.current === requestedView) setToolError(errorMessage(error));
    } finally {
      if (activeProjectIdRef.current === requestedProjectId && currentViewRef.current === requestedView) setToolBusy(false);
    }
  }

  const currentView = views.find((item) => item.id === view) ?? views[0];
  const canAsk = Boolean(activeProject && question.trim() && !askBusy);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#workspace" aria-label="TRACE home">
          <span className="brand-mark">T</span>
          <span><strong>TRACE</strong><small>CODE INTELLIGENCE</small></span>
        </a>
        <div className={`backend-status ${health ? "is-online" : "is-offline"}`} role="status">
          <span className="status-dot" />
          <span>{health ? `Backend online · v${health.version}` : "Backend unavailable"}</span>
        </div>
        {healthError && <p className="backend-error" title={healthError}>Start FastAPI to connect.</p>}

        <section className="sidebar-section" aria-labelledby="projects-heading">
          <div className="section-heading"><h2 id="projects-heading">Repositories</h2><span>{projects.length}</span></div>
          <form className="upload-form" onSubmit={handleUpload}>
            <label
              className={`upload-drop ${isDragging ? "is-drag-over" : ""}`}
              htmlFor="repo-upload"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <span className="upload-plus" aria-hidden="true">＋</span>
              <strong>{uploadFile ? uploadFile.name : "Add repository"}</strong>
              <small>{uploadFile ? `${(uploadFile.size / (1024 * 1024)).toFixed(1)} MiB selected` : "ZIP archive · up to 25 MiB"}</small>
            </label>
            <input id="repo-upload" type="file" accept=".zip,application/zip" onChange={handleFileChange} />
            <button className="button button-primary upload-button" disabled={!uploadFile || uploadBusy || !health} aria-label="Upload ZIP">
              {uploadBusy ? <><span className="spinner" /> Indexing repository…</> : "Upload ZIP"}
            </button>
          </form>
          <p className="upload-hint">Python files are indexed for code search. Other files stay available for repository exploration.</p>
          {uploadError && <p className="inline-error" role="alert">{uploadError}</p>}
          {uploadNotice && <p className="inline-success" role="status">{uploadNotice}</p>}

          <div className="project-list" aria-label="Projects uploaded this session">
            {projects.length === 0 ? (
              <p className="no-projects">Your uploaded repositories will appear here.</p>
            ) : (
              projects.map((project) => (
                <div
                  key={project.project_id}
                  className={`project-item ${activeProjectId === project.project_id ? "selected" : ""}`}
                >
                  <button
                    type="button"
                    className="project-select-btn"
                    onClick={() => setActiveProjectId(project.project_id)}
                    aria-pressed={activeProjectId === project.project_id}
                  >
                    <span className="project-icon" aria-hidden="true">⌘</span>
                    <span className="project-copy">
                      <strong>{project.name}</strong>
                      <small>{project.python_files} Python files · {project.indexed_chunks} chunks</small>
                    </span>
                    <span
                      className={`project-health ${project.semantic_index_status === "ready" ? "ready" : "fallback"}`}
                      title={project.semantic_index_status === "ready" ? "Semantic index ready" : "Lexical retrieval available"}
                    />
                  </button>

                  <div className="project-menu-anchor">
                    <button
                      type="button"
                      className={`project-menu-btn ${openMenuProjectId === project.project_id ? "active" : ""}`}
                      aria-label="Repository options"
                      title={`Options for ${project.name}`}
                      aria-haspopup="menu"
                      aria-expanded={openMenuProjectId === project.project_id}
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuProjectId((current) => current === project.project_id ? null : project.project_id);
                      }}
                    >
                      <span aria-hidden="true">•••</span>
                    </button>

                    {openMenuProjectId === project.project_id && (
                      <div className="project-context-menu" role="menu" aria-label={`Options for ${project.name}`}>
                        <button
                          type="button"
                          role="menuitem"
                          className="context-menu-item"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartRename(project);
                          }}
                        >
                          <span className="menu-item-icon" aria-hidden="true">✎</span>
                          <span>Rename</span>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          className="context-menu-item danger"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartDelete(project);
                          }}
                        >
                          <span className="menu-item-icon" aria-hidden="true">✕</span>
                          <span>Delete</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <nav className="workspace-nav" aria-label="Repository tools">
          <div className="section-heading"><h2>Workspace</h2></div>
          {views.map((item) => (
            <button key={item.id} className={`nav-item ${view === item.id ? "active" : ""}`} onClick={() => setView(item.id)} aria-current={view === item.id ? "page" : undefined} disabled={!activeProject}>
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>{item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer"><span className="footer-dot" /> Local-first · Project-isolated</div>
      </aside>

      <main className="workspace" id="workspace">
        <header className="topbar">
          <div className="breadcrumbs"><span>TRACE</span><span className="crumb-separator">/</span><span>{activeProject?.name ?? "Workspace"}</span><span className="crumb-separator">/</span><strong>{currentView.label}</strong></div>
          <div className="topbar-right"><span className="local-badge"><span /> LOCAL SESSION</span></div>
        </header>

        {!activeProject ? (
          <section
            className={`welcome-screen ${isDragging ? "is-drag-over" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="welcome-orbit" aria-hidden="true"><div /><div /><div /></div>
            <p className="eyebrow">CODE, WITH CONTEXT</p>
            <h1>Understand the shape<br />of your codebase.</h1>
            <p className="welcome-copy">Upload a Python repository to ask grounded questions, inspect source evidence, and explore how the pieces fit together.</p>
            <div className="welcome-steps"><span><b>01</b> Upload a ZIP</span><span><b>02</b> Ask in plain language</span><span><b>03</b> Trace every answer to code</span></div>

            {uploadBusy ? (
              <div className="welcome-uploading-card">
                <span className="spinner spinner-large" />
                <div className="welcome-uploading-info">
                  <h3>Indexing {uploadFile ? uploadFile.name : "repository"}…</h3>
                  <p>Extracting Python files, parsing AST structures, generating vector embeddings, and building the FAISS index.</p>
                </div>
              </div>
            ) : uploadFile ? (
              <div className="welcome-staged-box">
                <div className="welcome-staged-info">
                  <span className="project-icon" aria-hidden="true">📦</span>
                  <div>
                    <strong>{uploadFile.name}</strong>
                    <span>{(uploadFile.size / (1024 * 1024)).toFixed(2)} MB · Ready to index</span>
                  </div>
                </div>
                <div className="welcome-staged-actions">
                  <button
                    type="button"
                    className="button button-primary welcome-staged-upload-btn"
                    onClick={() => performUpload(uploadFile)}
                    disabled={!health}
                  >
                    Upload & Inspect Repository <span aria-hidden="true">→</span>
                  </button>
                  <button
                    type="button"
                    className="button welcome-change-btn"
                    onClick={() => {
                      setAutoUploadOnSelect(true);
                      document.getElementById("repo-upload")?.click();
                    }}
                  >
                    Choose different file
                  </button>
                </div>
              </div>
            ) : (
              <div className="welcome-actions">
                <button
                  type="button"
                  className="button button-primary welcome-button"
                  onClick={() => {
                    setAutoUploadOnSelect(true);
                    document.getElementById("repo-upload")?.click();
                  }}
                  disabled={!health}
                >
                  Choose a repository <span aria-hidden="true">→</span>
                </button>
                <span className="welcome-or">or drag & drop a .zip file anywhere</span>
              </div>
            )}

            {uploadError && (
              <div className="notice notice-error welcome-error-notice">
                <span aria-hidden="true">!</span>
                <div>
                  <strong>Upload failed</strong>
                  <p>{uploadError}</p>
                </div>
              </div>
            )}

            {!health && <p className="welcome-backend">Start the backend at <code>127.0.0.1:8000</code> to begin.</p>}
          </section>
        ) : (
          <div className="main-layout">
            <section className="content-column">
              <div className="page-heading">
                <div><p className="eyebrow">{view === "ask" ? "REPOSITORY ASSISTANT" : "REPOSITORY EXPLORER"}</p><h1>{currentView.label}</h1><p>{view === "ask" ? `Ask a question about ${activeProject.name}. TRACE will show the code it used.` : `Explore ${activeProject.name} using TRACE's indexed source and static analysis.`}</p></div>
                <span className="project-chip"><span className="project-chip-dot" />{activeProject.name}</span>
              </div>

              {activeProject.semantic_index_status !== "ready" && (
                <div className="notice notice-warn"><span aria-hidden="true">!</span><div><strong>Semantic retrieval is unavailable for this project.</strong><p>TRACE will use lexical matching for questions. {activeProject.semantic_index_error || "No semantic index was created during upload."}</p></div></div>
              )}

              {view === "ask" ? (
                <>
                  <div className="conversation" aria-live="polite">
                    {turns.length === 0 && !askBusy && (
                      <div className="ask-empty"><div className="sparkle">✳</div><h2>Start with a question</h2><p>Ask how a function works, where a setting is used, or how a request moves through the repository.</p><div className="example-queries"><span>Try questions like</span><code>How is an uploaded archive checked?</code><code>Where is the retry limit configured?</code></div></div>
                    )}
                    {turns.map((turn, index) => {
                      const check = turn.response.retrieval_diagnostics?.evidence_check;
                      const fallback = turn.response.retrieval_diagnostics?.fallback_reason;
                      return <article className="turn" key={`${turn.response.project_id}-${index}`}>
                        <div className="question-line"><span className="avatar user-avatar">Y</span><div><small>YOU</small><p>{turn.question}</p></div></div>
                        <div className="answer-line"><span className="avatar trace-avatar">T</span><div className="answer-content">
                          <div className="answer-meta"><small>TRACE</small><span className={`evidence-state ${check?.status === "insufficient" ? "state-warn" : "state-grounded"}`}>{check?.status === "insufficient" ? "INSUFFICIENT EVIDENCE" : turn.response.references.length ? "SOURCE ANCHORS PRESENT" : "REPOSITORY RESPONSE"}</span></div>
                          <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.response.answer || "TRACE returned no answer text."}</ReactMarkdown></div>
                          {fallback && <p className="fallback-note">Using {turn.response.retrieval_mode || "lexical"} retrieval: {fallback}</p>}
                          {check?.status === "insufficient" && <p className="grounding-note">{check.reason}</p>}
                          {turn.response.references.length > 0 && <div className="mobile-evidence"><h3>Source evidence</h3>{turn.response.references.map((reference, refIndex) => <EvidenceCard key={`${reference.chunk_id ?? reference.file_path}-${refIndex}`} reference={reference} />)}</div>}
                        </div></div>
                      </article>;
                    })}
                    {askBusy && <div className="pending-answer"><span className="avatar trace-avatar">T</span><div><div className="answer-meta"><small>TRACE</small><span className="spinner" /> SEARCHING SOURCE</div><p>Looking through the indexed code for “{pendingQuestion}”</p></div></div>}
                    {askError && <div className="notice notice-error" role="alert"><span>!</span><div><strong>Question could not be completed</strong><p>{askError}</p></div></div>}
                  </div>
                  <form className="ask-composer" onSubmit={handleAsk}>
                    <label htmlFor="question-input" className="sr-only">Ask a question about this repository</label>
                    <textarea id="question-input" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about a function, setting, or flow…" rows={2} maxLength={4000} disabled={askBusy} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (canAsk) event.currentTarget.form?.requestSubmit(); } }} />
                    <div className="composer-bottom"><div className="composer-options"><label htmlFor="retrieval-mode">RETRIEVAL</label><select id="retrieval-mode" value={activeProject.semantic_index_status === "ready" ? retrievalMode : "lexical"} onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)} title={activeProject.semantic_index_status === "ready" ? "Select retrieval mode" : "Semantic index unavailable; using lexical retrieval"}><option value="hybrid" disabled={activeProject.semantic_index_status !== "ready"}>{activeProject.semantic_index_status === "ready" ? "Hybrid" : "Hybrid (unavailable)"}</option><option value="semantic" disabled={activeProject.semantic_index_status !== "ready"}>{activeProject.semantic_index_status === "ready" ? "Semantic" : "Semantic (unavailable)"}</option><option value="lexical">Lexical</option></select><span className="composer-help">Enter to ask · Shift+Enter for a new line</span></div><button className="button button-primary ask-button" aria-label="Submit question to TRACE" disabled={!canAsk}>{askBusy ? "Searching…" : "Ask TRACE"}<span aria-hidden="true">↗</span></button></div>
                  </form>
                  <p className="privacy-note">TRACE checks for direct question-term matches in retrieved source. Review the code to confirm full context.</p>
                </>
              ) : view === "files" ? (
                <FilesView
                  projectId={activeProject.project_id}
                  onExplainFile={(chosenPath) => {
                    setFilePath(chosenPath);
                    setView("file");
                  }}
                />
              ) : (
                <Explorer view={view} filePath={filePath} setFilePath={setFilePath} functionName={functionName} setFunctionName={setFunctionName} result={toolResult} busy={toolBusy} error={toolError} onSubmit={handleTool} />
              )}
            </section>

            <aside className="evidence-rail" aria-label="Source evidence">
              <div className="rail-heading"><div><p className="eyebrow">TRACEABILITY</p><h2>Source evidence</h2></div><span className="evidence-count">{evidence.length.toString().padStart(2, "0")}</span></div>
              <p className="rail-description">Retrieved source snippets with their original project paths and line ranges.</p>
              {evidence.length ? <div className="evidence-list">{evidence.map((reference, index) => <EvidenceCard key={`${reference.chunk_id ?? reference.file_path}-${index}`} reference={reference} />)}</div> : <div className="rail-empty"><span className="rail-empty-icon">⌁</span><strong>No source attached</strong><p>Ask a repository question or explain a file or function to see the exact source TRACE used.</p></div>}
              <div className="rail-footnote"><span>↳</span><p>Paths and line numbers come directly from TRACE's project-local index.</p></div>
            </aside>
          </div>
        )}
      </main>
      <footer className="mobile-footer">TRACE · local-first code intelligence</footer>

      {renameTarget && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target === e.currentTarget && !renameBusy) {
              setRenameTarget(null);
              setRenameError("");
            }
          }}
        >
          <div
            className="modal-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="rename-modal-title"
          >
            <div className="modal-header">
              <h3 id="rename-modal-title">Rename Repository</h3>
              <p className="modal-description">Update the display name for this repository.</p>
            </div>
            <form onSubmit={handleSaveRename} className="modal-form">
              <label htmlFor="rename-input" className="modal-label">
                Repository Name
              </label>
              <input
                id="rename-input"
                type="text"
                className="modal-input"
                value={renameName}
                onChange={(e) => {
                  setRenameName(e.target.value);
                  if (renameError) setRenameError("");
                }}
                autoFocus
                disabled={renameBusy}
                placeholder="Repository display name"
              />
              {renameError && <p className="inline-error" role="alert">{renameError}</p>}
              <div className="modal-actions">
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={() => {
                    if (!renameBusy) {
                      setRenameTarget(null);
                      setRenameError("");
                    }
                  }}
                  disabled={renameBusy}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="button button-primary"
                  disabled={renameBusy || !renameName.trim()}
                >
                  {renameBusy ? <><span className="spinner" /> Saving…</> : "Save"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target === e.currentTarget && !deleteBusy) {
              setDeleteTarget(null);
              setDeleteError("");
            }
          }}
        >
          <div
            className="modal-dialog delete-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-modal-title"
          >
            <div className="modal-header">
              <h3 id="delete-modal-title">Delete Repository</h3>
              <p className="modal-description">
                Are you sure you want to delete <strong>{deleteTarget.name}</strong>?
              </p>
            </div>
            <div className="modal-body-warning">
              <p>
                This will permanently remove the repository, its stored files, indexed code chunks, and vector-store embeddings from disk. This action cannot be undone.
              </p>
              <p className="delete-id-note">
                Project ID: <code>{deleteTarget.project_id}</code>
              </p>
            </div>
            {deleteError && <p className="inline-error" role="alert">{deleteError}</p>}
            <div className="modal-actions">
              <button
                type="button"
                className="button button-secondary"
                onClick={() => {
                  if (!deleteBusy) {
                    setDeleteTarget(null);
                    setDeleteError("");
                  }
                }}
                disabled={deleteBusy}
              >
                Cancel
              </button>
              <button
                type="button"
                className="button button-danger"
                onClick={handleConfirmDelete}
                disabled={deleteBusy}
              >
                {deleteBusy ? <><span className="spinner" /> Deleting…</> : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Explorer({ view, filePath, setFilePath, functionName, setFunctionName, result, busy, error, onSubmit }: {
  view: View;
  filePath: string;
  setFilePath: (value: string) => void;
  functionName: string;
  setFunctionName: (value: string) => void;
  result: ToolResult | null;
  busy: boolean;
  error: string;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const needsInput = view === "file" || view === "function" || view === "usage";
  const inputLabel = view === "file" ? "Project-relative file path" : view === "function" ? "Function name" : "Function name";
  const placeholder = view === "file" ? "src/package/module.py" : "package.module.function or function_name";
  return (
    <div className="explorer-body">
      {needsInput && <form className="tool-form" onSubmit={onSubmit}>
        <label htmlFor="tool-query">{inputLabel}</label>
        <div className="tool-input-row"><input id="tool-query" value={view === "file" ? filePath : functionName} onChange={(event) => view === "file" ? setFilePath(event.target.value) : setFunctionName(event.target.value)} placeholder={placeholder} required />
          <button className="button button-primary" disabled={busy || !(view === "file" ? filePath.trim() : functionName.trim())}>{busy ? "Working…" : view === "file" ? "Explain file" : view === "function" ? "Explain function" : "Find usage"}</button></div>
        <p className="field-hint">{view === "file" ? "Use the path relative to the uploaded repository root." : "TRACE matches the function's simple or qualified Python name."}</p>
      </form>}
      {view === "overview" && <div className="explorer-intro"><span className="explorer-icon">⌂</span><h2>Repository map</h2><p>A concise overview built from files in the selected project.</p></div>}
      {view === "candidates" && <div className="explorer-intro"><span className="explorer-icon">◌</span><h2>Static review candidates</h2><p>Functions without a statically detected call site. Dynamic calls and framework hooks may be missed; review each result before removing code.</p></div>}
      {busy && <div className="loading-panel" role="status"><span className="spinner" /> {view === "overview" ? "Mapping repository…" : view === "candidates" ? "Scanning Python functions…" : "Inspecting source…"}</div>}
      {error && <div className="notice notice-error" role="alert"><span>!</span><div><strong>TRACE could not complete this action</strong><p>{error}</p></div></div>}
      {!busy && !error && result && <div className="result-panel"><ToolOutput result={result} /></div>}
      {!busy && !error && !result && needsInput && <div className="empty-panel">Enter a {view === "file" ? "file path" : "function name"} to inspect this repository.</div>}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function FilesView({
  projectId,
  onExplainFile,
}: {
  projectId: string;
  onExplainFile: (filePath: string) => void;
}) {
  const [data, setData] = useState<ProjectFilesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetchProjectFiles(projectId)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [projectId]);

  function loadPreview(filePath: string) {
    if (selectedPath === filePath) {
      setSelectedPath(null);
      setPreviewContent(null);
      return;
    }
    setSelectedPath(filePath);
    setPreviewLoading(true);
    getJson<{ reference?: { code?: string } }>("/explain_file", { project_id: projectId, filename: filePath })
      .then((res) => {
        setPreviewContent(res.reference?.code ?? "(No content returned)");
      })
      .catch((err) => {
        setPreviewContent(`Error loading preview: ${errorMessage(err)}`);
      })
      .finally(() => {
        setPreviewLoading(false);
      });
  }

  const files = useMemo(() => {
    if (!data) return [];
    const q = filter.trim().toLowerCase();
    if (!q) return data.files;
    return data.files.filter((f) =>
      f.path.toLowerCase().includes(q) ||
      f.symbols.some((s) => s.toLowerCase().includes(q))
    );
  }, [data, filter]);

  return (
    <div className="files-view">
      <div className="files-header">
        <div className="files-stats">
          <span className="stat-pill"><strong>{data?.total_files ?? 0}</strong> files</span>
          <span className="stat-pill"><strong>{data?.python_files ?? 0}</strong> Python modules</span>
        </div>
        <div className="files-search-box">
          <input
            type="search"
            placeholder="Filter files by path or symbol…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="files-search-input"
            aria-label="Filter files"
          />
        </div>
      </div>

      {loading && <div className="loading-panel"><span className="spinner" /> Loading repository files…</div>}
      {error && <div className="notice notice-error" role="alert"><span>!</span><div><strong>Could not load project files</strong><p>{error}</p></div></div>}

      {!loading && !error && files.length === 0 && (
        <div className="empty-panel">{filter ? "No files matched your filter." : "No files found in this repository."}</div>
      )}

      {!loading && !error && files.length > 0 && (
        <div className="files-table-wrapper">
          <table className="files-table">
            <thead>
              <tr>
                <th>File Path</th>
                <th>Size</th>
                <th>Lines</th>
                <th>Symbols</th>
                <th className="th-actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file) => (
                <tr key={file.path} className={selectedPath === file.path ? "file-row selected-file-row" : "file-row"}>
                  <td className="file-cell-path">
                    <span className="file-badge-ext">{file.is_python ? "PY" : "FILE"}</span>
                    <strong title={file.path}>{file.path}</strong>
                    {file.syntax_error && <span className="syntax-error-pill" title={file.syntax_error}>Syntax error</span>}
                  </td>
                  <td className="file-cell-size">{formatBytes(file.size_bytes)}</td>
                  <td className="file-cell-lines">{file.line_count}</td>
                  <td className="file-cell-symbols">
                    {file.symbols.length > 0 ? (
                      <div className="symbols-wrap">
                        {file.symbols.slice(0, 3).map((sym) => (
                          <span key={sym} className="symbol-tag" title={sym}>{sym}</span>
                        ))}
                        {file.symbols.length > 3 && (
                          <span className="symbol-more">+{file.symbols.length - 3}</span>
                        )}
                      </div>
                    ) : (
                      <span className="empty-symbol">—</span>
                    )}
                  </td>
                  <td className="file-cell-actions">
                    <button
                      type="button"
                      className="button button-action-sm"
                      onClick={() => loadPreview(file.path)}
                      title="Preview source code"
                    >
                      {selectedPath === file.path ? "Hide" : "Preview"}
                    </button>
                    <button
                      type="button"
                      className="button button-action-sm button-accent-sm"
                      onClick={() => onExplainFile(file.path)}
                      title="Explain this file in detail"
                    >
                      Explain →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedPath && (
        <div className="file-preview-card">
          <div className="file-preview-header">
            <div>
              <span className="eyebrow">SOURCE PREVIEW</span>
              <h3>{selectedPath}</h3>
            </div>
            <button
              type="button"
              className="button button-action-sm"
              onClick={() => setSelectedPath(null)}
            >
              Close ✕
            </button>
          </div>
          {previewLoading ? (
            <div className="loading-panel"><span className="spinner" /> Loading source preview…</div>
          ) : (
            <pre className="code-block"><code>{previewContent}</code></pre>
          )}
        </div>
      )}
    </div>
  );
}

