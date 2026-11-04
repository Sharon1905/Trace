import type { AskResponse, ProjectDeleteResponse, ProjectFilesResponse, ProjectSession, RetrievalMode, SourceReference, UploadResponse } from "./types";

const configuredBase = import.meta.env.VITE_TRACE_API_BASE_URL?.trim() ?? "";
const API_BASE_URL = configuredBase.replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function readResponse<T>(response: Response): Promise<T> {
  const raw = await response.text();
  let payload: unknown = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      throw new Error(`TRACE returned an unreadable response (HTTP ${response.status}).`);
    }
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown } | null)?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => typeof item === "string" ? item : item?.msg ?? "Invalid request").join("; ")
      : typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`;
    const error = new Error(message) as Error & { status?: number };
    error.status = response.status;
    throw error;
  }
  if (payload === null || typeof payload !== "object") {
    throw new Error("TRACE returned an empty or invalid response.");
  }
  return payload as T;
}

function isSourceReference(value: unknown): value is SourceReference {
  if (!value || typeof value !== "object") return false;
  const reference = value as Partial<SourceReference>;
  return typeof reference.file_path === "string"
    && reference.file_path.length > 0
    && Number.isInteger(reference.line_start)
    && Number.isInteger(reference.line_end)
    && Number(reference.line_start) > 0
    && Number(reference.line_end) >= Number(reference.line_start)
    && typeof reference.code === "string";
}

export async function getJson<T>(path: string, params: Record<string, string>): Promise<T> {
  const query = new URLSearchParams(params);
  return readResponse<T>(await fetch(apiUrl(`${path}?${query}`), { headers: { Accept: "application/json" } }));
}

export async function uploadZip(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const result = await readResponse<UploadResponse>(await fetch(apiUrl("/upload"), { method: "POST", body: form }));
  if (!/^[0-9a-f]{32}$/.test(result.project_id)
      || !Number.isInteger(result.indexed_chunks)
      || !Number.isInteger(result.python_files)
      || !Array.isArray(result.parse_errors)
      || typeof result.semantic_index_status !== "string") {
    throw new Error("TRACE returned an invalid upload response. No project was selected.");
  }
  return result;
}

export async function askProject(projectId: string, question: string, retrievalMode: RetrievalMode): Promise<AskResponse> {
  const result = await readResponse<AskResponse>(await fetch(apiUrl("/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ project_id: projectId, question, retrieval_mode: retrievalMode })
  }));
  if (result.project_id !== projectId
      || typeof result.mode !== "string"
      || !(typeof result.answer === "string" || result.answer === null || result.answer === undefined)
      || !Array.isArray(result.references)
      || !result.references.every(isSourceReference)
      || result.references.some((reference) => reference.project_id != null && reference.project_id !== projectId)) {
    throw new Error("TRACE returned an invalid answer or source-reference response.");
  }
  return result;
}

export async function fetchProjects(): Promise<ProjectSession[]> {
  return readResponse<ProjectSession[]>(await fetch(apiUrl("/projects"), { headers: { Accept: "application/json" } }));
}

export async function fetchProjectFiles(projectId: string): Promise<ProjectFilesResponse> {
  return getJson<ProjectFilesResponse>("/project_files", { project_id: projectId });
}

export async function renameProject(projectId: string, name: string): Promise<ProjectSession> {
  return readResponse<ProjectSession>(await fetch(apiUrl(`/projects/${projectId}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ name })
  }));
}

export async function deleteProject(projectId: string): Promise<ProjectDeleteResponse> {
  return readResponse<ProjectDeleteResponse>(await fetch(apiUrl(`/projects/${projectId}`), {
    method: "DELETE",
    headers: { Accept: "application/json" }
  }));
}

