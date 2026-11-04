export type RetrievalMode = "lexical" | "semantic" | "hybrid";

export interface SourceReference {
  project_id?: string | null;
  chunk_id?: string | null;
  window_id?: string | null;
  file_path: string;
  kind?: string | null;
  line_start: number;
  line_end: number;
  code: string;
  symbol?: string | null;
  score?: number | null;
}

export interface UploadResponse {
  project_id: string;
  indexed_chunks: number;
  python_files: number;
  parse_errors: { file: string; error: string }[];
  semantic_index_status: string;
  embedding_model?: string | null;
  semantic_index_error?: string | null;
}

export interface AskResponse {
  project_id: string;
  mode: string;
  answer?: string | null;
  references: SourceReference[];
  retrieval_mode?: string | null;
  retrieval_diagnostics?: {
    requested_mode?: string;
    used_mode?: string;
    fallback_reason?: string | null;
    evidence_check?: {
      status: "supported" | "insufficient";
      sufficient: boolean;
      query_content_term_count: number;
      anchored_terms: string[];
      required_anchor_count: number;
      reason: string;
    };
    [key: string]: unknown;
  } | null;
}

export interface ProjectSession extends UploadResponse {
  name: string;
  added_at: string;
}

export interface ProjectFileInfo {
  path: string;
  name: string;
  size_bytes: number;
  line_count: number;
  is_python: boolean;
  chunks: number;
  symbols: string[];
  syntax_error?: string | null;
}

export interface ProjectFilesResponse {
  project_id: string;
  total_files: number;
  python_files: number;
  files: ProjectFileInfo[];
}

export interface ProjectDeleteResponse {
  project_id: string;
  deleted: boolean;
  message: string;
}

export interface ApiError extends Error {
  status?: number;
}

