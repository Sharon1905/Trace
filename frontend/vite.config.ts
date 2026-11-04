import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8001",
      "/projects": "http://127.0.0.1:8001",
      "/project_files": "http://127.0.0.1:8001",
      "/upload": "http://127.0.0.1:8001",
      "/ask": "http://127.0.0.1:8001",
      "/explain_repo": "http://127.0.0.1:8001",
      "/explain_file": "http://127.0.0.1:8001",
      "/explain_function": "http://127.0.0.1:8001",
      "/function_usage": "http://127.0.0.1:8001",
      "/unused_functions": "http://127.0.0.1:8001"
    }
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    restoreMocks: true
  }
});
