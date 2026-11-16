import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built files are served by FastAPI (app.py); `npm run dev` proxies /api to a local `python app.py`.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
