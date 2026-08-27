import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "ui",
  plugins: [react()],
  // Tauri points its dev window here and fails loudly if the port moves.
  server: { port: 1420, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true },
});
