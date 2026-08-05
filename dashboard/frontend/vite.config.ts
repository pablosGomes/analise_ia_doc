import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Em dev, o Vite roda em :5173 e repassa /api para o FastAPI local em :8090
// (server.py), inclusive o stream SSE. Em produção o build estático é servido
// pelo próprio FastAPI (ver ../server.py), então não há proxy nenhum.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8090", changeOrigin: true },
      "/saude": "http://127.0.0.1:8090",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
