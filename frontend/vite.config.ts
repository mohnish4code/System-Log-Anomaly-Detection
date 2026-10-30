import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and forwards API calls to FastAPI on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    chunkSizeWarningLimit: 600, // recharts alone is ~510 kB minified (~150 kB gzipped)
    rollupOptions: {
      output: { manualChunks: { react: ["react", "react-dom"], charts: ["recharts"] } },
    },
  },
});
