import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Native file-change events (inotify) are unreliable on Windows-mounted
    // drives under WSL (/mnt/c/...) -- edits don't trigger HMR without this,
    // and the dev server silently keeps serving stale compiled modules.
    watch: {
      usePolling: true,
    },
  },
})
