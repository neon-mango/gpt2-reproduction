import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' — работа из любого подкаталога (GitHub Pages проектных сайтов)
export default defineConfig({
    base: './',
    plugins: [react()],
});
