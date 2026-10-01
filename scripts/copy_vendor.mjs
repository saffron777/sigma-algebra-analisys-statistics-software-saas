// Copies third-party browser assets from node_modules into the Flask static folder,
// so the app works offline (no CDN at runtime).
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "sigma", "presentation", "web", "static", "vendor");

const assets = [
  ["node_modules/plotly.js-dist-min/plotly.min.js", "plotly.min.js"],
  ["node_modules/alpinejs/dist/cdn.min.js", "alpine.min.js"],
  ["node_modules/marked/marked.min.js", "marked.min.js"],
  ["node_modules/dompurify/dist/purify.min.js", "purify.min.js"],
];

mkdirSync(target, { recursive: true });
for (const [source, name] of assets) {
  const from = join(root, source);
  if (!existsSync(from)) {
    console.warn(`[copy_vendor] missing ${source} (run npm install)`);
    continue;
  }
  cpSync(from, join(target, name));
  console.log(`[copy_vendor] ${name}`);
}
