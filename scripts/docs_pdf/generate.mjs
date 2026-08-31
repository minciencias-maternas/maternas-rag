#!/usr/bin/env node
/**
 * generate.mjs — Genera un PDF con estilo consistente a partir de un
 * Markdown del proyecto, renderizando los bloques ```mermaid``` como
 * diagramas reales (no como texto).
 *
 * Uso:
 *   node generate.mjs
 *   node generate.mjs --input ../../docs/OTRO_DOC.md --output ../../docs/OTRO_DOC.pdf
 *   node generate.mjs --header "Maternas RAG"
 *
 * Por defecto convierte docs/DOCUMENTACION_oss120b_20260831.md -> docs/DOCUMENTACION.pdf
 * (rutas relativas a la raíz del repo, dos niveles arriba de este script).
 *
 * No descarga Chromium: usa puppeteer-core apuntando al Chrome/Edge ya
 * instalado en la máquina (o PUPPETEER_EXECUTABLE_PATH si se define).
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = { input: null, output: null, header: null, verbose: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--input" || a === "-i") args.input = argv[++i];
    else if (a === "--output" || a === "-o") args.output = argv[++i];
    else if (a === "--header") args.header = argv[++i];
    else if (a === "--verbose") args.verbose = true;
    else if (a === "--help" || a === "-h") {
      console.log(
        "Uso: node generate.mjs [--input archivo.md] [--output archivo.pdf] [--header \"Texto\"] [--verbose]"
      );
      process.exit(0);
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));

const inputPath = path.resolve(
  REPO_ROOT,
  args.input || "docs/DOCUMENTACION_oss120b_20260831.md"
);
const outputPath = path.resolve(
  REPO_ROOT,
  args.output || inputPath.replace(/\.md$/i, ".pdf")
);

if (!existsSync(inputPath)) {
  console.error(`No existe el archivo de entrada: ${inputPath}`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Localizar un Chrome/Edge instalado (sin descargar Chromium propio)
// ---------------------------------------------------------------------------

function findChromeExecutable() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) {
    return process.env.PUPPETEER_EXECUTABLE_PATH;
  }

  const candidates = [];
  if (process.platform === "win32") {
    const pf = process.env["PROGRAMFILES"] || "C:\\Program Files";
    const pf86 = process.env["PROGRAMFILES(X86)"] || "C:\\Program Files (x86)";
    const localAppData = process.env["LOCALAPPDATA"] || "";
    candidates.push(
      path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(localAppData, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
      path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe")
    );
  } else if (process.platform === "darwin") {
    candidates.push(
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/Applications/Chromium.app/Contents/MacOS/Chromium"
    );
  } else {
    candidates.push(
      "/usr/bin/google-chrome",
      "/usr/bin/google-chrome-stable",
      "/usr/bin/chromium-browser",
      "/usr/bin/chromium",
      "/usr/bin/microsoft-edge"
    );
  }

  for (const c of candidates) {
    if (c && existsSync(c)) return c;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Markdown -> HTML
// ---------------------------------------------------------------------------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdownToHtml(markdown) {
  const renderer = new marked.Renderer();

  // Bloques ```mermaid``` -> <pre class="mermaid"> con el texto escapado
  // (mermaid.js en el navegador lee .textContent, así que el texto debe
  // llegar como entidades HTML, no como marcado real). marked@12 usa la
  // firma posicional clásica del Renderer, no tokens.
  const defaultCode = renderer.code.bind(renderer);
  renderer.code = (code, infostring, escaped) => {
    const lang = (infostring || "").trim();
    if (lang === "mermaid") {
      return `<pre class="mermaid">${escapeHtml(code)}</pre>\n`;
    }
    return defaultCode(code, infostring, escaped);
  };

  // Blockquotes: distinguir el callout de advertencia (arranca con ⚠️)
  // del callout informativo (metadata del encabezado, notas normales).
  // `quote` ya llega como HTML interno renderizado.
  renderer.blockquote = (quote) => {
    const isWarning = /⚠️/.test(quote);
    const cls = isWarning ? "callout callout-warn" : "callout callout-info";
    return `<blockquote class="${cls}">\n${quote}</blockquote>\n`;
  };

  marked.setOptions({ gfm: true, breaks: false });
  return marked.parse(markdown, { renderer });
}

function extractTitle(markdown, override) {
  if (override) return override;
  const m = markdown.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : "Documentación";
}

// ---------------------------------------------------------------------------
// Plantilla HTML — colores/estilos calcados del PDF de referencia
// (docs/DOCUMENTACION.pdf: h1 #1F497D, h2/h3 #2E75B6, callout #EBF3FB con
// barra izquierda #2E75B6, tablas con header #2E75B6 y bandas #EBF3FB,
// código inline #C7254E sobre #F4F4F4 con borde #DDDDDD).
// ---------------------------------------------------------------------------

function buildHtmlDocument(bodyHtml, title) {
  return `<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<style>
  :root {
    --blue-dark: #1F497D;
    --blue: #2E75B6;
    --blue-pale: #EBF3FB;
    --blue-rule: #BDD7EE;
    --gray-rule: #CCCCCC;
    --gray-border: #DDDDDD;
    --code-bg: #F4F4F4;
    --code-fg: #C7254E;
    --text: #1A1A1A;
    --muted: #444444;
    --warn-bg: #FFF7E6;
    --warn-border: #D9A400;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    color: var(--text);
    font-family: "Segoe UI", Arial, Helvetica, sans-serif;
    font-size: 9.5pt;
    line-height: 1.55;
  }
  body { padding: 0 2px; }
  h1, h2, h3, h4, h5 {
    color: var(--blue);
    font-weight: 600;
    break-after: avoid-page;
  }
  h1 {
    color: var(--blue-dark);
    font-size: 17pt;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--blue-dark);
    margin: 0 0 16px 0;
  }
  h2 {
    font-size: 13pt;
    margin: 26px 0 12px 0;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--blue-rule);
  }
  h3 {
    font-size: 11pt;
    margin: 18px 0 8px 0;
  }
  h4 {
    color: #262626;
    font-size: 10pt;
    margin: 16px 0 6px 0;
  }
  p { margin: 8px 0; }
  a { color: var(--blue); }
  ul, ol { margin: 8px 0; padding-left: 22px; }
  li { margin: 3px 0; }
  hr {
    border: none;
    height: 1px;
    background: var(--gray-rule);
    margin: 22px 0;
  }

  blockquote.callout {
    margin: 16px 0;
    padding: 10px 16px;
    background: var(--blue-pale);
    border-left: 4px solid var(--blue);
    border-radius: 2px;
    break-inside: avoid-page;
  }
  blockquote.callout-warn {
    background: var(--warn-bg);
    border-left-color: var(--warn-border);
  }
  blockquote.callout p { margin: 4px 0; color: var(--muted); font-size: 9pt; }
  blockquote.callout > *:first-child { margin-top: 0; }
  blockquote.callout > *:last-child { margin-bottom: 0; }

  code {
    font-family: Consolas, "Courier New", monospace;
    background: var(--code-bg);
    border: 1px solid var(--gray-border);
    border-radius: 3px;
    padding: 1px 5px;
    color: var(--code-fg);
    font-size: 0.88em;
  }
  pre {
    background: #F6F8FA;
    border: 1px solid var(--gray-border);
    border-radius: 4px;
    padding: 10px 14px;
    overflow-x: auto;
    break-inside: avoid-page;
    margin: 12px 0;
  }
  pre code {
    background: none;
    border: none;
    padding: 0;
    color: var(--text);
    font-size: 0.8em;
    line-height: 1.5;
    white-space: pre;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 0.92em;
  }
  thead { display: table-header-group; }
  tr { break-inside: avoid; }
  th {
    background: var(--blue);
    color: #FFFFFF;
    text-align: left;
    font-weight: 600;
    padding: 7px 10px;
  }
  td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--gray-border);
    vertical-align: top;
  }
  tbody tr:nth-child(even) td { background: var(--blue-pale); }

  .mermaid {
    text-align: center;
    margin: 16px 0;
    break-inside: avoid-page;
    background: none;
    border: none;
    padding: 0;
  }
  .mermaid svg {
    max-width: 100%;
    height: auto;
  }
</style>
</head>
<body>
${bodyHtml}
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Render y export a PDF
// ---------------------------------------------------------------------------

async function main() {
  const markdown = readFileSync(inputPath, "utf-8");
  const bodyHtml = renderMarkdownToHtml(markdown);
  const title = extractTitle(markdown, null);
  const headerText = args.header || title;
  const html = buildHtmlDocument(bodyHtml, title);

  const executablePath = findChromeExecutable();
  if (!executablePath) {
    console.error(
      "No se encontró Chrome ni Edge instalado. Definí PUPPETEER_EXECUTABLE_PATH " +
        "apuntando al ejecutable del navegador."
    );
    process.exit(1);
  }
  if (args.verbose) console.log(`Usando navegador: ${executablePath}`);

  const tmpDir = mkdtempSync(path.join(tmpdir(), "docs-pdf-"));
  const tmpHtmlPath = path.join(tmpDir, "doc.html");
  writeFileSync(tmpHtmlPath, html, "utf-8");

  const mermaidBundlePath = path.resolve(
    __dirname,
    "node_modules/mermaid/dist/mermaid.min.js"
  );
  const mermaidSource = readFileSync(mermaidBundlePath, "utf-8");

  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
  });

  try {
    const page = await browser.newPage();

    if (args.verbose) {
      page.on("console", (msg) => console.log(`[page] ${msg.text()}`));
      page.on("pageerror", (err) => console.error(`[pageerror] ${err}`));
    }

    await page.goto("file://" + tmpHtmlPath.replace(/\\/g, "/"), {
      waitUntil: "load",
    });

    await page.addScriptTag({ content: mermaidSource });

    const mermaidResult = await page.evaluate(async () => {
      // eslint-disable-next-line no-undef
      mermaid.initialize({
        startOnLoad: false,
        theme: "default",
        securityLevel: "loose",
        fontFamily: "Segoe UI, Arial, sans-serif",
        flowchart: { htmlLabels: true, useMaxWidth: true },
        sequence: { useMaxWidth: true },
      });
      const nodes = Array.from(document.querySelectorAll(".mermaid"));
      let ok = 0;
      let failed = 0;
      for (const node of nodes) {
        const src = node.textContent;
        try {
          // eslint-disable-next-line no-undef
          const { svg } = await mermaid.render(
            "diagram-" + Math.random().toString(36).slice(2),
            src
          );
          node.innerHTML = svg;
          ok++;
        } catch (e) {
          failed++;
          node.innerHTML =
            '<div style="color:#B00020;border:1px solid #B00020;padding:8px;text-align:left;font-size:11px;">' +
            "Error renderizando diagrama Mermaid: " +
            String(e && e.message ? e.message : e) +
            "</div>";
        }
      }
      return { total: nodes.length, ok, failed };
    });

    if (args.verbose || mermaidResult.failed > 0) {
      console.log(
        `Diagramas Mermaid: ${mermaidResult.ok}/${mermaidResult.total} OK` +
          (mermaidResult.failed
            ? `, ${mermaidResult.failed} con error (revisar el PDF)`
            : "")
      );
    }

    await page.pdf({
      path: outputPath,
      format: "Letter",
      printBackground: true,
      displayHeaderFooter: true,
      margin: { top: "1in", bottom: "0.9in", left: "0.97in", right: "0.97in" },
      headerTemplate: `
        <div style="width:100%;font-size:6px;color:#343434;text-align:center;font-family:Arial, sans-serif;">
          ${escapeHtml(headerText)}
        </div>`,
      footerTemplate: `
        <div style="width:100%;font-size:6px;color:#343434;text-align:center;font-family:Arial, sans-serif;">
          Página <span class="pageNumber"></span> de <span class="totalPages"></span>
        </div>`,
    });

    console.log(`PDF generado: ${outputPath}`);
  } finally {
    await browser.close();
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
