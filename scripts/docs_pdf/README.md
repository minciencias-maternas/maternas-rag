# docs_pdf — generador de PDF para la documentación

Convierte un Markdown del proyecto (por defecto `docs/DOCUMENTACION_oss120b_20260831.md`) a
PDF, con el mismo estilo visual del PDF de referencia (`docs/DOCUMENTACION_oss120b_20260831.pdf`):
título/encabezados en azul, tabla de metadata como callout, tablas con
encabezado azul y bandas alternadas, código inline en rojo/gris, y
encabezado + pie de página con numeración en cada hoja. Los bloques
` ```mermaid ` se renderizan como diagramas reales (no como texto), usando
`mermaid.js` dentro de una página headless de Chrome/Edge.

No descarga un Chromium propio: usa `puppeteer-core` apuntando al
Chrome o Edge ya instalado en la máquina.

## Instalación (una sola vez)

```bash
cd scripts/docs_pdf
npm install
```

## Uso

```bash
# Regenera docs/DOCUMENTACION_oss120b_20260831.md -> docs/DOCUMENTACION_oss120b_20260831.pdf (default)
node generate.mjs

# Con otro archivo / salida / texto de encabezado
node generate.mjs --input docs/OTRO_DOC.md --output docs/OTRO_DOC.pdf --header "Maternas RAG"

# Ver qué navegador se usó y si algún diagrama Mermaid falló al renderizar
node generate.mjs --verbose
```

Las rutas de `--input`/`--output` son relativas a la raíz del repo (no al
directorio de este script).

## Si no encuentra un navegador

El script busca Chrome/Edge en las rutas de instalación estándar de
Windows/macOS/Linux. Si tu navegador vive en otro lado, definí la
variable de entorno antes de correr el script:

```bash
export PUPPETEER_EXECUTABLE_PATH="C:/ruta/a/chrome.exe"
```

## Cuándo regenerar el PDF

Después de editar `docs/DOCUMENTACION_oss120b_20260831.md` (secciones, tablas o
diagramas Mermaid), volvé a correr `node generate.mjs` para que
`docs/DOCUMENTACION_oss120b_20260831.pdf` quede sincronizado. El script no se ejecuta
solo — es una acción manual, para no acoplar la generación del PDF al
flujo normal de commits.
