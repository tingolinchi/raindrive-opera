# Servidor de lluvia de Europa (OPERA) para RainDrive Radar EU

Qué hace: cada 5 minutos baja los compuestos de radar europeos de EUMETNET/OPERA (licencia CC BY 4.0, API de MeteoGate sin clave),
los convierte en teselas PNG pequeñas y publica los **últimos 12 fotogramas (1 hora)** junto a un `meta.json`. La app descargará
solo las teselas que necesite (no hay que bajar 2 MB de HDF5 en el móvil).

Estado: el núcleo numérico y el flujo tienen pruebas (`python3 -m unittest test_opera test_build_opera`, 16 pruebas, simulando la red y
la lectura HDF5). NO está probado: la lectura real del HDF5 (`h5py`), la API real desde GitHub, ni la publicación. La geometría se
comprobó contra las esquinas de un fichero real medidas el 14/11/2026 (error < 1 m).

## Qué se publica
- `meta.json`: `generated` (UTC), `latest` (hora del último fotograma), `grid` (esquina noroeste, tamaño de píxel 0,02°, teselas de 5°
  = 250 × 250 px), `encoding`, `frames` (lista con la hora `t`, la `stamp`, las teselas presentes y el % sin dato), `errors` y `source`.
- `f/<AAAAMMDDTHHMM>/<fila>_<col>.png`: tesela en gris de 8 bits, **fila 0 = norte**. La esquina noroeste de la tesela (f, c) es
  lat = 70 − 5·f, lon = −12 + 5·c. Rejilla: lon −12 a 58, lat 30 a 70.
- Código de cada píxel: `0` = **SIN DATO** (sin cobertura: la app dirá "sin datos", nunca "seco"); `1` = cubierto y sin eco;
  `2..255` = dBZ = (v − 2) · 0,5 − 8 (en la práctica de 0 a 70 dBZ).
- **Una tesela que falta en un fotograma = fuera de cobertura (sin datos), no "sin lluvia".** Un fotograma que falla se omite y se anota
  en `errors`; si ninguno sale bien, no se publica nada y el último queda (la app lo tratará como caducado pasados ~30 min).
- Filtro: ecos < 0 dBZ = sin eco; > 70 dBZ se recortan; los ecos de un solo píxel sin vecinos se quitan (interferencias).
  El máximo de 83,5 dBZ visto en la muestra del 14/11 era probablemente ruido, sin verificar.

## Puesta en marcha (prototipo, GitHub Pages)
1. Crea un repositorio PÚBLICO `raindrive-opera` y sube el contenido de esta carpeta (`opera-feed/`) a su raíz, incluida `.github`.
2. Settings → Pages → Build and deployment → Source: **GitHub Actions**.
3. Actions → "Lluvia de Europa (OPERA)" → Run workflow. El registro debe decir `OK: 12 fotogramas, último …`.
4. Abre `https://<usuario>.github.io/raindrive-opera/meta.json`.
5. Disparador cada 5 min: carpeta `opera-trigger/` (igual que `traffic-trigger/`: token fine-grained con Actions: Read and write solo para
   `raindrive-opera`, `npx wrangler secret put GITHUB_TOKEN`, `npx wrangler deploy`). El cron del workflow (cada 15 min) queda de respaldo.

## Antes de publicar la app con anuncios
- **GitHub Pages no vale para producción con publicidad:** sus condiciones dicen que no está pensado para sitios de negocio o SaaS
  comercial, y tiene un límite blando de 100 GB/mes de transferencia
  ([límites de GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)). Para el prototipo y las
  pruebas está bien; para el lanzamiento hay que mover la publicación a Cloudflare (R2 + Worker) u otro alojamiento. `build_opera.py` solo
  escribe una carpeta `out/`: cambiar el destino es cambiar el último paso del workflow.
- Los 10 builds/hora de Pages no aplican a los despliegues con Actions, según esa misma página.
- Mira la latencia real en el log y en `meta.json`: en la única medida (14/11) el compuesto estaba disponible ≤ 4,9 min después de su hora.
- Atribución obligatoria en la app: "Datos de radar: OPERA / EUMETNET (CC BY 4.0)".
- Coste de la caché de Actions: cada ejecución guarda una entrada (~decenas de MB); GitHub borra las antiguas por orden de uso.
