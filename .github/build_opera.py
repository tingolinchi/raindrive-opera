#!/usr/bin/env python3
"""
Genera las teselas de lluvia de Europa a partir de los compuestos OPERA (EUMETNET, CC BY 4.0).

Cada ejecución:
  1. Pregunta a la API de MeteoGate (sin clave) los ficheros ODIM de la última hora y elige los 12 últimos (uno cada 5 min).
  2. Descarga solo los que aún no están en el directorio de salida (que se conserva entre ejecuciones con la caché de Actions).
  3. Convierte cada uno a teselas PNG de 8 bits (ver opera_core.py) en out/f/<AAAAMMDDTHHMM>/<fila>_<col>.png.
  4. Borra los fotogramas que ya no están entre los 12 últimos y escribe out/meta.json.
Si no hay ningún fotograma válido, sale con error y NO publica (queda lo anterior, que la app tratará como caducado).

Uso:  python3 build_opera.py out
Requisitos: pip install -r requirements.txt   (h5py numpy pillow)
Sin probar contra el servicio real al escribirlo: el núcleo numérico sí tiene pruebas (test_opera.py); la lectura HDF5 y la red no.
"""
import datetime as dt, io, json, os, re, shutil, sys, time, urllib.request
import numpy as np
from PIL import Image
import opera_core as oc

API = "https://api.meteogate.eu/eu-eumetnet-weather-radar/collections/observations/locations/0-20010-0-OPERA"
FRAMES = 12
STEP_MIN = 5
WINDOW_MIN = 75
NAME_RE = re.compile(r"OPERA@(\d{8}T\d{4})@0@DBZH\.h5$")
UA = {"User-Agent": "raindrive-opera-feed/1.0 (+https://github.com/tingolinchi)"}


def http_get(url, timeout=90, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.headers
        except Exception as e:  # red, 5xx, tiempo agotado
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"fallo al pedir {url}: {last}")


def latest_links(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    ini = (now - dt.timedelta(minutes=WINDOW_MIN)).strftime("%Y-%m-%dT%H:%MZ")
    fin = now.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{API}?datetime={ini}/{fin}&f=CoverageJSON&standard_name=DBZH&format=ODIM"
    body, headers = http_get(url)
    print("API: límite restante", headers.get("X-RateLimit-Remaining"), "/", headers.get("X-RateLimit-Limit"))
    out = {}
    for l in json.loads(body).get("links", []):
        if l.get("type") == "application/x-odim":
            m = NAME_RE.search(l.get("href", ""))
            if m:
                out[m.group(1)] = l["href"]
    stamps = sorted(out)[-FRAMES:]
    return [(s, out[s]) for s in stamps]


def attr(g, name):
    v = g.attrs.get(name)
    return v.decode() if isinstance(v, bytes) else v


def read_odim(data):
    """Bytes de un compuesto ODIM HDF5 -> (matriz dBZ float64 con nodata/undetect convertidos, nodata, undetect, hora UTC).
    Falla (ValueError) si la geometría no es la esperada: antes pintar nada que equivocarse de sitio."""
    import h5py
    with h5py.File(io.BytesIO(data), "r") as f:
        w, wh = f["where"], f["what"]
        projdef = str(attr(w, "projdef"))
        for needle in ("+proj=laea", "+lat_0=55", "+lon_0=10", "+x_0=1950000", "+y_0=-2100000"):
            if needle not in projdef:
                raise ValueError(f"proyección inesperada: {projdef}")
        if (int(w.attrs["xsize"]), int(w.attrs["ysize"])) != (oc.XSIZE, oc.YSIZE) or float(w.attrs["xscale"]) != oc.CELL or float(w.attrs["yscale"]) != oc.CELL:
            raise ValueError("tamaño o celda inesperados")
        x, y = oc.laea_forward(float(w.attrs["LL_lon"]), float(w.attrs["LL_lat"]))
        if abs(float(x)) > 100 or abs(float(y) + oc.YSIZE * oc.CELL) > 100:
            raise ValueError("esquina inferior izquierda no coincide con la geometría esperada")
        d = f["dataset1/data1"]
        dw = d["what"]
        gain, off = float(dw.attrs["gain"]), float(dw.attrs["offset"])
        if str(attr(dw, "quantity")) != "DBZH":
            raise ValueError("el fichero no es DBZH")
        raw = d["data"][()].astype(np.float64)
        nodata, undetect = float(dw.attrs["nodata"]), float(dw.attrs["undetect"])
        t = dt.datetime.strptime(str(attr(wh, "date")) + str(attr(wh, "time")), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    if raw.shape != (oc.YSIZE, oc.XSIZE):
        raise ValueError(f"matriz de forma inesperada {raw.shape}")
    # nodata/undetect se comparan sobre el valor crudo y se pasan ya escalados junto al resto
    dbz = raw * gain + off
    return dbz, nodata * gain + off, undetect * gain + off, t


def png_bytes(tile):
    buf = io.BytesIO()
    Image.fromarray(tile, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_frame(out_dir, stamp, url, rows, cols):
    data, _ = http_get(url, timeout=180)
    dbz, nodata, undetect, t = read_odim(data)
    if t.strftime("%Y%m%dT%H%M") != stamp:
        print(f"  aviso: la hora del fichero ({t:%Y%m%dT%H%M}) no coincide con su nombre ({stamp}); se usa la del fichero")
        stamp = t.strftime("%Y%m%dT%H%M")
    code = oc.encode(dbz, nodata, undetect)
    nodata_pct = 100.0 * float((code == oc.CODE_NODATA).mean())
    echo_pct = 100.0 * float((code >= 2).mean())
    if nodata_pct > 95.0:
        raise ValueError(f"fotograma casi vacío ({nodata_pct:.1f} % sin dato): no se publica")
    grid = oc.resample(code, rows, cols)
    tl = oc.tiles(grid)
    tl2 = oc.tiles(oc.coarsen(grid), oc.COARSE_TILE_PX)
    # f/ = teselas finas, g/ = teselas gruesas (vista general). Se escriben en carpetas temporales y se renombran al final.
    pending = []
    for level, tiles_ in (("f", tl), ("g", tl2)):
        tmp = os.path.join(out_dir, level, stamp + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for (r, c), tile in tiles_.items():
            with open(os.path.join(tmp, oc.tile_name(r, c) + ".png"), "wb") as fh:
                fh.write(png_bytes(tile))
        pending.append((tmp, os.path.join(out_dir, level, stamp)))
    for tmp, final in pending:
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)
    return {"t": t.strftime("%Y-%m-%dT%H:%M:00Z"), "stamp": stamp, "tiles": sorted(oc.tile_name(r, c) for (r, c) in tl),
            "tiles2": sorted(oc.tile_name(r, c) for (r, c) in tl2),
            "nodata_pct": round(nodata_pct, 1), "echo_pct": round(echo_pct, 2)}


def main(out_dir):
    os.makedirs(os.path.join(out_dir, "f"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "g"), exist_ok=True)
    links = latest_links()
    if not links:
        sys.exit("La API no devolvió ningún compuesto reciente: no se publica nada.")
    links = sorted(links)[-FRAMES:]
    # Fotogramas ya generados (de una ejecución anterior): se reutilizan.
    prev = {}
    try:
        for fr in json.load(open(os.path.join(out_dir, "meta.json")))["frames"]:
            # Solo se reutilizan los fotogramas completos (teselas finas y gruesas); los de la versión anterior se rehacen.
            if "tiles2" in fr and all(os.path.isdir(os.path.join(out_dir, lv, fr["stamp"])) for lv in ("f", "g")):
                prev[fr["stamp"]] = fr
    except (OSError, ValueError, KeyError):
        pass
    rows = cols = None
    frames, errors = [], []
    for stamp, url in links:
        if stamp in prev:
            frames.append(prev[stamp]); continue
        try:
            if rows is None:
                rows, cols, _ = oc.source_index_for_grid()
            print("descargando", stamp)
            frames.append(build_frame(out_dir, stamp, url, rows, cols))
        except Exception as e:
            errors.append(f"{stamp}: {e}")
            print("  FALLO", stamp, e)
    keep = {fr["stamp"] for fr in frames}
    for level in ("f", "g"):
        for name in os.listdir(os.path.join(out_dir, level)):
            if name not in keep:
                shutil.rmtree(os.path.join(out_dir, level, name), ignore_errors=True)
    if not frames:
        sys.exit("Ningún fotograma válido: no se publica. Errores: " + "; ".join(errors))
    frames.sort(key=lambda f: f["stamp"])
    meta = {
        "version": 1,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest": frames[-1]["t"],
        "step_minutes": STEP_MIN,
        "grid": {"lon_w": oc.LON_W, "lat_n": oc.LAT_N, "lon_e": oc.LON_E, "lat_s": oc.LAT_S,
                 "res_deg": oc.RES_DEG, "tile_deg": oc.TILE_DEG, "tile_px": oc.TILE_PX},
        "grid2": {"lon_w": oc.LON_W, "lat_n": oc.LAT_N, "lon_e": oc.LON_E, "lat_s": oc.LAT_S,
                  "res_deg": oc.COARSE_RES_DEG, "tile_deg": oc.COARSE_TILE_DEG, "tile_px": oc.COARSE_TILE_PX, "path": "g"},
        "encoding": {"nodata": oc.CODE_NODATA, "no_echo": oc.CODE_NOECHO, "dbz_formula": "(v-2)*0.5-8", "min_dbz": oc.ECHO_FLOOR_DBZ, "max_dbz": oc.ECHO_CEIL_DBZ},
        "frames": frames,
        "errors": errors,
        "source": "OPERA / EUMETNET, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "note": "Una tesela ausente en un fotograma = fuera de la cobertura (sin datos). No significa 'sin lluvia'.",
    }
    tmp = os.path.join(out_dir, "meta.json.tmp")
    json.dump(meta, open(tmp, "w"), separators=(",", ":"))
    os.replace(tmp, os.path.join(out_dir, "meta.json"))
    print(f"OK: {len(frames)} fotogramas, último {frames[-1]['t']}, errores: {len(errors)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
