"""
Núcleo de conversión del compuesto OPERA (EUMETNET) a teselas pequeñas para la app. Solo numpy (sin pyproj ni h5py):
así se puede probar sin red ni ficheros HDF5.

Entrada: matriz de dBZ del compuesto (3800 x 4400 celdas de 1 km, proyección LAEA, fila 0 = arriba), con los valores
especiales `nodata` (sin cobertura) y `undetect` (cubierto, sin eco).
Salida: teselas de 5° x 5° en rejilla lon/lat de 0,02°, 8 bits por píxel:
    0      = SIN DATO (sin cobertura; la app debe decir "sin datos", nunca "seco")
    1      = cubierto, sin eco
    2..255 = dBZ = (valor - 2) * 0.5 - 8   (de -8 a 119,5 dBZ; en la práctica se recorta a 0..70 dBZ)
Convención de cada tesela: fila 0 = norte. Ver LEEME.md.
"""
import math
import numpy as np

# Proyección del compuesto (la que da el propio fichero en "projdef"):
#   +proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84
A = 6378137.0
E2 = 0.00669437999014133  # WGS84
LAT0, LON0 = 55.0, 10.0
X0, Y0 = 1950000.0, -2100000.0
XSIZE, YSIZE, CELL = 3800, 4400, 1000.0  # esquina inferior izquierda del fichero = (0, -4.400.000) m; la superior derecha = (3.800.000, 0)

# Rejilla de salida
TILE_DEG = 5
RES_DEG = 0.02
LON_W, LAT_N = -12.0, 70.0   # esquina noroeste de la rejilla (múltiplos de TILE_DEG)
LON_E, LAT_S = 58.0, 30.0
TILE_PX = int(round(TILE_DEG / RES_DEG))  # 250

ECHO_FLOOR_DBZ = 0.0   # por debajo: "sin eco" (ruido)
ECHO_CEIL_DBZ = 70.0   # por encima: se recorta (probables interferencias o granizo extremo)
CODE_NODATA, CODE_NOECHO = 0, 1

# Nivel grueso (vista general): misma zona y mismos códigos, a 0,05° y teselas de 10° x 10° (200 x 200 px).
COARSE_TILE_DEG = 10
COARSE_RES_DEG = 0.05
COARSE_TILE_PX = int(round(COARSE_TILE_DEG / COARSE_RES_DEG))  # 200
COARSE_RATIO = COARSE_RES_DEG / RES_DEG                        # 2,5 píxeles finos por píxel grueso


def _q(sin_phi):
    e = math.sqrt(E2)
    return (1 - E2) * (sin_phi / (1 - E2 * sin_phi ** 2) - (1 / (2 * e)) * np.log((1 - e * sin_phi) / (1 + e * sin_phi)))


def laea_forward(lon_deg, lat_deg):
    """lon/lat (grados, WGS84) -> (x, y) en metros en el sistema del fichero (con X0/Y0 ya sumados). Snyder, ec. 24-xx (elipsoidal)."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon0, lat0 = math.radians(LON0), math.radians(LAT0)
    qp = float(_q(1.0))
    q0 = float(_q(math.sin(lat0)))
    q = _q(np.sin(lat))
    rq = A * math.sqrt(qp / 2)
    beta = np.arcsin(np.clip(q / qp, -1, 1))
    beta0 = math.asin(q0 / qp)
    d = A * (math.cos(lat0) / math.sqrt(1 - E2 * math.sin(lat0) ** 2)) / (rq * math.cos(beta0))
    dl = lon - lon0
    b = rq * np.sqrt(2 / (1 + math.sin(beta0) * np.sin(beta) + math.cos(beta0) * np.cos(beta) * np.cos(dl)))
    x = b * d * np.cos(beta) * np.sin(dl)
    y = (b / d) * (math.cos(beta0) * np.sin(beta) - math.sin(beta0) * np.cos(beta) * np.cos(dl))
    return x + X0, y + Y0


def source_index_for_grid(lon_w=LON_W, lat_n=LAT_N, lon_e=LON_E, lat_s=LAT_S, res=RES_DEG):
    """Para cada píxel de salida (centro), fila y columna en el compuesto; -1 si cae fuera. Devuelve (rows, cols, shape)."""
    n_lon = int(round((lon_e - lon_w) / res))
    n_lat = int(round((lat_n - lat_s) / res))
    lons = lon_w + (np.arange(n_lon) + 0.5) * res
    lats = lat_n - (np.arange(n_lat) + 0.5) * res
    lon_g, lat_g = np.meshgrid(lons, lats)
    x, y = laea_forward(lon_g, lat_g)
    col = np.floor(x / CELL).astype(np.int64)
    row = np.floor((0.0 - y) / CELL).astype(np.int64)   # fila 0 = arriba (y = 0)
    out = (col < 0) | (col >= XSIZE) | (row < 0) | (row >= YSIZE)
    row[out] = -1
    col[out] = -1
    return row, col, (n_lat, n_lon)


def encode(dbz, nodata, undetect):
    """Compuesto (YSIZE, XSIZE) float -> códigos uint8 con el filtro de ruido. Los valores no finitos cuentan como sin dato."""
    raw = np.asarray(dbz, dtype=np.float64)
    is_nodata = ~np.isfinite(raw) | (raw == nodata)
    is_undetect = (raw == undetect) & ~is_nodata
    echo = ~is_nodata & ~is_undetect & (raw >= ECHO_FLOOR_DBZ)
    code = np.full(raw.shape, CODE_NOECHO, dtype=np.uint8)
    code[is_nodata] = CODE_NODATA
    v = np.clip(np.nan_to_num(raw, nan=0.0, posinf=ECHO_CEIL_DBZ, neginf=0.0), ECHO_FLOOR_DBZ, ECHO_CEIL_DBZ)
    enc = np.clip(np.round((v + 8.0) * 2.0) + 2, 2, 255).astype(np.uint8)
    code[echo] = enc[echo]
    return despeckle(code)


def despeckle(code):
    """Quita ecos aislados de un solo píxel (sin ningún vecino con eco entre los 8 de alrededor): interferencias típicas."""
    has_echo = code >= 2
    pad = np.pad(has_echo, 1, constant_values=False)
    neigh = np.zeros(code.shape, dtype=np.int16)
    h, w = code.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            neigh += pad[dy:dy + h, dx:dx + w]
    isolated = has_echo & (neigh == 0)
    out = code.copy()
    out[isolated] = CODE_NOECHO
    return out


def resample(code, rows, cols):
    """Remuestreo por vecino más próximo del compuesto codificado a la rejilla de salida; fuera del compuesto = sin dato."""
    out = np.full(rows.shape, CODE_NODATA, dtype=np.uint8)
    ok = rows >= 0
    out[ok] = code[rows[ok], cols[ok]]
    return out


def tiles(grid, tile_px=TILE_PX):
    """Parte la rejilla de salida en teselas tile_px x tile_px. Devuelve {(fila_tesela, col_tesela): matriz}. Omite las 100 % sin dato."""
    res = {}
    for r in range(grid.shape[0] // tile_px):
        for c in range(grid.shape[1] // tile_px):
            t = grid[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px]
            if (t != CODE_NODATA).any():
                res[(r, c)] = t
    return res


def _edges(n_out, ratio):
    return np.rint(np.arange(n_out + 1) * ratio).astype(np.int64)


def coarsen(grid, ratio=COARSE_RATIO):
    """Reduce la rejilla fina a la gruesa (bloques de `ratio` x `ratio` píxeles; con 2,5 los bloques miden 2 o 3).
    Regla de cada bloque (la misma que sigue la app: nunca se inventa un "seco"):
      - si la mitad o más de sus píxeles son SIN DATO, el resultado es SIN DATO;
      - si no, el máximo del bloque (el eco más fuerte; si no hay eco, "cubierto sin eco")."""
    h, w = grid.shape
    ho, wo = int(round(h / ratio)), int(round(w / ratio))
    er, ec = _edges(ho, ratio), _edges(wo, ratio)
    if er[-1] != h or ec[-1] != w:
        raise ValueError("la rejilla fina no es múltiplo exacto de la gruesa")
    mx = np.maximum.reduceat(np.maximum.reduceat(grid, er[:-1], axis=0), ec[:-1], axis=1)
    nd = (grid == CODE_NODATA).astype(np.uint32)
    cnt = np.add.reduceat(np.add.reduceat(nd, er[:-1], axis=0), ec[:-1], axis=1)
    size = np.diff(er)[:, None] * np.diff(ec)[None, :]
    out = mx.copy()
    out[cnt * 2 >= size] = CODE_NODATA
    return out.astype(np.uint8)


def tile_name(r, c):
    """Nombre estable de tesela: fila 0 = la más al norte; lat/lon de su esquina noroeste en grados."""
    return f"{r}_{c}"


def tile_bounds(r, c):
    """(lat_norte, lon_oeste, lat_sur, lon_este) de la tesela."""
    n = LAT_N - r * TILE_DEG
    w = LON_W + c * TILE_DEG
    return n, w, n - TILE_DEG, w + TILE_DEG


def decode_dbz(code):
    """Código -> dBZ (float; NaN si sin dato o sin eco)."""
    c = np.asarray(code, dtype=np.float64)
    return np.where(c >= 2, (c - 2) * 0.5 - 8.0, np.nan)


# ---------------------------------------------------------------------------------------------------------------------
# Imagen resumen (vista de toda Europa): una sola imagen RGBA por fotograma, ya coloreada, en proyección Mercator (la que usa
# Google Maps para colocar una imagen entre dos esquinas). La app solo la baja y la escala; no calcula nada.
# Misma paleta, mismo suavizado y misma opacidad que EuMosaic.kt / RadarPalette.kt de la app.
# ---------------------------------------------------------------------------------------------------------------------
OV_WIDTH = 1200                # píxeles de ancho de la imagen resumen (todo el ancho de la rejilla, 70°)
OV_ALPHA = 0xD2
OV_EDGE_LO, OV_EDGE_HI = 11.0, 13.0
BACKGROUND_DBZ = 9.0           # dBZ que se da a "cubierto sin eco" al suavizar (RadarSmoothRenderer.BACKGROUND_DBZ)
OV_NODATA_RGBA = (0x6E, 0x6E, 0x6E, 0x44)   # RadarSmoothRenderer.NO_DATA_ARGB
OV_PALETTE = [0x0000FC, 0x0094FC, 0x00FCFC, 0x438323, 0x00C000, 0x00FF00, 0xFFFF00, 0xFFBB00, 0xFF7F00, 0xFF0000, 0xC8005A]  # desde 12 dBZ, tramos de 6


def _merc(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def _inv_merc(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def overview_size(width=OV_WIDTH):
    """(ancho, alto) de la imagen resumen: el alto sale de la proyección Mercator de la rejilla."""
    dx = math.radians(LON_E - LON_W)
    dy = float(_merc(LAT_N) - _merc(LAT_S))
    return width, int(round(width * dy / dx))


def overview_rgba(coarse, width=OV_WIDTH):
    """Rejilla gruesa de códigos (filas = latitud, fila 0 = norte) -> matriz RGBA uint8 (alto, ancho, 4) en Mercator.
    - Fuera de cobertura (SIN DATO o junto a su borde): gris translúcido, nunca transparente ("seco").
    - Cubierto sin eco o por debajo de 11 dBZ: transparente. Entre 11 y 13 dBZ: aparece poco a poco; desde 13: opacidad OV_ALPHA."""
    gh, gw = coarse.shape
    w, h = overview_size(width)
    c = coarse.astype(np.float32)
    f = np.where(c >= 2, (c - 2) * 0.5 - 8.0, np.where(c == CODE_NOECHO, BACKGROUND_DBZ, np.nan)).astype(np.float32)
    # Centro de cada píxel de salida -> posición en la rejilla gruesa (mismas fórmulas que EuGrid.smoothDbzOrNaN)
    yN, yS = _merc(LAT_N), _merc(LAT_S)
    lat = _inv_merc(yN - (np.arange(h) + 0.5) / h * (yN - yS))
    lon = LON_W + (np.arange(w) + 0.5) / w * (LON_E - LON_W)
    fy = (LAT_N - lat) / COARSE_RES_DEG - 0.5
    fx = (lon - LON_W) / COARSE_RES_DEG - 0.5
    y0 = np.floor(fy).astype(np.int64); x0 = np.floor(fx).astype(np.int64)
    ty = (fy - y0).astype(np.float32)[:, None]; tx = (fx - x0).astype(np.float32)[None, :]
    y0c, y1c = np.clip(y0, 0, gh - 1), np.clip(y0 + 1, 0, gh - 1)
    x0c, x1c = np.clip(x0, 0, gw - 1), np.clip(x0 + 1, 0, gw - 1)
    a = f[np.ix_(y0c, x0c)]; b = f[np.ix_(y0c, x1c)]; cc = f[np.ix_(y1c, x0c)]; d = f[np.ix_(y1c, x1c)]
    v = (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + d * tx) * ty     # NaN si algún vecino es SIN DATO
    out = np.zeros((h, w, 4), dtype=np.uint8)
    nod = np.isnan(v)
    out[nod] = OV_NODATA_RGBA
    ok = ~nod & (v > OV_EDGE_LO)
    idx = np.clip(np.floor((np.maximum(np.nan_to_num(v, nan=0.0), 12.0) - 12.0) / 6.0).astype(np.int64), 0, len(OV_PALETTE) - 1)
    pal = np.array([[(p >> 16) & 255, (p >> 8) & 255, p & 255] for p in OV_PALETTE], dtype=np.uint8)
    vv = np.nan_to_num(v, nan=0.0)
    alpha = np.where(vv >= OV_EDGE_HI, float(OV_ALPHA), OV_ALPHA * (vv - OV_EDGE_LO) / (OV_EDGE_HI - OV_EDGE_LO))
    alpha = np.where(ok, alpha, 0).astype(np.int64)     # int() como en Kotlin (trunca)
    px = ok & (alpha > 0)
    out[px, :3] = pal[idx[px]]
    out[px, 3] = alpha[px].astype(np.uint8)
    return out
