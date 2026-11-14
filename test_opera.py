import unittest
import numpy as np
import opera_core as oc


class Geometria(unittest.TestCase):
    def test_esquinas_medidas_con_pyproj(self):
        # Valores medidos por el usuario el 14/11/2026 con pyproj sobre un fichero real (esquinas LL y UR del compuesto).
        x, y = oc.laea_forward(-10.4345768386404, 31.7462153182675)
        self.assertAlmostEqual(float(x), 0.0, delta=1.0)
        self.assertAlmostEqual(float(y), -4400000.0, delta=1.0)
        x, y = oc.laea_forward(57.8119647501499, 67.6210371071631)
        self.assertAlmostEqual(float(x), 3800000.0, delta=1.0)
        self.assertAlmostEqual(float(y), 0.0, delta=1.0)

    def test_centro_de_proyeccion(self):
        x, y = oc.laea_forward(10.0, 55.0)
        self.assertAlmostEqual(float(x), oc.X0, delta=1e-6)
        self.assertAlmostEqual(float(y), oc.Y0, delta=1e-6)

    def test_ciudades_caen_dentro_del_compuesto_salvo_canarias(self):
        ciudades = {"Murcia": (37.98, -1.13), "Madrid": (40.42, -3.70), "Lisboa": (38.72, -9.14), "Helsinki": (60.17, 24.94)}
        for nombre, (lat, lon) in ciudades.items():
            x, y = oc.laea_forward(lon, lat)
            self.assertTrue(0 <= x < oc.XSIZE * oc.CELL and -oc.YSIZE * oc.CELL <= y < 0, nombre)
        x, y = oc.laea_forward(-15.43, 28.12)  # Las Palmas: fuera, como se midió
        self.assertFalse(0 <= x < oc.XSIZE * oc.CELL and -oc.YSIZE * oc.CELL <= y < 0)

    def test_indice_de_origen(self):
        rows, cols, shape = oc.source_index_for_grid()
        self.assertEqual(shape, (2000, 3500))
        i = int((oc.LAT_N - 40.42) / oc.RES_DEG)      # Madrid
        j = int((-3.70 - oc.LON_W) / oc.RES_DEG)
        self.assertGreaterEqual(rows[i, j], 0)
        x, y = oc.laea_forward(-3.70, 40.42)
        self.assertLess(abs(cols[i, j] - x / 1000), 3)  # el píxel de salida es de 0,02° (~2 km): tolerancia de 3 celdas
        self.assertLess(abs(rows[i, j] - (-y / 1000)), 3)
        k = int((oc.LAT_N - 28.12) / oc.RES_DEG) if 28.12 > oc.LAT_S else None
        self.assertIsNone(k)  # Canarias (lat 28) queda por debajo de la rejilla de salida (hasta lat 30)
        i2 = int((oc.LAT_N - 30.5) / oc.RES_DEG); j2 = int((55.0 - oc.LON_W) / oc.RES_DEG)
        self.assertEqual(rows[i2, j2], -1)  # lat 30,5 / lon 55: fuera del compuesto (esquina sudeste)


class Codificacion(unittest.TestCase):
    ND, UD = -9999000.0, -8888000.0

    def test_codigos_especiales(self):
        raw = np.array([[self.ND, self.UD, 20.0, 20.0], [20.0, 20.0, -5.0, 90.0]])
        code = oc.encode(raw, self.ND, self.UD)
        self.assertEqual(code[0, 0], oc.CODE_NODATA)
        self.assertEqual(code[0, 1], oc.CODE_NOECHO)

    def test_ida_y_vuelta_dbz(self):
        raw = np.full((5, 5), 25.0)  # bloque con eco (no aislado)
        code = oc.encode(raw, self.ND, self.UD)
        d = oc.decode_dbz(code)
        self.assertAlmostEqual(float(d[2, 2]), 25.0, delta=0.5)

    def test_recorte_y_ruido(self):
        raw = np.full((5, 5), 95.0)
        self.assertAlmostEqual(float(oc.decode_dbz(oc.encode(raw, self.ND, self.UD))[2, 2]), oc.ECHO_CEIL_DBZ, delta=0.5)
        raw = np.full((5, 5), -5.0)  # por debajo del suelo: sin eco, no sin dato
        self.assertTrue((oc.encode(raw, self.ND, self.UD) == oc.CODE_NOECHO).all())

    def test_eco_aislado_se_quita_pero_no_el_agrupado(self):
        raw = np.full((9, 9), self.UD)
        raw[1, 1] = 40.0                 # un solo píxel
        raw[5:7, 5:7] = 30.0             # grupo de 4
        code = oc.encode(raw, self.ND, self.UD)
        self.assertEqual(code[1, 1], oc.CODE_NOECHO)
        self.assertGreaterEqual(int(code[5, 5]), 2)

    def test_nodata_nunca_se_vuelve_sin_eco(self):
        raw = np.full((4, 4), self.ND)
        raw[1:3, 1:3] = 20.0
        code = oc.encode(raw, self.ND, self.UD)
        self.assertEqual(int(code[0, 0]), oc.CODE_NODATA)

    def test_valores_no_finitos_son_sin_dato(self):
        raw = np.array([[np.nan, np.inf], [20.0, 20.0]])
        code = oc.encode(raw, self.ND, self.UD)
        self.assertEqual(int(code[0, 0]), oc.CODE_NODATA)
        self.assertEqual(int(code[0, 1]), oc.CODE_NODATA)


class Teselas(unittest.TestCase):
    def test_remuestreo_fuera_es_sin_dato(self):
        code = np.full((oc.YSIZE, oc.XSIZE), oc.CODE_NOECHO, dtype=np.uint8)
        rows, cols, _ = oc.source_index_for_grid()
        g = oc.resample(code, rows, cols)
        i2 = int((oc.LAT_N - 30.5) / oc.RES_DEG); j2 = int((55.0 - oc.LON_W) / oc.RES_DEG)
        self.assertEqual(int(g[i2, j2]), oc.CODE_NODATA)   # fuera del compuesto
        i = int((oc.LAT_N - 40.42) / oc.RES_DEG); j = int((-3.70 - oc.LON_W) / oc.RES_DEG)
        self.assertEqual(int(g[i, j]), oc.CODE_NOECHO)   # Madrid: cubierto

    def test_teselas_omiten_las_vacias_y_conservan_limites(self):
        g = np.zeros((2000, 3500), dtype=np.uint8)
        g[1000, 1000] = 40
        t = oc.tiles(g)
        self.assertEqual(list(t.keys()), [(4, 4)])
        self.assertEqual(oc.tile_bounds(4, 4), (50.0, -12.0 + 20.0, 45.0, -12.0 + 25.0))
        self.assertEqual(t[(4, 4)].shape, (250, 250))


if __name__ == "__main__":
    unittest.main()
