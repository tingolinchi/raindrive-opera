"""Prueba del flujo de build_opera.py con red y lectura HDF5 simuladas (no prueba h5py ni la API real)."""
import datetime as dt, io, json, os, tempfile, unittest
import numpy as np
from PIL import Image
import build_opera as bo
import opera_core as oc


def fake_frame(stamp, rain=True, nodata_all=False):
    raw = np.full((oc.YSIZE, oc.XSIZE), -8888000.0)       # cubierto, sin eco
    raw[:, :400] = -9999000.0                              # una banda sin cobertura
    if nodata_all:
        raw[:] = -9999000.0
    if rain:
        # lluvia sobre Madrid
        x, y = oc.laea_forward(-3.70, 40.42)
        c, r = int(x / 1000), int(-y / 1000)
        raw[r - 5:r + 5, c - 5:c + 5] = 30.0
    t = dt.datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=dt.timezone.utc)
    return raw, -9999000.0, -8888000.0, t


class Flujo(unittest.TestCase):
    def setUp(self):
        self.stamps = ["20261114T0840", "20261114T0845", "20261114T0850", "20261114T0855"]
        self.calls = []
        self.bad = set()
        bo.latest_links = lambda now=None: [(s, "http://x/OPERA@%s@0@DBZH.h5" % s) for s in self.stamps]
        def fake_get(url, timeout=90, tries=3):
            self.calls.append(url); return url.split("@")[1].encode(), {}
        bo.http_get = fake_get
        def fake_read(data):
            stamp = data.decode()
            if stamp in self.bad:
                return fake_frame(stamp, nodata_all=True)
            return fake_frame(stamp)
        bo.read_odim = fake_read

    def test_genera_meta_y_png_y_reutiliza(self):
        with tempfile.TemporaryDirectory() as d:
            bo.main(d)
            meta = json.load(open(os.path.join(d, "meta.json")))
            self.assertEqual(len(meta["frames"]), 4)
            self.assertEqual(meta["latest"], "2026-11-14T08:55:00Z")
            fr = meta["frames"][-1]
            self.assertTrue(fr["tiles"])
            tile = np.array(Image.open(os.path.join(d, "f", fr["stamp"], fr["tiles"][0] + ".png")))
            self.assertEqual(tile.shape, (250, 250))
            # Madrid: debe haber algún píxel con eco en las teselas
            self.assertTrue(any((np.array(Image.open(os.path.join(d, "f", fr["stamp"], t + ".png"))) >= 2).any() for t in fr["tiles"]))
            n = len(self.calls)
            self.stamps.append("20261114T0900")
            bo.main(d)
            self.assertEqual(len(self.calls), n + 1)                     # solo se descarga el nuevo
            self.assertEqual(len(json.load(open(os.path.join(d, "meta.json")))["frames"]), 5)

    def test_borra_los_que_salen_de_la_ventana(self):
        with tempfile.TemporaryDirectory() as d:
            self.stamps = ["20261114T%02d%02d" % (8, m) for m in range(0, 60, 5)] + ["20261114T0900"]  # 13
            bo.main(d)
            self.assertEqual(len(os.listdir(os.path.join(d, "f"))), 12)

    def test_fotograma_vacio_se_descarta_y_no_se_publica_si_no_queda_ninguno(self):
        with tempfile.TemporaryDirectory() as d:
            self.bad = set(self.stamps)
            with self.assertRaises(SystemExit):
                bo.main(d)
            self.assertFalse(os.path.exists(os.path.join(d, "meta.json")))

    def test_un_fotograma_malo_no_tumba_a_los_demas(self):
        with tempfile.TemporaryDirectory() as d:
            self.bad = {"20261114T0845"}
            bo.main(d)
            meta = json.load(open(os.path.join(d, "meta.json")))
            self.assertEqual([f["stamp"] for f in meta["frames"]], ["20261114T0840", "20261114T0850", "20261114T0855"])
            self.assertEqual(len(meta["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
