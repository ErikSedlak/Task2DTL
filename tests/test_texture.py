"""Textured mesh loading and texture rendering (DESIGN §6.1 / §6.2, M6)."""

import numpy as np
from PIL import Image

from lsg.geom.render import TextureRenderer, sample_bilinear
from lsg.geom.transforms import invert
from lsg.io.scan3r import TexturedMesh, load_textured_mesh, read_obj
from synth import H, K, W, look_at

RED, GREEN, BLUE, WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)


def quadrant_texture(n=64):
    """Image quadrants: top-left red, top-right green, bottom-left blue, bottom-right white."""
    t = np.zeros((n, n, 3), np.uint8)
    h = n // 2
    t[:h, :h], t[:h, h:], t[h:, :h], t[h:, h:] = RED, GREEN, BLUE, WHITE
    return t


def facing_quad():
    """Quad in the plane y = 2 (facing a camera at the origin looking along +y), z up.

    Corner uv: bottom-left (0, 0), bottom-right (1, 0), top-right (1, 1), top-left (0, 1).
    """
    V = np.array([[-1, 2, -1], [1, 2, -1], [1, 2, 1], [-1, 2, 1]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return TexturedMesh("q", V, F, uv[F], quadrant_texture())


def test_read_obj_per_vertex_and_per_corner_uv(tmp_path):
    p = tmp_path / "m.obj"
    p.write_text(
        "mtllib x.mtl\nv 0 0 0\nvn 0 0 1\nvt 0.1 0.2\nv 1 0 0\nvt 0.3 0.4\nv 0 1 0\nvt 0.5 0.6\n"
        "v 1 1 0\nvt 0.7 0.8\nusemtl m\nf 1/1/1 2/2/2 3/3/3\nf 2/4 4/1 3//1\n",
        encoding="ascii",
    )
    V, F, UV = read_obj(p)
    assert V.shape == (4, 3) and F.tolist() == [[0, 1, 2], [1, 3, 2]]
    np.testing.assert_allclose(UV[0], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_allclose(UV[1], [[0.7, 0.8], [0.1, 0.2], [0.0, 0.0]])  # a//c corner: no uv


def test_load_textured_mesh(tmp_path):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "mesh.refined.v2.obj").write_text("v 0 0 0\nvt 0 0\nv 1 0 0\nvt 1 0\nv 0 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    Image.fromarray(quadrant_texture(8)).save(d / "mesh.refined_0.png")
    tm = load_textured_mesh(tmp_path, "ref")
    assert tm.texture.shape == (8, 8, 3) and tm.faces.shape == (1, 3) and tm.corner_uv.shape == (1, 3, 2)


def test_bilinear_sampling_flips_v():
    t = quadrant_texture()
    s = sample_bilinear(t, np.array([[0.1, 0.9], [0.9, 0.9], [0.1, 0.1], [0.9, 0.1], [0.5, 0.25]], np.float32))
    assert s[:4].tolist() == [list(RED), list(GREEN), list(BLUE), list(WHITE)]
    # halfway between blue and white columns -> mixed colour
    assert 0 < s[4, 0] < 255 and s[4, 2] == 255


def test_texture_render_orientation_and_misses():
    tr = TextureRenderer(facing_quad())
    pose = look_at([0, 0, 0], [0, 2, 0])  # camera x = world +x, camera y = world -z
    rgb, hit = tr.render_color(K, invert(pose), W, H)
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

    def px(x, z):  # world point on the quad (y = 2) -> pixel (row, col)
        return int(round(cy - z / 2 * fy)), int(round(cx + x / 2 * fx))

    assert tuple(rgb[px(-0.5, 0.5)]) == RED  # upper-left of the quad: uv (0, 1) side
    assert tuple(rgb[px(0.5, 0.5)]) == GREEN
    assert tuple(rgb[px(-0.5, -0.5)]) == BLUE
    assert tuple(rgb[px(0.5, -0.5)]) == WHITE
    # the quad (half-width 1 m at 2 m) does not fill the image: border columns miss
    assert not hit[:, 0].any() and (rgb[~hit] == 0).all() and hit[px(0, 0)]
