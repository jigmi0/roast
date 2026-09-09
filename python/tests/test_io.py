"""File formats: NIfTI, layout files and the mesh/solver formats."""

import numpy as np
import pytest

from roast.config import example_dir
from roast.io import (load_untouch, read_cap_info, read_elec_loc,
                      read_medit, read_pos, save_inr, save_msh, save_spm_mapping,
                      load_spm_mapping, srow_to_spm_mat, spm_mat_to_srow, ImageGeom)

MNI = example_dir() / "MNI152_T1_1mm.nii"


def test_load_untouch_keeps_raw_data_and_header():
    volume = load_untouch(MNI)
    assert volume.img.shape == (182, 218, 182)
    assert volume.img.dtype == np.int16          # unscaled, as stored
    assert volume.srow().shape == (4, 4)


def test_save_round_trips_data_and_orientation(tmp_path):
    volume = load_untouch(MNI)
    out = tmp_path / "copy.nii"
    volume.save(out)
    reloaded = load_untouch(out)
    assert np.array_equal(reloaded.img, volume.img)
    assert np.allclose(reloaded.srow(), volume.srow())


def test_spm_mat_conversion_is_a_one_voxel_shift():
    srow = np.array([[1.0, 0, 0, -90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
    mat = srow_to_spm_mat(srow)
    assert mat[:3, 3].tolist() == [-91.0, -127.0, -73.0]
    assert np.allclose(spm_mat_to_srow(mat), srow)


def test_geom_mean_resolution():
    geom = ImageGeom(np.diag([0.5, 1.0, 1.5, 1.0]), [2, 2, 2])
    assert geom.mean_resolution == pytest.approx(1.0)


def test_cap_info_has_no_header_row():
    cap = read_cap_info("1010")
    assert cap.names[0] == "LPA"
    assert cap.index("Fp1") == 3
    assert cap.coords.shape == (len(cap), 3)


def test_elec_loc_matches_the_cap_order():
    names = read_elec_loc()
    cap = read_cap_info("1010")
    indices = [cap.index(name) for name in names]
    assert len(names) == 72
    assert indices == sorted(indices)          # the .loc file follows capInfo's order
    assert names[-1] == "Iz"                   # the reference comes last


def test_inr_and_medit_round_trip(tmp_path):
    volume = np.zeros((4, 5, 6), dtype=np.uint8)
    volume[1:3, 1:3, 1:3] = 2
    path = save_inr(volume, tmp_path / "vol.inr")
    raw = path.read_bytes()
    assert raw[:14] == b"#INRIMAGE-4#{\n"
    assert len(raw) == 256 + volume.size

    mesh = tmp_path / "mesh.mesh"
    mesh.write_text("MeshVersionFormatted 1\nVertices\n2\n"
                    "0 0 0 1\n1 1 1 2\nTriangles\n1\n1 2 1 3\n"
                    "Tetrahedra\n1\n1 2 1 2 4\nEnd\n")
    node, elem, face = read_medit(mesh)
    assert node.shape == (2, 4) and elem.shape == (1, 5) and face.shape == (1, 4)
    assert elem[0, 4] == 4


def test_save_msh_writes_gmsh_22(tmp_path):
    node = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elem = np.array([[1, 2, 3, 4, 7]])
    text = save_msh(node, elem, tmp_path / "m.msh").read_text()
    assert text.startswith("$MeshFormat\n2.2 0 8\n")
    assert "$Nodes\n4\n" in text
    assert "1 4 2 7 7 1 2 3 4" in text


def test_read_pos_parses_node_tables(tmp_path):
    path = tmp_path / "res.pos"
    path.write_text('$PostFormat\n1 0.5\n2 1.5\n3 2.5\n$End\n')
    ids, values = read_pos(path, 1)
    assert ids.tolist() == [1, 2, 3]
    assert values.ravel().tolist() == [0.5, 1.5, 2.5]


def test_spm_mapping_round_trip(tmp_path):
    image = ImageGeom(np.diag([1.0, 1.0, 1.0, 1.0]), [10, 11, 12])
    tpm = ImageGeom(np.diag([1.5, 1.5, 1.5, 1.0]), [5, 6, 7])
    affine = np.eye(4)
    affine[0, 3] = 3.0
    save_spm_mapping(tmp_path / "map.mat", image, tpm, affine)
    back_image, back_tpm, back_affine = load_spm_mapping(tmp_path / "map.mat")
    assert back_image.dim.tolist() == [10, 11, 12]
    assert back_tpm.dim.tolist() == [5, 6, 7]
    assert np.allclose(back_affine, affine)


def test_nyhead_mapping_loads():
    image, tpm, affine = load_spm_mapping(example_dir() / "nyhead_T1orT2_seg8.mat")
    assert image.dim.tolist() == [394, 466, 620]
    assert affine.shape == (4, 4)
