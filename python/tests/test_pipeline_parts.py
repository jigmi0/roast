"""Segmentation helpers, the solver front-end and the path bookkeeping."""

from pathlib import Path

import numpy as np
import pytest

from roast.config import NUM_OF_TISSUE
from roast.mesh.iso2mesh import sort_mesh
from roast.pipeline.log import find_previous_run, load_options, write_roast_log
from roast.pipeline.review import _currents_from_config
from roast.pipeline.simulate import model_paths
from roast.segment.masks import binary_mask_generate, brain_crop
from roast.solver.getdp import write_pro_file
from roast.solver.post import interpolate_to_grid
from roast.solver.prepare import free_boundary
from tests.test_options import _options


def test_binary_mask_generate_picks_the_winning_tissue():
    a = np.zeros((2, 2, 2)); a[0, 0, 0] = 1.0
    b = np.zeros((2, 2, 2)); b[1, 1, 1] = 2.0
    empty, first, second = binary_mask_generate(a, b)
    assert first[0, 0, 0] and second[1, 1, 1]
    assert empty.sum() == 6                       # everything else is unassigned


def test_brain_crop_bounds_the_white_matter():
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    mask[5:20, 6:21, 7:22] = 1
    bbox = brain_crop(mask)
    assert bbox[0].tolist() == [6, 7, 8]          # 1-based, inclusive
    assert bbox[1].tolist() == [20, 21, 22]


def test_free_boundary_drops_shared_faces():
    tets = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    faces = free_boundary(tets)
    assert len(faces) == 6                        # 8 faces, one shared pair removed
    assert not any(sorted(face) == [2, 3, 4] for face in faces.tolist())


def test_sort_mesh_keeps_the_mesh_consistent():
    node = np.array([[0.0, 0, 0, 1], [5, 0, 0, 1], [0, 5, 0, 1], [0, 0, 5, 1]])
    elem = np.array([[1, 2, 3, 4, 1]])
    face = np.array([[1, 2, 3, 1]])
    sorted_node, sorted_elem, sorted_face = sort_mesh(node, elem, face)
    # the same tetrahedron, re-indexed onto the sorted node list
    assert sorted(sorted_elem[0, :4]) == [1, 2, 3, 4]
    original = {tuple(row[:3]) for row in node}
    assert {tuple(row[:3]) for row in sorted_node} == original
    assert sorted_face.shape == face.shape


def test_interpolate_to_grid_is_linear_inside_the_hull():
    points = np.array([[1.0, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 3, 3]])
    values = np.array([0.0, 2, 2, 2, 6])
    grid = interpolate_to_grid(points, values, [3, 3, 3])
    assert grid[0, 0, 0] == pytest.approx(0.0)
    assert grid[2, 2, 2] == pytest.approx(6.0)
    assert np.isnan(grid).sum() < grid.size


def test_write_pro_file_declares_every_region(tmp_path):
    sigma = dict(white=0.126, gray=0.276, csf=1.65, bone=0.01, skin=0.465,
                 air=2.5e-14, gel=[0.3, 0.3], electrode=[5.9e7, 5.9e7])
    text = write_pro_file(tmp_path / "p.pro", "subj", "tag", np.array([1.0, -1.0]),
                          sigma, [0, 1], 2, np.array([50.0, 50.0])).read_text()
    assert f"gel1 = Region[{NUM_OF_TISSUE + 1}];" in text
    assert f"elec1 = Region[{NUM_OF_TISSUE + 2 + 1}];" in text
    assert f"usedElec1 = Region[{NUM_OF_TISSUE + 4 + 1}];" in text
    assert "sigma[air] = 2.5e-14;" in text
    assert "du_dn1[] = 20;" in text               # 1000 * 1 mA / 50 mm^2
    assert "du_dn2[] = -20;" in text
    assert 'File "subj_tag_v.pos"' in text


def test_write_pro_file_omits_the_voltage_for_a_lead_field(tmp_path):
    sigma = dict(white=1, gray=1, csf=1, bone=1, skin=1, air=1, gel=[1, 1],
                 electrode=[1, 1])
    text = write_pro_file(tmp_path / "p.pro", "subj", "tag", np.array([1.0, -1.0]),
                          sigma, [0, 1], 2, np.array([1.0, 1.0]), "3").read_text()
    assert "_v.pos" not in text
    assert 'File "subj_tag_e3.pos"' in text


def test_model_paths_follow_the_matlab_naming():
    paths = model_paths("a/subj.nii", "a/subj.nii", None, multiaxial=False)
    assert paths.spm.name == "subj_T1orT2.nii"
    assert paths.masks.name == "subj_T1orT2_SPM_masks.nii"
    assert paths.mapping.name == "subj_T1orT2_seg8.mat"

    paths = model_paths("a/subj.nii", "a/subj_1mm.nii", "a/t2.nii", multiaxial=True)
    assert paths.spm.name == "subj_1mm_T1andT2.nii"
    assert paths.masks.name == "subj_1mm_multiaxial_masks.nii"
    assert paths.mapping.name == "subj_1mm_niftyReg.mat"


def test_currents_are_recovered_from_the_recipe_text():
    currents = _currents_from_config("Fp1 (1 mA), P4 (-1 mA), C3 (0.5 mA)")
    assert currents.tolist() == [1.0, -1.0, 0.5]


def test_log_assigns_a_tag_and_finds_the_previous_run(tmp_path):
    subj = tmp_path / "subj.nii"
    options = _options(unique_tag=None)
    options = write_roast_log(subj, options)
    assert options.unique_tag
    assert (tmp_path / "subj_roastLog").exists()

    again = _options(unique_tag=None)
    assert find_previous_run(subj, again, "roast") == options.unique_tag
    assert find_previous_run(subj, _options(unique_tag=None, zero_pad=10),
                             "roast") is None

    reloaded = load_options(tmp_path / f"subj_{options.unique_tag}_roastOptions.json",
                            "roast")
    assert reloaded.config_txt == options.config_txt
    assert reloaded.elec_para[0].elec_size.tolist() == [[6.0, 2.0]]


def test_interpolate_to_grid_handles_several_columns_at_once():
    points = np.array([[1.0, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 3, 3]])
    values = np.column_stack([[0.0, 2, 2, 2, 6], [1.0, 1, 1, 1, 1]])
    grid = interpolate_to_grid(points, values, [3, 3, 3])
    assert grid.shape == (3, 3, 3, 2)
    assert grid[0, 0, 0, 0] == pytest.approx(0.0)
    assert grid[2, 2, 2, 0] == pytest.approx(6.0)
    finite = grid[..., 1][~np.isnan(grid[..., 1])]
    assert np.allclose(finite, 1.0)


def test_getdp_is_run_inside_the_subject_folder_with_absolute_paths(tmp_path, monkeypatch):
    from roast.solver.getdp import getdp_command

    (tmp_path / "example").mkdir()
    monkeypatch.chdir(tmp_path)
    command, workdir = getdp_command("example/subject1.nii", "tag")
    assert workdir == (tmp_path / "example").resolve()
    assert command[1] == str(workdir / "subject1_tag.pro")
    assert command[command.index("-msh") + 1] == str(workdir / "subject1_tag_ready.msh")
    assert all(Path(item).is_absolute() for item in (command[0], command[1]))


def test_multiaxial_setup_runs_from_the_repository_root(tmp_path, monkeypatch):
    import subprocess
    from roast.segment import multiaxial

    base = tmp_path / "lib" / "multiaxial"
    base.mkdir(parents=True)
    (base / "setupLinux.sh").write_text("#!/bin/sh\n")
    monkeypatch.setattr(multiaxial, "lib_dir", lambda: tmp_path / "lib")
    monkeypatch.setattr(multiaxial, "arch", lambda: "glnxa64")
    monkeypatch.setattr(multiaxial, "roast_root", lambda: tmp_path)
    calls = []

    def fake_run(command, cwd=None, **kwargs):
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(multiaxial.subprocess, "run", fake_run)
    interpreter = multiaxial.multiaxial_python()
    assert calls[0][1] == str(tmp_path)                 # not lib/multiaxial
    assert interpreter == base / "multiaxialEnvLinux" / "bin" / "python3"
