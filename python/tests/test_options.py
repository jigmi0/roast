"""Option parsing, validation and run identity."""

import numpy as np
import pytest

from roast.electrodes import ElecPara, elec_preproc
from roast.pipeline.options import (RoastOptions, TargetOptions, build_elec_para,
                                    is_new_options, parse_recipe,
                                    validate_conductivities, validate_mesh_options)


def _options(**overrides):
    defaults = dict(
        config_txt="Fp1 (1 mA), P4 (-1 mA)", elec_para=build_elec_para(["Fp1", "P4"]),
        subj_ras_rspd="subj.nii", t2=None, multiaxial=False, affine=np.eye(4),
        mri2mni=np.eye(4), landmarks=np.zeros((16, 3)),
        mesh_opt=dict(radbound=5, angbound=30, distbound=0.3, reratio=3, maxvol=10),
        conductivities=validate_conductivities(None, 2), unique_tag="t",
        resamp=False, zero_pad=0, is_non_ras=False)
    defaults.update(overrides)
    return RoastOptions(**defaults)


def test_parse_recipe_defaults_to_fp1_p4():
    names, currents, is_lead_field = parse_recipe(None)
    assert names == ["Fp1", "P4"]
    assert currents.tolist() == [1.0, -1.0]
    assert not is_lead_field


def test_parse_recipe_rejects_unbalanced_currents():
    with pytest.raises(ValueError, match="not balanced"):
        parse_recipe(["Fp1", 1, "P4", -0.5])


def test_parse_recipe_recognises_the_lead_field_keyword():
    assert parse_recipe("leadField")[2] is True


def test_default_sizes_follow_the_electrode_type():
    assert build_elec_para(["a"], elec_type="disc")[0].elec_size.tolist() == [[6, 2]]
    assert build_elec_para(["a"], elec_type="pad")[0].elec_size.tolist() == [[50, 30, 3]]
    assert build_elec_para(["a"], elec_type="ring")[0].elec_size.tolist() == [[4, 6, 2]]


def test_pads_default_to_a_left_right_orientation():
    assert build_elec_para(["a"], elec_type="pad")[0].elec_ori == "lr"
    assert build_elec_para(["a"], elec_type="disc")[0].elec_ori is None


@pytest.mark.parametrize("elec_type, size, message", [
    ("disc", [6, 2, 1], "Redundant size info"),
    ("pad", [50, 30], "Insufficient size info"),
    ("pad", [30, 50, 3], "width of the pad"),
    ("pad", [50, 30, 2], "at least be 3 mm"),
    ("ring", [7, 6, 2], "inner radius"),
    ("disc", [-1, 2], "non-negative"),
])
def test_size_validation(elec_type, size, message):
    with pytest.raises(ValueError, match=message):
        build_elec_para(["a"], elec_type=elec_type, elec_size=size)


def test_mixed_types_need_one_entry_per_electrode():
    with pytest.raises(ValueError, match="which type for each electrode"):
        build_elec_para(["a", "b"], elec_type=["disc"])


def test_conductivities_expand_per_electrode():
    values = validate_conductivities({"gel": 0.5}, 3)
    assert values["gel"].tolist() == [0.5, 0.5, 0.5]
    assert values["electrode"].shape == (3,)


def test_conductivities_reject_unknown_tissue():
    with pytest.raises(ValueError, match="Unrecognized tissue names"):
        validate_conductivities({"muscle": 0.5}, 1)


def test_mesh_options_reject_unknown_keys():
    with pytest.raises(ValueError, match="Unrecognized mesh options"):
        validate_mesh_options({"radbound": 5, "nope": 1})
    assert validate_mesh_options(None)["maxvol"] == 10


def test_is_new_options_detects_a_changed_recipe():
    assert not is_new_options(_options(), _options())
    assert is_new_options(_options(config_txt="other"), _options())
    assert is_new_options(_options(zero_pad=10), _options())
    assert is_new_options(_options(multiaxial=True), _options())


def test_cap_type_class_is_shared_across_the_10_xx_systems():
    assert not is_new_options(_options(elec_para=build_elec_para(["Fp1"], "1005")),
                              _options(elec_para=build_elec_para(["Fp1"], "1010")))
    assert is_new_options(_options(elec_para=build_elec_para(["Fp1"], "biosemi")),
                          _options(elec_para=build_elec_para(["Fp1"], "1010")))


def _target_options(**overrides):
    defaults = dict(roast_tag="lf", target_coord=np.array([[10.0, 20, 30]]),
                    target_coord_mni=None, target_coord_original=None,
                    opt_type="max-l1", desired_intensity=1.0, elec_num=None,
                    target_radius=2, k=None, u0=np.array([[0.0, 0, 1.0]]),
                    orient=["radial-in"], unique_tag="t")
    defaults.update(overrides)
    return TargetOptions(**defaults)


def test_target_runs_are_order_independent():
    two = _target_options(target_coord=np.array([[1.0, 2, 3], [4.0, 5, 6]]),
                          u0=np.array([[0.0, 0, 1], [1.0, 0, 0]]),
                          orient=["radial-in", "right"])
    flipped = _target_options(target_coord=np.array([[4.0, 5, 6], [1.0, 2, 3]]),
                              u0=np.array([[1.0, 0, 0], [0.0, 0, 1]]),
                              orient=["right", "radial-in"])
    assert not is_new_options(two, flipped)
    assert is_new_options(_target_options(opt_type="wls-l1", k=0.2), _target_options())


def test_elec_preproc_orders_electrodes_by_the_cap_layout(tmp_path):
    paras, order = elec_preproc("subj.nii", ["P4", "Fp1"], [ElecPara()])
    assert order.tolist() == [1, 0]                 # Fp1 comes before P4 in capInfo
    assert paras[0].ind_p.tolist() == [3, 64]


def test_elec_preproc_rejects_unknown_electrodes():
    with pytest.raises(ValueError, match="Unrecognized electrodes"):
        elec_preproc("subj.nii", ["NotAnElectrode"], [ElecPara()])
