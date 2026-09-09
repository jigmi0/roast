"""Solving the volume conductor model with getDP (http://getdp.info).

The electrostatic problem is written out as a ``.pro`` file and handed to the
solver bundled under ``lib/getdp-3.2.0``.  The injected currents enter as
Neumann conditions on the outer surface of each electrode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..config import NUM_OF_TISSUE, arch, lib_dir
from ..utils.logging import get_logger
from .prepare import load_elec_areas

__all__ = ["getdp_path", "write_pro_file", "solve_by_getdp"]

logger = get_logger()

_TISSUE_REGIONS = ("white", "gray", "csf", "bone", "skin", "air")


def getdp_path() -> Path:
    """Location of the bundled getDP executable for this platform."""
    system = arch()
    binary = {"win64": "getdp.exe", "glnxa64": "getdp", "maci64": "getdpMac"}[system]
    path = lib_dir() / "getdp-3.2.0" / "bin" / binary
    if system != "win64" and path.exists():
        path.chmod(0o755)
    return path


def write_pro_file(path, subj_name, uni_tag, current, sigma, ind_use, num_of_elec,
                   areas, lf_tag: str = "") -> Path:
    """Write the getDP problem definition.

    ``ind_use`` holds the 0-based indices of the electrodes that carry current
    in this run: all of them for a plain simulation, a single pair when a lead
    field is being generated.
    """
    ind_use = np.asarray(ind_use, dtype=int).ravel()
    lines = []
    add = lines.append

    add("Group {\n")
    for i, name in enumerate(_TISSUE_REGIONS, start=1):
        add(f"{name} = Region[{i}];")
    for i, index in enumerate(ind_use, start=1):
        add(f"gel{i} = Region[{NUM_OF_TISSUE + index + 1}];")
    for i, index in enumerate(ind_use, start=1):
        add(f"elec{i} = Region[{NUM_OF_TISSUE + num_of_elec + index + 1}];")

    gel_str = "".join(f"gel{i}, " for i in range(1, len(ind_use) + 1))
    elec_str = "".join(f"elec{i}, " for i in range(1, len(ind_use) + 1))
    used_str = "".join(f"usedElec{i}, " for i in range(1, len(ind_use) + 1))
    for i, index in enumerate(ind_use, start=1):
        add(f"usedElec{i} = Region[{NUM_OF_TISSUE + 2 * num_of_elec + index + 1}];")

    tissues = ", ".join(_TISSUE_REGIONS)
    add(f"DomainC = Region[{{{tissues}, {(gel_str + elec_str)[:-2]}}}];")
    add(f"AllDomain = Region[{{{tissues}, {(gel_str + elec_str + used_str)[:-2]}}}];\n")
    add("}\n")

    add("Function {\n")
    for name in _TISSUE_REGIONS:
        add(f"sigma[{name}] = {_fmt(sigma[name])};")
    for i, index in enumerate(ind_use, start=1):
        add(f"sigma[gel{i}] = {_fmt(np.atleast_1d(sigma['gel'])[index])};")
    for i, index in enumerate(ind_use, start=1):
        add(f"sigma[elec{i}] = {_fmt(np.atleast_1d(sigma['electrode'])[index])};")
    for i, index in enumerate(ind_use, start=1):
        add(f"du_dn{i}[] = {_fmt(1000.0 * current[index] / areas[index])};")
    add("}\n")

    add(_JACOBIAN_AND_INTEGRATION)
    add(_FUNCTION_SPACE)

    add("Formulation {")
    add("  { Name Electrostatics_v; Type FemEquation;")
    add("    Quantity {")
    add("      { Name v; Type Local; NameOfSpace Hgrad_v_Ele; }")
    add("    }")
    add("    Equation {")
    add("      Galerkin { [ sigma[] * Dof{d v} , {d v} ]; In DomainC; ")
    add("                 Jacobian Vol; Integration GradGrad; }\n")
    for i in range(1, len(ind_use) + 1):
        add(f"      Galerkin{{ [ -du_dn{i}[], {{v}} ]; In usedElec{i};")
        add("                 Jacobian Sur; Integration GradGrad;}")
    add("    }")
    add("  }")
    add("}\n")

    add(_RESOLUTION_AND_POSTPROCESSING)

    add("PostOperation {\n")
    add("{ Name Map; NameOfPostProcessing EleSta_v;")
    add("   Operation {")
    if not lf_tag:
        add(f'     Print [ v, OnElementsOf DomainC, File "{subj_name}_{uni_tag}_v.pos", '
            "Format NodeTable ];")
    add(f'     Print [ e, OnElementsOf DomainC, Smoothing, '
        f'File "{subj_name}_{uni_tag}_e{lf_tag}.pos", Format NodeTable ];')
    add("   }")
    add("}\n")
    add("}")

    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def _fmt(value) -> str:
    """Format a number the way MATLAB's ``num2str`` does for the .pro file."""
    return "%.10g" % float(value)


_JACOBIAN_AND_INTEGRATION = """Jacobian {
  { Name Vol ;
    Case {
      { Region All ; Jacobian Vol ; }
    }
  }
  { Name Sur ;
    Case {
      { Region All ; Jacobian Sur ; }
    }
  }
}

Integration {
  { Name GradGrad ;
    Case { {Type Gauss ;
            Case { { GeoElement Triangle    ; NumberOfPoints  3 ; }
                   { GeoElement Quadrangle  ; NumberOfPoints  4 ; }
                   { GeoElement Tetrahedron ; NumberOfPoints  4 ; }
                   { GeoElement Hexahedron  ; NumberOfPoints  6 ; }
                   { GeoElement Prism       ; NumberOfPoints  9 ; } }
           }
         }
  }
}
"""

_FUNCTION_SPACE = """FunctionSpace {
  { Name Hgrad_v_Ele; Type Form0;
    BasisFunction {
      // v = v  s   ,  for all nodes
      //      n  n
      { Name sn; NameOfCoef vn; Function BF_Node;
        Support AllDomain; Entity NodesOf[ All ]; }
    }
  }
}
"""

_RESOLUTION_AND_POSTPROCESSING = """Resolution {
  { Name EleSta_v;
    System {
      { Name Sys_Ele; NameOfFormulation Electrostatics_v; }
    }
    Operation { 
      Generate[Sys_Ele]; Solve[Sys_Ele]; SaveSolution[Sys_Ele];
    }
  }
}

PostProcessing {
  { Name EleSta_v; NameOfFormulation Electrostatics_v;
    Quantity {
      { Name v; 
        Value { 
          Local { [ {v} ]; In AllDomain; Jacobian Vol; } 
        }
      }
      { Name e; 
        Value { 
          Local { [ -{d v} ]; In AllDomain; Jacobian Vol; }
        }
      }
    }
  }
}
"""


def solve_by_getdp(subj, current, sigma, ind_use, uni_tag, lf_tag: str = "") -> None:
    """Run getDP on the prepared mesh."""
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem

    areas = load_elec_areas(subj, uni_tag)
    num_of_elec = len(areas)
    pro_file = directory / f"{subj_name}_{uni_tag}.pro"
    write_pro_file(pro_file, subj_name, uni_tag, np.atleast_1d(current), sigma,
                   ind_use, num_of_elec, areas, lf_tag)

    solver = getdp_path()
    command = [str(solver), str(pro_file), "-solve", "EleSta_v",
               "-msh", str(directory / f"{subj_name}_{uni_tag}_ready.msh"), "-pos", "Map"]
    result = subprocess.run(command, cwd=str(directory) if str(directory) else None)
    if result.returncode:
        raise RuntimeError("getDP solver cannot work properly on your system. "
                           "Please check any error message you got.")
    for suffix in (".pre", ".res"):
        (directory / f"{subj_name}_{uni_tag}{suffix}").unlink(missing_ok=True)
