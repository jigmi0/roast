"""The command-line front-end."""

import pytest

from roast import __version__
from roast.cli import build_parser, main, _parse_recipe


def test_recipe_pairs_are_parsed():
    parser = build_parser()
    assert _parse_recipe(["F1:0.3", "P2:-0.3"], parser) == ["F1", 0.3, "P2", -0.3]


@pytest.mark.parametrize("pair", ["F1", "F1:", ":1", "F1:abc"])
def test_bad_recipe_pairs_are_usage_errors(pair, capsys):
    with pytest.raises(SystemExit):
        _parse_recipe([pair], build_parser())
    assert "electrode:current pair" in capsys.readouterr().err


def test_every_command_can_run_without_figures():
    parser = build_parser()
    for argv in (["simulate", "--no-show"],
                 ["target", "--sim-tag", "lf", "--target", "1", "2", "3", "--no-show"],
                 ["review", "--sim-tag", "t", "--no-show"]):
        assert parser.parse_args(argv).no_show is True


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
