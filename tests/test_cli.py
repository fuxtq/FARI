from fari.cli import build_parser


def test_public_commands_are_stable():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("edit", "diagnose", "build-data", "evaluate"):
        assert command in help_text
