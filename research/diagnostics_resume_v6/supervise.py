"""Use the sealed GPU controller in a separate, single-use continuation directory."""
from research.diagnostics_v2 import supervise as original
from research.diagnostics_resume_v6.prepare import BASE, load_prepared, source_seal


def main():
    # Overrides are process-local. The original source and its child runners remain sealed.
    original.BASE = BASE
    original.load_prepared = load_prepared
    original.source_seal = source_seal
    original.main()


if __name__ == '__main__':
    main()
