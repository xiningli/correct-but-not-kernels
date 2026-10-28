"""Observe a manually launched continuation, verify its archive and acknowledge it."""
from research.diagnostics_v2 import monitor as original
from research.diagnostics_resume_v6.prepare import BASE


def main():
    original.BASE = BASE
    original.PROBE = original.PROBE.replace("b=root/'runs/diagnostics-v6'", "b=root/'runs/diagnostics-v6-resume31b'")
    remote = original.remote

    def observe(code, timeout=50):
        response = remote(code, timeout)
        if code == original.PROBE:
            # No staging/download/launch path in this observer, even if old CPU metadata is copied.
            response['preparation'] = {}
        return response

    original.remote = observe
    original.main()


if __name__ == '__main__':
    main()
