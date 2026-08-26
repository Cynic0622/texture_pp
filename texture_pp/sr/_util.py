"""SR-specific utilities: silent_run."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout


def silent_run(func, *f_args, **f_kwargs):
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        return func(*f_args, **f_kwargs)
