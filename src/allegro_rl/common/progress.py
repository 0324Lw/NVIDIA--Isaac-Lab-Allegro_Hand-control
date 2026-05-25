from __future__ import annotations

import time
from contextlib import contextmanager


def env_fps(env_steps: int, start_time: float) -> float:
    return float(env_steps) / max(time.time() - float(start_time), 1e-6)


@contextmanager
def tqdm_or_dummy(*args, **kwargs):
    try:
        from tqdm import tqdm
        bar = tqdm(*args, **kwargs)
        try:
            yield bar
        finally:
            bar.close()
    except Exception:
        class Dummy:
            def update(self, *_, **__): pass
            def set_postfix(self, *_, **__): pass
            def write(self, msg): print(msg, flush=True)
        yield Dummy()
