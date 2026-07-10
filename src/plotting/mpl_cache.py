"""Pin matplotlib's cache dir off NFS home -- import BEFORE matplotlib.

The default MPLCONFIGDIR (``~/.cache/matplotlib``) is on NFS home on this
cluster; ``usetex`` writes its tex cache there via ``TemporaryDirectory``, whose
cleanup then races on ``.nfs*`` lock files and crashes with
``OSError: [Errno 39] Directory not empty``. Importing this module (as the very
first import in a figure script, before numpy/matplotlib) redirects the cache to
local disk. ``setdefault`` preserves any user-provided ``MPLCONFIGDIR``.
"""

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"mplconfig-{os.getuid()}")
)
