"""Shared geometry kernels.

UI-independent, dependency-light (numpy only). Consumed by `platform/` and
`encounters/` alike.

⚠️ Nothing here may import a FastAPI application module, a server, or a UI. That boundary is
what lets an analysis result be reproduced without running SCI-LiDAR Studio.
"""

__version__ = "0.1.0"
