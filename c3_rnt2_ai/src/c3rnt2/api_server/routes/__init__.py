from .core import register_core_routes
from .local_lab import register_local_lab_routes
from .utilities import register_utility_routes

__all__ = [
    "register_core_routes",
    "register_local_lab_routes",
    "register_utility_routes",
]
