# tools/Python — shared utilities for topology-optimization JSON workflows.
from .config_loader import (
    load_json_config,
    req_num,
    req_int,
    req_str,
    has_field_path,
    get_field_path,
)
from .bc_processor import supports_to_fixed_dofs
from .passive_regions import parse_passive_regions
from .load_cases import validate_load_cases
from .nodal_projection import project_q4_element_density_to_nodes

__all__ = [
    "load_json_config",
    "req_num",
    "req_int",
    "req_str",
    "has_field_path",
    "get_field_path",
    "supports_to_fixed_dofs",
    "parse_passive_regions",
    "validate_load_cases",
    "project_q4_element_density_to_nodes",
]
