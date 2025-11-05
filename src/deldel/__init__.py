"""DelDel: herramientas para análisis de modelos de clasificación."""
from .engine import (
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    DeltaRecord,
    DeltaRecordLite,
    ModelCall,
    PCA3D,
    ScoreAdaptor,
    build_weighted_frontier,
    compute_frontier_planes_weighted,
    fit_tls_plane_weighted,
    fit_quadric_svd_weighted,
    fit_quadrics_from_records_weighted,
    fit_cubic_from_records_weighted,
    plot_frontiers_implicit_interactive_v2,
)
from .datasets import make_corner_class_dataset
from .experiments import (
    prune_and_orient_planes_unified_globalmaj,
    run_corner_pipeline_experiments,
    run_corner_pipeline_with_low_dim,
    run_iris_random_forest_pipeline,
    run_low_dim_spaces_demo,
)
from .frontier_planes_all_modes import (
    compute_frontier_planes_all_modes,
    plot_planes_with_point_lines,
)
from .find_low_dim_spaces_fast import find_low_dim_spaces

__all__ = [
    "ChangePointConfig",
    "DelDel",
    "DelDelConfig",
    "DeltaRecord",
    "DeltaRecordLite",
    "ModelCall",
    "PCA3D",
    "ScoreAdaptor",
    "build_weighted_frontier",
    "compute_frontier_planes_weighted",
    "compute_frontier_planes_all_modes",
    "fit_tls_plane_weighted",
    "fit_quadric_svd_weighted",
    "fit_quadrics_from_records_weighted",
    "fit_cubic_from_records_weighted",
    "plot_frontiers_implicit_interactive_v2",
    "plot_planes_with_point_lines",
    "make_corner_class_dataset",
    "prune_and_orient_planes_unified_globalmaj",
    "run_corner_pipeline_experiments",
    "run_corner_pipeline_with_low_dim",
    "run_iris_random_forest_pipeline",
    "run_low_dim_spaces_demo",
    "find_low_dim_spaces",
]
