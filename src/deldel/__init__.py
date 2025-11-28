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
)
from .graphing import plot_frontiers_implicit_interactive_v2
from .datasets import make_corner_class_dataset, make_high_dim_classification_dataset
from .experiments import (
    run_corner_pipeline_experiments,
    run_corner_pipeline_with_low_dim,
    run_corner_random_forest_pipeline,
    run_iris_random_forest_pipeline,
    run_low_dim_spaces_demo,
)
from .globalmaj import prune_and_orient_planes_unified_globalmaj
from .frontier_planes_all_modes import (
    compute_frontier_planes_all_modes,
    plot_planes_with_point_lines,
)
from .find_low_dim_spaces_fast import (
    find_low_dim_spaces,
    find_low_dim_spaces_deterministic,
    find_low_dim_spaces_precision_boost,
    find_low_dim_spaces_support_first,
)
from .reporting_plotting import (
    describe_regions_metrics,
    describe_regions_report,
    plot_selected_regions_interactive,
)
from . import stage_infer_ab
from .subspace_change_detector import (
    MultiClassSubspaceExplorer,
    SubspacePlane,
    SubspaceReport,
)

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
    "make_high_dim_classification_dataset",
    "prune_and_orient_planes_unified_globalmaj",
    "run_corner_pipeline_experiments",
    "run_corner_pipeline_with_low_dim",
    "run_corner_random_forest_pipeline",
    "run_iris_random_forest_pipeline",
    "run_low_dim_spaces_demo",
    "find_low_dim_spaces",
    "find_low_dim_spaces_deterministic",
    "find_low_dim_spaces_precision_boost",
    "find_low_dim_spaces_support_first",
    "describe_regions_metrics",
    "describe_regions_report",
    "plot_selected_regions_interactive",
    "MultiClassSubspaceExplorer",
    "SubspacePlane",
    "SubspaceReport",
    "stage_infer_ab",
]
