from therapanacea_project.metrics.false_acceptance_rate import (
    FalseAcceptanceRate,
)
from therapanacea_project.metrics.false_rejection_rate import FalseRejectionRate
from therapanacea_project.metrics.half_total_error_rate import (
    HalfTotalErrorRate,
)

dict_metrics = {
    "FalseAcceptanceRate": FalseAcceptanceRate,
    "FalseRejectionRate": FalseRejectionRate,
    "HalfTotalErrorRate": HalfTotalErrorRate,
}
