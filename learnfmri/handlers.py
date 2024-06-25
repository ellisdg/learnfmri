from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.handlers import from_engine
from learnfmri.metrics import (ContrastiveLossMetric, CosineSimilarityMetric, CosineSimilarityDifference,
                               PairIdentificationAccuracyMetric, AntagonisticContrastiveLossMetric,
                               AntagonistCosineSimilarity, ThirdPartyContrastiveLossMetric)

from collections.abc import Callable


class ContrastiveLossMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2"]),
                 save_details: bool = True):
        metric_fn = ContrastiveLossMetric()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class CosineSimilarityMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2"]),
                 save_details: bool = True):
        metric_fn = CosineSimilarityMetric()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class CosineSimilarityDifferenceMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2",
                                                                 "antagonist_representations"]),
                 save_details: bool = True):
        metric_fn = CosineSimilarityDifference()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class PairIdentificationAccuracyMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2"]),
                 save_details: bool = True):
        metric_fn = PairIdentificationAccuracyMetric()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class AntagonistCosineSimilarityMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2",
                                                                 "antagonist_representations"]),
                 save_details: bool = True):
        metric_fn = AntagonistCosineSimilarity()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class AntagonisticContrastiveLossMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2",
                                                                 "antagonist_representations"]),
                 save_details: bool = True):
        metric_fn = AntagonisticContrastiveLossMetric()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)


class ThirdPartyContrastiveLossMetricHandler(IgniteMetricHandler):
    def __init__(self, output_transform: Callable = from_engine(["main_representations1",
                                                                 "main_representations2",
                                                                 "antagonist_representations"]),
                 save_details: bool = True):
        metric_fn = ThirdPartyContrastiveLossMetric()
        super().__init__(metric_fn, save_details=save_details,
                         output_transform=output_transform)
