from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from monai.config import IgniteInfo
from monai.engines.utils import IterationEvents, default_metric_cmp_fn, default_prepare_batch
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
from monai.engines.evaluator import Evaluator

Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


class AntagonisticContrastiveEvaluator(Evaluator):

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        main_network: torch.nn.Module,
        antagonistic_network: torch.nn.Module,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        # inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        # amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=False,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )
        self.main_network = main_network
        self.antagonistic_network = antagonistic_network

    def _iteration(self, engine: AntagonisticContrastiveEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)

        # unpack the batch data
        main_batch1, main_batch2, antagonist_batch = batch
        main_batch1 = main_batch1.to(engine.state.device)
        main_batch2 = main_batch2.to(engine.state.device)
        antagonist_batch = antagonist_batch.to(engine.state.device)

        # put iteration outputs into engine.state
        engine.state.output = {"main_batch1": main_batch1,
                               "main_batch2": main_batch2,
                               "antagonist_batch": antagonist_batch}

        # forward pass
        with engine.mode(engine.network):
            representations1 = engine.main_network(main_batch1)
            representations2 = engine.main_network(main_batch2)
            antagonist_representations = engine.antagonistic_network(antagonist_batch)
            engine.state.output["representations1"] = representations1
            engine.state.output["representations2"] = representations2
            engine.state.output["antagonist_representations"] = antagonist_representations

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
