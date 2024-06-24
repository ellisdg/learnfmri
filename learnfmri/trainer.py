from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence


from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.transforms import Transform
from monai.utils import min_version, optional_import
from monai.engines import Trainer
from monai.engines.utils import IterationEvents
import torch

Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


class AntagonisticContrastiveTrainer(Trainer):
    def __init__(
            self,
            device: str | torch.device,
            max_epochs: int,
            train_data_loader: Iterable | DataLoader,
            main_network: torch.nn.Module,
            antagonistic_network: torch.nn.Module,
            main_optimizer: Optimizer,
            antagonistic_optimizer: Optimizer,
            loss_function: Callable,
            epoch_length: int | None = None,
            non_blocking: bool = False,
            prepare_batch: Callable = default_prepare_batch,
            iteration_update: Callable[[Engine, Any], Any] | None = None,
            # inferer: Inferer | None = None,  # implement later?
            postprocessing: Transform | None = None,
            key_train_metric: dict[str, Metric] | None = None,
            additional_metrics: dict[str, Metric] | None = None,
            metric_cmp_fn: Callable = default_metric_cmp_fn,
            train_handlers: Sequence | None = None,
            # amp: bool = False  # implement later?
            event_names: list[str | EventEnum | type[EventEnum]] | None = None,
            event_to_attr: dict | None = None,
            decollate: bool = True,
            optim_set_to_none: bool = False,
            to_kwargs: dict | None = None,
            amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers,
            amp=False,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        # TODO: add compilation for the networks
        self.main_network = main_network
        self.antagonistic_network = antagonistic_network

        self.main_optimizer = main_optimizer
        self.antagonistic_optimizer = antagonistic_optimizer
        self.loss_function = loss_function
        self.optim_set_to_none = optim_set_to_none

    def _iteration(self, engine: AntagonisticContrastiveTrainer, batchdata: dict[str, torch.Tensor]) -> dict:
        # TODO: add support for AMP

        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)

        # unpack the batch data
        main_batch1, main_batch2, antagonist_batch = batch
        main_batch1 = main_batch1.to(engine.state.device)
        main_batch2 = main_batch2.to(engine.state.device)
        antagonist_batch = antagonist_batch.to(engine.state.device)

        # put iteration outputs into engine.state
        engine.state.output = {"main_image1": main_batch1,
                               "main_image2": main_batch2,
                               "antagonist_image": antagonist_batch}

        engine.main_network.train()
        engine.antagonistic_network.train()

        # forward pass
        representations1 = engine.main_network(main_batch1)
        representations2 = engine.main_network(main_batch2)
        antagonist_representations = engine.antagonistic_network(antagonist_batch)

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)

        engine.state.output["main_representations1"] = representations1
        engine.state.output["main_representations2"] = representations2
        engine.state.output["antagonist_representations"] = antagonist_representations

        # zero the gradients
        engine.state.antagonistic_optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        # compute antagonistic contrastive loss
        # compute the distance from the representations to the antagonistic representations
        antagonistic_distance1 = engine.loss_function(representations1, antagonist_representations)
        antagonistic_distance2 = engine.loss_function(representations2, antagonist_representations)
        antagonistic_loss = (antagonistic_distance1 + antagonistic_distance2) / 2

        antagonistic_loss.backward()
        engine.state.antagonistic_optimizer.step()

        # zero the gradients for the main network
        engine.state.main_optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        # compute the representation loss
        representation_loss = engine.loss_function(representations1, representations2) - antagonistic_loss
        representation_loss.backward()
        engine.state.main_optimizer.step()

        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        engine.state.output["antagonistic_loss"] = antagonistic_loss.item()
        engine.state.output["representation_loss"] = representation_loss.item()

        return engine.state.output


