from learnfmri.handlers import (ContrastiveLossMetricHandler, CosineSimilarityMetricHandler,
                                CosineSimilarityDifferenceMetricHandler, PairIdentificationAccuracyMetricHandler,
                                AntagonistCosineSimilarityMetricHandler, AntagonisticContrastiveLossMetricHandler,
                                ThirdPartyContrastiveLossMetricHandler)
from learnfmri.trainer import AntagonisticContrastiveTrainer
from learnfmri.evaluator import AntagonisticContrastiveEvaluator
from monai.losses import ContrastiveLoss
from monai.handlers import StatsHandler, ValidationHandler, from_engine
import torch


def setup_evaluator(device, main_network, antagonistic_network, val_loader):
    evaluator = AntagonisticContrastiveEvaluator(
        device=device,
        val_data_loader=val_loader,
        main_network=main_network,
        antagonistic_network=antagonistic_network,
        key_val_metric={
            "loss": AntagonisticContrastiveLossMetricHandler(),  # loss for the main network
            "antagonist_loss": ThirdPartyContrastiveLossMetricHandler(),  # loss for the antagonistic network
            "contrastive_loss": ContrastiveLossMetricHandler(),  # loss without the antagonistic loss subtracted
            "cosine_similarity": CosineSimilarityMetricHandler(),
            "antagonist_cosine_similarity": AntagonistCosineSimilarityMetricHandler(),
            "cosine_similarity_difference": CosineSimilarityDifferenceMetricHandler(),
            "pair_identification_accuracy": PairIdentificationAccuracyMetricHandler(),
        },
            )
    return evaluator


def setup_trainer(device, main_network, antagonistic_network, train_loader, train_handlers, learning_rate=1e-3,
                  max_epochs=100):
    loss = ContrastiveLoss()
    main_optimizer = torch.optim.Adam(main_network.parameters(), learning_rate)
    antagonistic_optimizer = torch.optim.Adam(antagonistic_network.parameters(), learning_rate)

    trainer = AntagonisticContrastiveTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        main_network=main_network,
        main_optimizer=main_optimizer,
        antagonistic_network=antagonistic_network,
        antagonistic_optimizer=antagonistic_optimizer,
        loss_function=loss,
        train_handlers=train_handlers,
    )
    return trainer


def train_fold(main_network, antagonistic_network, train_loaders, val_loaders, fold_index, device, learning_rate=1e-3,
               interval=1, max_epochs=100):

    evaluator = setup_evaluator(device=device,
                                main_network=main_network,
                                antagonistic_network=antagonistic_network,
                                val_loader=val_loaders[fold_index])

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=interval, epoch_level=True),
        StatsHandler(tag_name="loss", output_transform=from_engine(["representation_loss"], first=True)),
        StatsHandler(tag_name="antagonist_loss", output_transform=from_engine(["antagonist_loss"], first=True)),
    ]

    trainer = setup_trainer(device=device,
                            main_network=main_network,
                            antagonistic_network=antagonistic_network,
                            train_loader=train_loaders[fold_index],
                            train_handlers=train_handlers,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs)

    trainer.run()
    return main_network, antagonistic_network


def main():
    pass


if __name__ == "__main__":
    main()
