# Description: Custom metrics for the learnfmri package.
import torch

from monai.metrics.metric import CumulativeIterationMetric


class CosineSimilarityMetric(CumulativeIterationMetric):
    def __init__(self):
        super().__init__()

    def _compute_tensor(self, input1, input2) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors. Inputs should be of shape (batch_size, representation_dim).

        returns the cosine similarity between the two tensors with shape (batch_size,).
        """
        return torch.nn.functional.cosine_similarity(input1, input2, dim=1)

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # take the mean of the batches
        return torch.mean(data)


class AntagonistCosineSimilarity(CosineSimilarityMetric):
    def __init__(self):
        super().__init__()

    def _compute_tensor(self, input1, input2, antagonistic_input) -> torch.Tensor:
        sim1 = torch.nn.functional.cosine_similarity(input1, antagonistic_input, dim=1)
        sim2 = torch.nn.functional.cosine_similarity(input2, antagonistic_input, dim=1)
        return (sim1 + sim2) / 2


class CosineSimilarityDifference(CumulativeIterationMetric):
    """
    Which is closer? The two inputs or the antagonistic input?
    If the two inputs are closer, the value will be positive.
    If the antagonistic input is closer, the value will be negative.
    """

    def __init__(self):
        super().__init__()

    def _compute_tensor(self, input1, input2, antagonistic_input) -> torch.Tensor:
        sim = torch.nn.functional.cosine_similarity(input1, input2, dim=1)
        asim1 = torch.nn.functional.cosine_similarity(input1, antagonistic_input, dim=1)
        asim2 = torch.nn.functional.cosine_similarity(input2, antagonistic_input, dim=1)
        return sim - (asim1 + asim2) / 2

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # take the mean of the batches
        return torch.mean(data)


class PairIdentificationAccuracyMetric(CumulativeIterationMetric):
    """
    Computes the accuracy of identifying the correct pairs in each batch.
    This isn't a perfect metric as it depends on the batch size: larger batch sizes will have lower accuracy while
    smaller batch sizes will have higher accuracy.
    """

    def __init__(self):
        super().__init__()

    def _compute_tensor(self, input1, input2) -> torch.Tensor:
        # concatenate the inputs along the batch dimension
        repr = torch.cat([input1, input2], dim=0)
        similarity = torch.nn.functional.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)

        # is the diagonal the highest value?
        correct = torch.argmax(similarity, dim=1) == torch.arange(similarity.shape[0])

        return correct

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # compute the accuracy
        return torch.mean(data.type(torch.float))


class ContrastiveLossMetric(CumulativeIterationMetric):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def _compute_tensor(self, input1, input2) -> torch.Tensor:
        # modified from MONAI ContrastiveLoss
        temperature_tensor = torch.as_tensor(self.temperature).to(input1.device)
        batch_size = input1.shape[0]

        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input1.device)

        repr = torch.cat([input1, input2], dim=0)
        sim_matrix = torch.nn.functional.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / temperature_tensor)
        denominator = negatives_mask * torch.exp(sim_matrix / temperature_tensor)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        return loss_partial

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # take the mean of the batches
        return torch.mean(data)


class AntagonisticContrastiveLossMetric(CumulativeIterationMetric):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def _compute_tensor(self, input1, input2, antagonistic_input) -> torch.Tensor:
        contrastive_loss = super()._compute_tensor(input1, input2)
        antagonistic_contrastive_loss1 = super()._compute_tensor(input1, antagonistic_input)
        antagonistic_contrastive_loss2 = super()._compute_tensor(input2, antagonistic_input)

        return contrastive_loss - (antagonistic_contrastive_loss1 + antagonistic_contrastive_loss2) / 2

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # take the mean of the batches
        return torch.mean(data)


class ThirdPartyContrastiveLossMetric(CumulativeIterationMetric):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def _compute_tensor(self, input1, input2, input3) -> torch.Tensor:
        contrastive_loss1 = super()._compute_tensor(input1, input2)
        contrastive_loss2 = super()._compute_tensor(input2, input3)

        return (contrastive_loss1 + contrastive_loss2) / 2

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")
        # take the mean of the batches
        return torch.mean(data)
