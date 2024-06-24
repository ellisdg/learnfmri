import torch
from monai.losses import ContrastiveLoss


class AntagonisticContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5, batch_size: int = -1):
        super(AntagonisticContrastiveLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature, batch_size=batch_size)

    def forward(self, representations1, representations2, antagonistic_representations):
        """
        Computes the Antagonistic Contrastive Loss.
        The contrastive representation loss will encourage the representations of the anchor and positive samples to
        be close to each other but far from the representations of the negative samples and the antagonistic
        samples. The antagonistic representation loss will encourage the representations of the antagonistic samples
        to be close to anchor samples.
        Args:
            representations1: Tensor of shape (batch_size, representation_dim) containing the
            representations of the first set of samples.
            representations2: Tensor of shape (batch_size, representation_dim) containing the
            representations of the second set of samples.
            samples.
            antagonistic_representations: Tensor of shape (batch_size, representation_dim) containing the representations
            of the antagonistic samples.
            For the antagonistic samples, the labels are the same as the anchor samples.
            An antagonistic loss function will be computed to encourage the representations of the anchor and antagonistic
            samples to be as close as possible.
        Returns:
            The contrastive representation loss and the antagonistic representation loss.

        """
        # compute the distance from the representations to the antagonistic representations
        antagonistic_distance1 = self.contrastive_loss(representations1, antagonistic_representations)
        antagonistic_distance2 = self.contrastive_loss(representations2, antagonistic_representations)
        antagonistic_loss = (antagonistic_distance1 + antagonistic_distance2) / 2

        # compute the representation loss
        representation_loss = self.contrastive_loss(representations1, representations2) - antagonistic_loss
        return representation_loss, antagonistic_loss
