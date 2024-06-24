# Learn fMRI representations using contrastive learning
The idea is to learn a representation of the fMRI data using a self-supervised approach.
With contrastive learning fMRI data from teh same subject will be more similar in the learned
representation space than data from different subjects.
Representation learning isn't new, but I want to add a feature where the representations
are dissimilar to any representation that could be learned using T1w iamges.
To do this, I will subtract the similarity of the T1w representation to the fMRI representation from  the 
loss function.
This will force the model to learn a representation that the T1w representation model cannot learn.
The fMRI representation network will be trained in similar to siamese networks, and with a contrastive loss function.
The T1w representation network will be trained with a contrastive loss function to the fMRI representation network.
So the only job of the T1w representation network is to learn a representation that is as
similar as possible to the fMRI representation network.
Therefore, the T1w network is antagonistic to the fMRI network.

I am calling this "template-free antagonistic fMRI representation learning".