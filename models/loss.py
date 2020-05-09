import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, mlt=False):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.mlt = mlt

    def forward(self, outputs, batch_labels):
        if self.mlt:  # calculate mtl loss
            (outputs_a, outputs_b, outputs_c) = outputs
            (batch_labels_a, batch_labels_b, batch_labels_c) = batch_labels
            return self.loss(outputs_a, batch_labels_a) + \
                   self.loss(outputs_b, batch_labels_b) + \
                   self.loss(outputs_c, batch_labels_c)
        else:
            return self.loss(outputs, batch_labels)
