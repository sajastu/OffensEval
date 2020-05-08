
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, batch_labels):
        if len(outputs) > 1: # calculate mtl loss
            (outputs_a, outputs_b, outputs_c) = outputs
            (batch_labels_a, batch_labels_b, batch_labels_c) = batch_labels
            return self.loss(outputs_a, batch_labels_a) + \
                   self.loss(outputs_b, batch_labels_b) + \
                   self.loss(outputs_c, batch_labels_c)
        else:
            outputs = outputs[0]
            batch_labels = batch_labels[0]
            return self.loss(outputs, batch_labels)

