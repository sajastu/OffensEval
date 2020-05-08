import torch.nn as nn
from transformers import BertModel


class Classifier(nn.Module):

    def __init__(self, args, is_eval=False):
        super(Classifier, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            args.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.is_eval_mode = is_eval
        self.linear = nn.Linear(768, 2 if args.task != 'c' else 3)

    def switch_state(self):
        self.is_eval_mode = not self.is_eval_mode

    def forward(self, input_ids, attention_mask=None, labels=None):

        bert_outputs = self.bert_model(input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask)

        # Should give the logits to the the linear layer
        model_output = self.linear(bert_outputs[1])

        return model_output