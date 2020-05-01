import torch.nn as nn
from transformers import BertModel


class Classifier(nn.Module):

    def __init__(self, args, is_eval=False):
        super(Classifier, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            args.init_checkpoint,
            num_labels=2 if args.task != 'c' else 3,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.is_eval_mode = is_eval
        self.linear_1 = nn.Linear(768, 2 if args.task != 'c' else 3)

    def switch_state(self):
        self.is_eval_mode = not self.is_eval_mode

    def forward(self, input_ids, attention_mask=None, labels=None):

        bert_outputs = self.bert_model(input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask)
        model_output = self.linear_1(bert_outputs[1])
        # if self.is_eval_mode:
        #     import pdb;pdb.set_trace()

        return model_output