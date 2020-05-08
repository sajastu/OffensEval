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

        return (model_output)


class MTLClassifier(nn.Module):
    def __init__(self, args, is_eval=False):
        super(MTLClassifier, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            args.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.is_eval_mode = is_eval
        self.linear_1 = nn.Linear(768, 2)
        self.linear_2 = nn.Linear(768, 3)
        self.linear_3 = nn.Linear(768, 4)

    def switch_state(self):
        self.is_eval_mode = not self.is_eval_mode

    def forward(self, input_ids, attention_mask=None, labels=None):
        bert_outputs = self.bert_model(input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask)

        # Should give the logits to the the linear layer
        model_output_1 = self.linear_1(bert_outputs[1])
        model_output_2 = self.linear_2(bert_outputs[1])
        model_output_3 = self.linear_3(bert_outputs[1])

        return (model_output_1, model_output_2, model_output_3)
