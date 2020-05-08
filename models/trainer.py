import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from models.loss import CELoss
from other.utils import logger


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def gpu_activate():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Running on GPU " + str(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        logger.info("Running on CPU")
    return device


def build_trainer(args, model, train_points, is_test=False):
    device = gpu_activate()
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)
    trainer = Trainer(args, model, device, train_points, is_test)

    return trainer


class Trainer(object):
    def __init__(self, args, model, device, data_points, is_test=False, train_stats=None):
        self.args = args
        self.model = model
        self.device = device
        self.loss = CELoss()

        if is_test:
            # Should load the model from checkpoint
            self.model.load_state_dict(torch.load(args.saved_model))
            logger.info('Loaded saved model from %s' % args.saved_model)
            self.model.eval()

        else:
            self.model.train()
            self.optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
            total_steps = data_points * self.args.epochs
            self.scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=0,
                                                             num_training_steps=total_steps)
            self.best_f1 = {'a':-1000, 'b':-1000, 'c':-1000}

    def step(self, batch, train_stats):
        batch = tuple(t.to(self.device) for t in batch)
        if not self.args.mlt:
            batch_input_ids, batch_input_masks, batch_labels = batch
            batch_labels = (batch_labels)
        else:
            batch_input_ids, batch_input_masks, batch_labelsa, batch_labelsb, batch_labelsc = batch
            batch_labels = (batch_labelsa, batch_labelsb, batch_labelsc)

        self.model.zero_grad()
        outputs = self.model(batch_input_ids,
                             attention_mask=batch_input_masks,
                             labels=batch_labels)

        loss = self.loss(outputs, batch_labels)
        loss = loss.sum()
        (loss / loss.numel()).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        self.scheduler.step()
        train_stats.update(float(loss.cpu().data.numpy()), batch_input_ids.size(0))

        return loss, train_stats

    def validate(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        batch_input_ids, batch_input_masks, batch_labels = batch
        with torch.no_grad():
            model_output = self.model(batch_input_ids,
                                 attention_mask=batch_input_masks,
                                 labels=batch_labels)

        predicted_label_ids = self._predict(model_output)
        label_ids = batch_labels.to('cpu').numpy()

        loss = self.loss(model_output, batch_labels)
        loss = loss.sum()

        return label_ids, predicted_label_ids, loss


    def predict(self, batch):
        return self.validate(batch)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def _predict(self, logits):
        return np.argmax(logits.to('cpu').numpy(), axis=1)

    def _maybe_save_model(self, f1score, task):
        if self.best_f1[task] <= f1score: # save model
            self.best_f1[task] = f1score
            torch.save(self.model.state_dict(), f'{self.args.model_path}/BEST_Task_{task}.pt')
            return True
        else:
            return False

