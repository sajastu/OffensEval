import os
import random
from os import path

import torch
from torch.utils.tensorboard import SummaryWriter

from data.data_loader import MyDataLoader
from models.classifier import Classifier
from models.trainer import build_trainer
from other.reporter import Statistics
from other.utils import cmdline_args, evaluate_model_normal
from other.utils import init_logger
from other.utils import logger


def train(args):
    init_logger(args.log_file)

    logger.info('Device ID %d' % args.device_id)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.device_id >= 0:
        torch.cuda.set_device(args.device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    dl = MyDataLoader(args)

    # Loading train and dev set
    dl.load_train_set(task=args.task)
    dl.load_dev_set(task=args.task)

    # Extracting sentences to prepare for BERT module
    training_sents, training_labels = dl.sent_label_extractor()
    test_sentences, test_labels, test_tweetids = dl.sent_label_extractor(is_eval=True)

    # Processing text through BERT ...
    training_inputs, trainig_masks, training_labels = dl.prepare_bert_data(training_sents, training_labels)
    test_inputs, test_masks, test_labels = dl.prepare_bert_data(test_sentences, test_labels)

    # Preparing data as input to neural network
    train_dataloader = dl.get_dataloader(training_inputs, trainig_masks, training_labels)
    dev_dataloader = dl.get_dataloader(test_inputs, test_masks, test_labels)

    logger.info("Loading model ...")

    # Constructing classification model
    model = Classifier(args)

    model.cuda()
    logger.info(model)

    # Building trainer model
    trainer = build_trainer(args, model, len(train_dataloader))

    # Creating TensorBoard object to keep track of train and loss.

    if not path.exists(args.model_path + '/stats'):
        os.makedirs(args.model_path + '/stats')

    writer = SummaryWriter(args.model_path + '/stats')

    train_stats = Statistics()
    f1_history = list()
    logger.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        train_stats.reset()
        trainer.set_train()
        for step, batch in enumerate(train_dataloader):
            loss, train_stats = trainer.step(batch, train_stats)

            if step % args.report_every == 0:
                train_stats._report_stat((epoch - 1) * len(train_dataloader) + step,
                                         args.epochs * len(train_dataloader), epoch, args.epochs)

            total_train_loss += loss

        avg_train_loss = total_train_loss / len(train_dataloader)

        writer.add_scalar('Train/loss', avg_train_loss, epoch)
        # torch.save(model.state_dict(), f'{args.model_path}/model-{epoch}-{args.task}.pt')
        logger.info(('Training loss for epoch %d/%d: %4.2f') % (epoch, args.epochs, avg_train_loss))

        print("\n-------------------------------")
        logger.info('Start validation ...')
        trainer.set_eval()
        y_hat = list()
        y = list()
        total_dev_loss = 0
        for step, batch_val in enumerate(dev_dataloader):
            true_labels_ids, predicted_labels_ids, loss = trainer.validate(batch_val)
            total_dev_loss += loss
            y.extend(true_labels_ids)
            y_hat.extend(predicted_labels_ids)
        avg_dev_loss = total_dev_loss / len(dev_dataloader)
        logger.info(("Total dev loss: %4.2f on epoch %d/%d") % (avg_dev_loss, epoch, args.epochs))

        import pdb;pdb.set_trace()
        f1_score, pr, rec = evaluate_model_normal(y, y_hat)
        print(('\n -Overall: F1: %4.3f (Precision: %4.3f, Recall: %4.3f)') % (f1_score, pr, rec))

        f1_history.append(f1_score)
        if (trainer._maybe_save_model(f1_score, epoch=epoch)):
            logger.info("[BEST model] saved! ")

        writer.add_scalar('Dev/loss', avg_dev_loss, epoch)
        writer.add_scalars('Dev/scores', {'F1': f1_score, 'Precision': pr, 'Recall': rec}, epoch)

        print("\n-------------------------------")

    logger.info("Training terminated!")


def test(args):
    init_logger(args.log_file)

    logger.info('Device ID %d' % args.device_id)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.device_id >= 0:
        torch.cuda.set_device(args.device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    dl = MyDataLoader(args)

    # Loading test set
    dl.load_test_set(task=args.task)

    # Extracting sentences to prepare for BERT module
    test_sentences, test_labels, test_tweetids = dl.sent_label_extractor(is_test=True)

    # Processing text through BERT ...
    test_inputs, test_masks, test_labels = dl.prepare_bert_data(test_sentences, test_labels)

    # Preparing data as input to neural network
    test_dataloader = dl.get_dataloader(test_inputs, test_masks, test_labels)

    logger.info("Loading model...")

    # Constructing classification model
    model = Classifier(args)
    model.cuda()
    # logger.info(model)

    trainer = build_trainer(args, model, len(test_dataloader), is_test=True)

    logger.info("Start prediction...")

    y, y_hat = list(), list()
    total_test_loss = 0
    for step, batch_val in enumerate(test_dataloader):
        true_labels_ids, predicted_labels_ids, loss = trainer.predict(batch_val)
        total_test_loss += loss
        y.extend(true_labels_ids)
        y_hat.extend(predicted_labels_ids)
    avg_test_loss = total_test_loss / len(test_dataloader)
    logger.info(("Total test loss: %4.3f") % (avg_test_loss))

    f1_score, pr, rec = evaluate_model_normal(y, y_hat)

    print('Prediction scores: \n')
    print(('Overall: F1: %4.3f (Precision: %4.3f, Recall: %4.3f)') % (f1_score, pr, rec))


if __name__ == '__main__':
    args = cmdline_args()
    eval(args.mode + '(args)')