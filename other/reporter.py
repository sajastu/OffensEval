import sys
from other.utils import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_docs=0, n_correct=0):
        self.loss = loss
        self.n_docs = n_docs

    def update(self, loss, docs):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += loss

        self.n_docs += docs

    def reset(self):
        self.loss = 0
        self.n_docs = 0

    def _get_loss(self):
        if (self.n_docs == 0):
            return 0
        return self.loss / self.n_docs

    def _report_stat(self, step, num_steps, epoch, total_epochs):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s (epoch %d/%d); loss: %4.2f ")
            % (step_fmt, epoch, total_epochs,
               self._get_loss()))
        sys.stdout.flush()