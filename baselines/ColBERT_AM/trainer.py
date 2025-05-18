from ColBERT_AM.infra.run import Run
from ColBERT_AM.infra.launcher import Launcher
from ColBERT_AM.infra.config import ColBERTConfig, RunConfig

from ColBERT_AM.training.training import train


class Trainer:
    def __init__(self, triples, queries, collection, config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.queries = queries
        self.collection = collection

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)

        launcher = Launcher(train)

        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)


    def best_checkpoint_path(self):
        return self._best_checkpoint_path

