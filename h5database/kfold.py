from . import database
from .database import Database
from .split import Split
import os


class KFold(object):

    def __init__(self, **kwargs):
        self.num_fold = kwargs.get('num_fold', 10)
        self.shuffle = kwargs.get('shuffle', True)
        kwargs['num_fold'] = self.num_fold
        kwargs['shuffle'] = self.shuffle
        self.root_dir = kwargs['export_dir']
        self.data_set: Database = Database(**kwargs)  # init dataset object
        # init split
        self.split = None
        stratified_labels = kwargs.get('stratified_labels', None)
        self.generate_split(stratified_labels=stratified_labels)

    def generate_split(self, stratified_labels=None):
        self.split = list(Split.k_fold_database(self.data_set, stratified_labels=stratified_labels))

    def run(self):
        for fold in range(self.num_fold):
            # redefine split
            self.data_set.export_dir = os.path.join(self.root_dir, str(fold))
            self.data_set.splits[database.train_name], self.data_set.splits[database.train_name] = self.split[fold]
            self.data_set.initialize(force_overwrite=False)
