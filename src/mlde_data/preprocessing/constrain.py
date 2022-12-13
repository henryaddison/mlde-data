import logging

logger = logging.getLogger(__name__)


class Constrain:
    def __init__(self, query):
        self.query = query

    def run(self, ds):
        logger.info(f"Selecting {self.query} portion of dataset")
        return ds.sel(self.query)
