from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range):  # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)
