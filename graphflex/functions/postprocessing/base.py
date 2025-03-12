from abc import ABC, abstractmethod
class PostProcessing(ABC):
    @abstractmethod
    def reduce(self, matrix, column_dct):
        pass

class NoneProcessor(PostProcessing):
    def reduce(self, matrix, column_dct):
        return matrix, column_dct
