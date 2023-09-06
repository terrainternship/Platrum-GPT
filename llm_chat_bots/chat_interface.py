from abc import ABC, abstractmethod


class ChatInterface(ABC):

    @abstractmethod
    def init(self, config):
        pass

    @abstractmethod
    def chat(self, message):
        pass
