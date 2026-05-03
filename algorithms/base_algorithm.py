import abc


class BaseAlgorithm(abc.ABC):
    @abc.abstractmethod
    def select_action(self, obs, legal_action, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def select_greedy_action(self, obs, legal_action, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def store_transition(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str):
        raise NotImplementedError
