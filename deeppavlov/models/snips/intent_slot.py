
class NNModel(Component, Serializable):
    """Abstract class for deep learning components."""
    @abstractmethod
    def train_on_batch(self, x: list, y: list):
        pass

    def process_event(self, event_name, data):
        pass
