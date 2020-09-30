from models.cnn_model import ConvModel


class EvaluateConvMnistModel(ConvModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = self.build_model()

    def build_model(self):

        self.model = ConvModel(self.config).build_model()

        return self.model

