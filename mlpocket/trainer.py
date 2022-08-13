class Trainer:
    """
    Trainer responsible for training the models available in our package
    """
    def __init__(self):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        Fits the model in the training data with data passed to the method

        :param X_train: Training inputs
        :param Y_train: Training labels
        :param X_val: Validation inputs
        :param Y_val: Validation labels
        :param kwargs: additional parameters specific to the model used

        :return:
        """
        raise NotImplementedError

    def test(self, x, y, **kwargs):
        raise NotImplementedError
