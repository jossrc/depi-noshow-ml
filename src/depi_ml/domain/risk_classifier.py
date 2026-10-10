"""Clasificador C4 deshabilitado hasta justificar y aprobar bandas."""


class RiskClassifier:
    status = "pending_validation"

    def classify(self, probability: float) -> None:
        return None
