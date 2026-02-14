from .evaluator_base import Text2SQLBaseEvaluator
from opencompass.registry import ICL_EVALUATORS

@ICL_EVALUATORS.register_module()
class Text2SQLMatchEvaluator(Text2SQLBaseEvaluator):

    def __init__(self) -> None:
        super().__init__()
        
    def score(self, predictions, references):
        super().score(predictions, references)
        accuracy = (self.scores["all"]["exact"] / self.scores["all"]["count"]) * 100
        return {"match_accuracy": accuracy}
    