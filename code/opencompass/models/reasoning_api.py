from opencompass.registry import MODELS

from .general_api import BaseGeneralApi


@MODELS.register_module()
class ReasoningAPI(BaseGeneralApi):
    THINKING_CONTROL_MODE = 'none'
    PARSE_REASONING = True
