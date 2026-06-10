from opencompass.registry import MODELS

from .general_api import BaseGeneralApi


@MODELS.register_module()
class Qwen35API(BaseGeneralApi):
    THINKING_CONTROL_MODE = 'extra_body'
    PARSE_REASONING = True
