from .base import BasicAPIWrapper


class OpenRouter(BasicAPIWrapper):

    DEFAULT_URL = 'https://openrouter.ai/api/v1/chat/completions'
    KEY_NAME = 'OPENROUTER_API_KEY'
    URL_NAME = 'OPENROUTER_API_URL'
