from transformers.models.llama.configuration_llama import LlamaConfig

class CustomLlamaConfig(LlamaConfig):
    model_type = "custom_llama3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
