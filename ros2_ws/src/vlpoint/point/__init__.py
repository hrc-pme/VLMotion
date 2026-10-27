try:
    from .model import LlavaLlamaForCausalLM
except ModuleNotFoundError:
    LlavaLlamaForCausalLM = None
