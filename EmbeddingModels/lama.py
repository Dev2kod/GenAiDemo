from huggingface_hub import model_info
info = model_info("LiquidAI/LFM2-350M-PII-Extract-JP")
print(info.pipeline_tag)
