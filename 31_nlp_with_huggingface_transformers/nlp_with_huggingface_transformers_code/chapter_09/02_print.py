from transformers import AutoConfig, AutoModelForSeq2SeqLM

def explore_model_architecture():
    """Examine DistilBart's configuration and architecture."""
    model_name = "sshleifer/distilbart-cnn-12-6"

    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    print("Model Architecture:")
    print(f"- Encoder layers: {config.encoder_layers}")
    print(f"- Decoder layers: {config.decoder_layers}")
    print(f"- Hidden size: {config.hidden_size}")
    print(f"- Attention heads: {config.encoder_attention_heads}")

    # Verify encoder-decoder structure
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("\nModel Components:")
    print(f"- Encoder: {type(model.model.encoder).__name__}")
    print(f"- Decoder: {type(model.model.decoder).__name__}")
    return model, config

# Example usage
model, config = explore_model_architecture()
print(model)
