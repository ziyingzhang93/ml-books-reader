from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                    trust_remote_code=True)
print(config.max_position_embeddings)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                          trust_remote_code=True)
print(tokenizer.model_max_length)
