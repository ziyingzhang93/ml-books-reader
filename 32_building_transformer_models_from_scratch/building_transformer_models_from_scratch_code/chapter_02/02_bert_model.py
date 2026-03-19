from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
print(model)
