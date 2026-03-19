from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
print(model)
print(model.embeddings.word_embeddings.state_dict())
