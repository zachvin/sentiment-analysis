from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


app = FastAPI()

class InputData(BaseModel):
    input: str

class distilbert_sentiment(nn.Module):
    def __init__(self, dropout, num_labels):
        super(distilbert_sentiment, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
        self.dropout = torch.nn.Dropout(dropout)
        self.l2 = torch.nn.Linear(768, num_labels, dtype=torch.float16)

    def forward(self, input_ids, attention_mask):
        x = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        return self.l2(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = distilbert_sentiment(dropout=0, num_labels=6).to(device)
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

encoder_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}


@app.post("/predict")
async def predict(data: InputData):
    
    inp = tokenizer(data.input, padding='max_length', truncation=True, return_tensors="pt", max_length=100).to(device)

    with torch.no_grad():
        logits = model(inp['input_ids'], inp['attention_mask'])
        pred = torch.argmax(logits, dim=1)
        pred = encoder_map.get(pred.cpu().numpy()[0], "output not recognized")

    return {"prediction": pred}