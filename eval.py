import json

from raid import run_detection, run_evaluation
from raid.utils import load_data
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import tqdm

class Deberta:
    def __init__(
        self
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained("/home/ubuntu/oracle/Research-Papers-Implementation/ai_text_detection/raid_benchmark/raid_detector/checkpoint-5000").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            _, fake = probs.detach().cpu().flatten().numpy().tolist()
            predictions.append(fake)
        return predictions
    


def my_detector(texts: list[str]) -> list[float]:
    model = Deberta()
    return model.inference(texts)
test_df = load_data(split="test")

predictions = run_detection(my_detector, test_df)

evaluations = run_evaluation(predictions,test_df)

with open('predictions.json') as f:
    json.dump(predictions, f)

with open('evaluations.json') as f:
    json.dump(evaluations, f)