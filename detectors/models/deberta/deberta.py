from transformers import DebertaForSequenceClassification, AutoTokenizer
import torch
import tqdm

class Deberta:
    def __init__(
        self
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DebertaForSequenceClassification.from_pretrained("/home/ubuntu/oracle/Research-Papers-Implementation/ai_text_detection/raid_benchmark/raid_detector/checkpoint-5000").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            _, fake = probs.detach().cpu().flatten().numpy().tolist()
            predictions.append(fake)
        return predictions
    