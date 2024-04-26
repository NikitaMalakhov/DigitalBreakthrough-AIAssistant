import torch
from transformers import AutoModel, AutoTokenizer

from typing import List

class Embeddings:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _average_pool(self, last_hidden_states: torch.tensor,
                 attention_mask: torch.tensor) -> torch.tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def exec(self, sentences: List[str]) -> torch.tensor:
        """Get embeddings for sentences

        Args:
            sentences (List[str]): list of sentences to embed

        Returns:
            torch.tensor: return tensor of embeddings
        """
        formatted_sentences = [f'query: {sentence}' for sentence in sentences]

        tokenized_sentences = self.tokenizer(
            formatted_sentences, max_length=512,
            padding=True, truncation=True, return_tensors='pt'
        )

        tokenized_sentences['input_ids'] = tokenized_sentences['input_ids'].to(self.device)
        tokenized_sentences['attention_mask'] = tokenized_sentences['attention_mask'].to(self.device)
        
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(len(formatted_sentences)):
                batch = {'input_ids': tokenized_sentences['input_ids'][i].unsqueeze(0), 'attention_mask': tokenized_sentences['attention_mask'][i].unsqueeze(0)}
                out = self.model(**batch)
                embeddings = self._average_pool(out.last_hidden_state, tokenized_sentences['attention_mask'][i].unsqueeze(0))
                outputs.append(embeddings.squeeze(0))
        return torch.stack(outputs).cpu()
