# BART Summarization

Implementation of dialogue summarization using Facebook's BART model.

## Files
- `bart_summarization.py`: Python script version
- `take2_final_BART(1)(1).ipynb`: Notebook version

## Usage
```python
from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
