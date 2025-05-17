# T5 Summarization

Implementation of dialogue summarization using Google's T5 model.

## Files
- `t5_summarization.py`: Python script version
- `t_5(1).ipynb`: Notebook version

## Usage
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
