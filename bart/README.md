# BART Summarization

Implementation of dialogue summarization using Facebook's BART model.

## Files
- `bart_summarization.py`: Python script version

## BART Summarization Notebook
Interactive version:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1IlrNeFZmtbwiUVSps60m87oaY5hq-K8A/view?usp=sharing)

## Usage
```python
from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
