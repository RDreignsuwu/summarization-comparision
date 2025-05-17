# T5 Summarization

Implementation of dialogue summarization using Google's T5 model.

## Files
- `t5_summarization.py`: Python script version

## T-5 Summarization Notebook
Interactive version:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ii5umfaAVesedhecjrJIgmlKBXGu7_Qs/view?usp=sharing)

## Usage
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
