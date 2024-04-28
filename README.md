# MedAI
Benchmarking application for Medical Diagnostic LLMs. Utilizes a custom weighted
similarity metric to test model performance on challenging NEJM test cases.

# Usage
1. run ```pip install -r requirements.txt```
2. Set required device in ```app.py```, if cuda is available can leverage bnb to
   quantize LLM models for smaller loading requirements
3. run ```python benchmark.py``` to generate a json file with benchmark results
