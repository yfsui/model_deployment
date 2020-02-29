# Model Deployment

Deploy the CNN model as a service on AWS

<br>

### Files included in this repo:

***my_lambda_pre_processor.py***: lambda function

***pre_processing***: preprocessing library
- word_embedding.py: load embedding dictionary
- text_processing.py: clean and tokenize text
- pre_processing.py: pre-process tweets
- nltk_tokenize.py: nltk tokenizer library
- resources
  - glove.txt: embedding dictionary (without vector)
