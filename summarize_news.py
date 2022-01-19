from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import requests
from bs4 import BeautifulSoup
import random

# Pull a random Top news article for today from CNN at http://lite.cnn.com/en
def get_random_article():
  re = requests.get('http://lite.cnn.com/en')
  soup = BeautifulSoup(re.text, 'html.parser')
  
  # get all the urls for CNN articles for today
  all_urls = soup.find_all('a')
  articles_urls = [tag["href"] for tag in all_urls if 'article' in tag["href"]]
  
  #randomly select an article from the list list of articles
  article_url = random.choice(articles_urls)
  
  return article_url

def get_text_article(url):
  # get the text from that random article
  re = requests.get(url)
  soup = BeautifulSoup(re.text, 'html.parser')
  
  TITLE = soup.find_all('h2')
  text = soup.find_all('p')
  ARTICLE_TO_SUMMARIZE = []
  for x in text:
    if x.string in x:
      ARTICLE_TO_SUMMARIZE.append(x.string)
  return TITLE[0].string, " ".join(ARTICLE_TO_SUMMARIZE)

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

article_url = 'http://lite.cnn.com' + str(get_random_article())
TITLE, ARTICLE_TO_SUMMARIZE = get_text_article(article_url)

# Generate the Predicted Summary
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
PRED = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

# print summarization
print('Original Article Link:', article_url)
print('TITLE: ',TITLE)
print('AUTOMATED ABSTRACTIVE SUMMARY:')
print(PRED.replace('. ', '.\n'))
