import openai
import os
import config

from openai.embeddings_utils import get_embedding
openai.api_key = os.getenv('OpenAI_API_KEY')

response = openai.Embedding.create(
    input="Your text string goes here",
    model="text-embedding-ada-002"
)
embeddings = response['data'][0]['embedding']

print(embeddings)