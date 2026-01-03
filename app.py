from io import open_code
import streamlit as st
import psycopg2
import torch
import open_clip
from PIL import Image

#Page config
st.set_page_config(page_title="AI Vision Search", page_icon="🤖", layout = "wide")

st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
.stHeader { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("OmniSense")
st.subheader("Your personal Multimodal AI-powered content retriever")
st.write("This application allows you to search for images and audio using natural language queries, leveraging the power of CLIP and Whisper models.")

#Database connection
DB_URL = 'postgresql://neondb_owner:npg_Y0rVlCEbae6y@ep-wispy-forest-ahzuv779-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

#load modules
@st.cache_resource
def load_models():
  model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
  tokenizer = open_clip.get_tokenizer('ViT-B-32')
  return model, preprocess, tokenizer

model, preprocess, tokenizer = load_models()

#Search UI
query = st.text_input("What are you looking for ?", placeholder="e.g. 'a photo of a dog'")
if query:
  #vectorize
  text = tokenizer([query])
  with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=1, keepdim=True)
  search_vector = text_features.tolist()[0]

  #Query DB
  cur.execute("SELECT filename FROM multimodal_search ORDER BY embedding <=> %s::vector LIMIT 3", (search_vector,))
  results = cur.fetchall()

  #Display results in pretty columns
  cols = st.columns(len(results))
  for i, res in enumerate(results):
    with cols[i]:
      st.image(res[0], use_column_width=True, caption=f"Match #{i+1}")
