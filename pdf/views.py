from django.shortcuts import render
from .models import UploadedFile
from django.views import View
from django.http import HttpResponse
import re
from PyPDF2 import PdfReader
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def create_similarity_matrix(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def extract_top_sentences(similarity_matrix, sentences, top_n=5):
    scores = similarity_matrix.sum(axis=1)
    top_sentence_indices = np.argsort(-scores)[:top_n]
    top_sentences = [sentences[i] for i in top_sentence_indices]
    return top_sentences

def encode_sentences(sentences, tokenizer, model):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_masks = attention_masks.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    embeddings = outputs[0][:, 0, :].detach()
    return embeddings.cpu().numpy()


class FileUploadView(View):

    def get(self, request):
        return render(request, 'pdf/home.html')
    
    def post(self, request):
        uploaded_file = request.FILES['document']
        instance = UploadedFile(file=uploaded_file)
        instance.save()
        top_n_sentences = 10
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        antecedentes_text = extract_antecedentes(text)  
        sentences = antecedentes_text.split('.')
        embeddings = encode_sentences(sentences, tokenizer, model)
        similarity_matrix = create_similarity_matrix(embeddings)
        top_sentences = extract_top_sentences(similarity_matrix, sentences, top_n=top_n_sentences)
        summary = ' '.join(top_sentences)

        result_pattern1 = apply_pattern1(text)
        result_pattern2 = apply_pattern2(text)

        context = {
            'pattern1': result_pattern1,
            'pattern2': result_pattern2,
            'summary': summary
        }
        return render(request, 'pdf/resultado.html', context)

class ResultView(View):
    def get(self, request):
        return render(request, 'pdf/resultado.html')
    
def home_view(request):
    return render(request, 'pdf/home.html')



def apply_pattern1(text):
    pattern = re.search(r"[CTS]-\d+\/\d{2,4}", text)
    return pattern.group() if pattern else 'No se encontró coincidencias'

def apply_pattern2(text):
    pattern = re.findall(r"RESUELVE(.*?)(Cópiese|Notifíquese)", text, re.DOTALL)
    if pattern:
        processed_pattern = []
        for item in pattern:
            cleaned_item = re.sub(r'[\.\-\*\s]*(Primero)', r'\1', item[0])
            cleaned_item = re.sub(r'\s+', ' ', cleaned_item)
            marked_item = re.sub(r'(?<!^)((Primero\.|Segundo\.|Tercero\.|Cuarto\.|Quinto\.|Sexto\.|Séptimo\.|Octavo\.|Noveno\.|Décimo\.)\s*-\s*)(\w+)', r'<br><strong>\2\3</strong>', cleaned_item, flags=re.DOTALL)
            processed_pattern.append(marked_item.strip())
        return processed_pattern
    else:
        return ['No se encontró coincidencias']



def extract_antecedentes(text):
    pattern = r"I\. ANTECEDENTES(.*?)II\."
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""
