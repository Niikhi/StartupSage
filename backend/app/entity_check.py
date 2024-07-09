import spacy
from typing import Dict, Set
import json
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load entity keywords from the JSON file
with open(os.path.join(current_dir, 'entity_keywords.json'), 'r') as f:
    entity_keywords = json.load(f)

def extract_entities(question: str) -> Dict[str, Set[str]]:
    doc = nlp(question)
    
    entities = {
        'LOCATION': set(),
        'CATEGORY': set(),
        'STATUS': set()
    }
    
    # Check for keyword matches
    question_lower = question.lower()
    for entity_type, keywords in entity_keywords.items():
        for keyword in keywords:
            if keyword.lower() in question_lower:
                entities[entity_type].add(keyword)
    
    # Use spaCy's named entity recognition as a fallback
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            entities['LOCATION'].add(ent.text)
    
    # Remove empty sets
    return {k: v for k, v in entities.items() if v}

