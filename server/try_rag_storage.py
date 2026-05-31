from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

FEW_SHOT_DATASET = [
    {
        "input": "Give me a dark amoled gaming face with step tracking and a battery ring.",
        "output": {"Config": {"Background": {"color": "black", "image": "/image/path/to/ai_generated_gaming.png"}, "Widget": {"color": "white", "Item": [{"name": "circle_step", "type": "circle", "position": "top"}]}, "Clock": {"type": "digital", "color": "white", "position": "right"}}}
    },
    {
        "input": "A clean clean white minimalist layout with heart rate monitoring",
        "output": {"Config": {"Background": {"color": "white", "image": "null"}, "Widget": {"color": "black", "Item": [{"name": "circle_hr", "type": "circle", "position": "bottom"}]}, "Clock": {"type": "analog", "color": "black", "position": "center"}}}
    }
]

def initialize_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=FEW_SHOT_DATASET,
        embeddings=OpenAIEmbeddings(),
        vectorstore_cls=Chroma,
        k=1
    )
    return example_selector