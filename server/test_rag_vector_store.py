import json
import pytest
from pathlib import Path
from server.rag_vector_store import _load_dataset_documents, build_vector_store, query_rag

@pytest.fixture
def setup_temporary_datasets(tmp_path):
    valid_file = tmp_path / "valid_prompts.jsonl"
    invalid_file = tmp_path / "corrupted_prompts.jsonl"
    
    valid_data = [
        {"layout": "minimalist_digital", "color_palette": "neon_blue"},
        {"layout": "classic_analog", "color_palette": "rosegold"}
    ]
    with open(valid_file, "w", encoding="utf-8") as f:
        for entry in valid_data:
            f.write(json.dumps(entry) + "\n")
            
    with open(invalid_file, "w", encoding="utf-8") as f:
        f.write("{broken_json: true\n")
        f.write("\n")
        f.write('{"layout": "valid_interspersed"}\n')
        
    return [valid_file, invalid_file], tmp_path

def test_dataset_ingestion_robustness(setup_temporary_datasets):
    paths, _ = setup_temporary_datasets
    
    documents = _load_dataset_documents(paths)
    
    assert len(documents) == 3, "Ingestion failed to skip blank lines or corrupt lines cleanly!"
    assert "neon_blue" in documents[0].page_content
    assert "valid_interspersed" in documents[2].page_content


def test_query_rag_empty_inputs():
    with pytest.raises(RuntimeError):
        query_rag(query="test", persist_dir=Path("/non_existent_directory_path_xyz"))
        
    print("Vector store parser and invalid data pipeline constraints verified!")