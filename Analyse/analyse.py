from datasets import load_dataset
corpus  = load_dataset("vidore/vidore_v3_pharmaceuticals", "corpus",  split="test")
queries = load_dataset("vidore/vidore_v3_pharmaceuticals", "queries", split="test")
qrels   = load_dataset("vidore/vidore_v3_pharmaceuticals", "qrels",   split="test")
# Évaluation : pip install vidore-benchmark
# Modèles VDR : vidore/colqwen2.5-3b-v1.0  ou  vidore/colpali-v1.3  (HuggingFace transformers)
