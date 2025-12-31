"""Configuration settings for multimodal RAG pipeline"""

from pathlib import Path

# Data storage paths
DATASTORE_PATH = Path(r"C:/Users/Admin/Desktop/DataStore")
MMQA_BASE_PATH = DATASTORE_PATH / "M2RAG_Images" / "mmqa"

# Image paths
TEST_IMAGES_PATH = MMQA_BASE_PATH / "test_images"
FIRST_50_IMAGES_PATH = MMQA_BASE_PATH / "first_50_images"


