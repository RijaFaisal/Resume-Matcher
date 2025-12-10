import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from src.api.ingest import read_txt, read_pdf, list_files, chunk_text, Ingestor


@pytest.fixture
def sample_text():
    return "Hello world. " * 50


@pytest.fixture
def mock_ingestor():
    with patch("src.api.ingest.SentenceTransformer") as mock_model:
        mock_model.return_value.encode.return_value = np.array([[0.1, 0.2]])
        ing = Ingestor(model_name="test-model")
        yield ing, mock_model


def test_read_txt():
    with patch("pathlib.Path.open", mock_open(read_data="content")):
        assert read_txt(Path("dummy.txt")) == "content"


def test_read_pdf():
    # Mock PyPDF2
    with patch("builtins.open", mock_open(read_data=b"pdf_content")):
        with patch("src.api.ingest.PyPDF2.PdfReader") as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Page text"
            mock_reader.return_value.pages = [mock_page]
            assert read_pdf(Path("dummy.pdf")) == "Page text"


def test_list_files(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").touch()
    (d / "b.pdf").touch()
    (d / "c.csv").touch()

    files = list_files(d)
    assert len(files) == 2
    assert any(f.name == "a.txt" for f in files)


def test_chunk_text(sample_text):
    chunks = chunk_text(sample_text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_ingestor_initialization(mock_ingestor):
    ing, _ = mock_ingestor
    assert ing.model_name == "test-model"


def test_embed_texts(mock_ingestor):
    ing, mock_model = mock_ingestor
    texts = ["hello"]
    embs = ing.embed_texts(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (1, 2)


@patch("src.api.ingest.faiss")
def test_build_faiss_index(mock_faiss, mock_ingestor):
    ing, _ = mock_ingestor
    embs = np.array([[0.1, 0.2]], dtype=np.float32)
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index

    index = ing.build_faiss_index(embs)

    mock_faiss.normalize_L2.assert_called_once()
    mock_index.add.assert_called_once()
    assert index == mock_index


@patch("src.api.ingest.pickle")
def test_save_index_and_metadata(mock_pickle, mock_ingestor, tmp_path):
    ing, _ = mock_ingestor
    index_dir = tmp_path / "index"
    mock_index = MagicMock()

    with patch("src.api.ingest.faiss.write_index") as mock_write:
        ing.save_index(mock_index, index_dir)
        mock_write.assert_called_once()

    ing.save_metadata([{"id": 1}], index_dir)
    assert (index_dir / "metadata.pkl").exists()


@patch("src.api.ingest.list_files")
@patch("src.api.ingest.read_txt")
def test_ingest_flow(mock_read, mock_list, mock_ingestor, tmp_path):
    ing, _ = mock_ingestor
    mock_list.return_value = [Path("test.txt")]
    mock_read.return_value = "Content"

    # Mock build_faiss_index and save calls to verify flow
    with patch.object(ing, "embed_texts") as mock_embed, patch.object(
        ing, "build_faiss_index"
    ) as mock_build, patch.object(ing, "save_index") as mock_save_idx, patch.object(
        ing, "save_metadata"
    ) as mock_save_meta:

        mock_embed.return_value = np.array([[0.1, 0.2]], dtype=np.float32)

        ing.ingest(Path("data"), tmp_path)

        mock_list.assert_called_once()
        mock_read.assert_called_once()
        mock_embed.assert_called_once()
        mock_build.assert_called_once()
        mock_save_idx.assert_called_once()
        mock_save_meta.assert_called_once()
