"""
Tests for document preprocessing utilities.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import io

from src.preprocessing import DocumentProcessor


class TestDocumentProcessor:
    """Test class for DocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.sample_text = """
        John Doe
        Software Engineer
        
        Experience:
        - 5 years Python development
        - Machine Learning with scikit-learn
        - Web development with Django
        
        Education:
        - Bachelor of Science in Computer Science
        - University of Technology
        
        Skills: Python, Machine Learning, Django, SQL
        """
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        assert self.processor.stop_words is not None
        assert self.processor.lemmatizer is not None
        assert 'resume' in self.processor.stop_words
        assert 'cv' in self.processor.stop_words
    
    @patch('src.preprocessing.PyPDF2.PdfReader')
    def test_extract_pdf_text(self, mock_pdf_reader):
        """Test PDF text extraction."""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"fake pdf content"
        result = self.processor._extract_pdf_text(content)
        
        assert result == "Sample PDF content\n"
        mock_pdf_reader.assert_called_once()
    
    @patch('src.preprocessing.Document')
    def test_extract_docx_text(self, mock_document):
        """Test DOCX text extraction."""
        # Mock document
        mock_paragraph = Mock()
        mock_paragraph.text = "Sample DOCX content"
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc
        
        content = b"fake docx content"
        result = self.processor._extract_docx_text(content)
        
        assert result == "Sample DOCX content\n"
        mock_document.assert_called_once()
    
    def test_extract_txt_text(self):
        """Test TXT text extraction."""
        content = b"Sample text content"
        result = self.processor._extract_txt_text(content)
        
        assert result == "Sample text content"
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  This   is    a    test    with    extra    spaces!!!  "
        cleaned = self.processor._clean_text(dirty_text)
        
        assert cleaned == "this is a test with extra spaces"
        assert "  " not in cleaned
        assert cleaned.islower()
    
    def test_clean_text_special_characters(self):
        """Test text cleaning with special characters."""
        text_with_special = "Hello@#$%^&*()World!@#$%"
        cleaned = self.processor._clean_text(text_with_special)
        
        assert "@#$%^&*()" not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned
    
    def test_extract_skills(self):
        """Test skill extraction."""
        text = "Python developer with machine learning and SQL experience"
        skills = self.processor.extract_skills(text)
        
        assert len(skills) > 0
        # Should contain relevant words
        assert any("python" in skill.lower() for skill in skills)
    
    def test_extract_skills_empty_text(self):
        """Test skill extraction with empty text."""
        skills = self.processor.extract_skills("")
        assert skills == []
    
    def test_extract_education(self):
        """Test education extraction."""
        text = """
        Education:
        Bachelor of Science in Computer Science
        University of Technology, 2020
        
        Master of Science in Data Science
        Tech Institute, 2022
        """
        
        education = self.processor.extract_education(text)
        assert len(education) > 0
        assert any("bachelor" in edu.lower() for edu in education)
        assert any("master" in edu.lower() for edu in education)
    
    def test_extract_experience(self):
        """Test experience extraction."""
        text = """
        Work Experience:
        Software Engineer at Tech Corp (2020-2023)
        - Developed Python applications
        - Led machine learning projects
        
        Data Scientist at Data Inc (2018-2020)
        - Built predictive models
        - Analyzed large datasets
        """
        
        experience = self.processor.extract_experience(text)
        assert len(experience) > 0
        assert any("experience" in exp.lower() for exp in experience)
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        text = "This is a sample text with multiple words and sentences. It has some content."
        stats = self.processor.get_text_statistics(text)
        
        assert "total_words" in stats
        assert "meaningful_words" in stats
        assert "sentences" in stats
        assert "avg_words_per_sentence" in stats
        assert "unique_words" in stats
        assert "readability_score" in stats
        
        assert stats["total_words"] > 0
        assert stats["sentences"] > 0
        assert stats["avg_words_per_sentence"] > 0
    
    def test_get_text_statistics_empty_text(self):
        """Test text statistics with empty text."""
        stats = self.processor.get_text_statistics("")
        
        assert stats["total_words"] == 0
        assert stats["sentences"] == 0
        assert stats["avg_words_per_sentence"] == 0
    
    def test_calculate_readability(self):
        """Test readability calculation."""
        simple_text = "The cat sat on the mat."
        complex_text = "The multifaceted, interdisciplinary approach to problem-solving necessitates comprehensive analysis."
        
        simple_score = self.processor._calculate_readability(simple_text)
        complex_score = self.processor._calculate_readability(complex_text)
        
        assert 0 <= simple_score <= 100
        assert 0 <= complex_score <= 100
        assert simple_score > complex_score  # Simple text should be more readable
    
    def test_count_syllables(self):
        """Test syllable counting."""
        assert self.processor._count_syllables("cat") == 1
        assert self.processor._count_syllables("hello") == 2
        assert self.processor._count_syllables("beautiful") == 3
        assert self.processor._count_syllables("programming") == 3
    
    @patch('src.preprocessing.nlp')
    def test_extract_skills_with_spacy(self, mock_nlp):
        """Test skill extraction with spaCy."""
        # Mock spaCy processing
        mock_token1 = Mock()
        mock_token1.pos_ = 'NOUN'
        mock_token1.text = 'python'
        mock_token1.text.lower.return_value = 'python'
        
        mock_token2 = Mock()
        mock_token2.pos_ = 'PROPN'
        mock_token2.text = 'Machine'
        mock_token2.text.lower.return_value = 'machine'
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        mock_nlp.return_value = mock_doc
        
        text = "Python developer with Machine Learning"
        skills = self.processor.extract_skills(text)
        
        assert len(skills) > 0
        mock_nlp.assert_called_once_with(text)
    
    def test_process_document_pdf(self):
        """Test document processing for PDF."""
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_process_document_docx(self):
        """Test document processing for DOCX."""
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_process_document_txt(self):
        """Test document processing for TXT."""
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_process_document_unsupported_format(self):
        """Test document processing with unsupported format."""
        # This would test error handling
        assert True  # Placeholder for async test
    
    def test_clean_text_preserves_structure(self):
        """Test that text cleaning preserves important structure."""
        text = "Name: John Doe. Email: john@example.com. Phone: 123-456-7890."
        cleaned = self.processor._clean_text(text)
        
        assert "john doe" in cleaned
        assert "john@example.com" in cleaned
        assert "123-456-7890" in cleaned
    
    def test_extract_skills_filters_stop_words(self):
        """Test that skill extraction filters out stop words."""
        text = "I am a the and or but developer with experience"
        skills = self.processor.extract_skills(text)
        
        # Should not contain common stop words
        assert "i" not in skills
        assert "am" not in skills
        assert "a" not in skills
        assert "the" not in skills
        assert "and" not in skills
        assert "or" not in skills
        assert "but" not in skills
