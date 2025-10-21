"""
Document preprocessing utilities for resume matching.
"""

import logging
import io
from typing import Optional, Dict, Any
from pathlib import Path
import re

# Document processing libraries
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

from .config import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

# Initialize spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class DocumentProcessor:
    """
    Document processor for extracting and cleaning text from various file formats.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stop words for resume processing
        self.stop_words.update([
            'resume', 'cv', 'curriculum', 'vitae', 'personal', 'information',
            'contact', 'phone', 'email', 'address', 'linkedin', 'github'
        ])
    
    async def process_document(
        self, content: bytes, file_extension: str
    ) -> str:
        """
        Process a document and extract clean text.
        
        Args:
            content: File content as bytes
            file_extension: File extension (pdf, docx, txt)
        
        Returns:
            Cleaned text content
        """
        try:
            # Extract text based on file type
            if file_extension == 'pdf':
                text = self._extract_pdf_text(content)
            elif file_extension in ['doc', 'docx']:
                text = self._extract_docx_text(content)
            elif file_extension == 'txt':
                text = content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text content
        
        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove special characters but keep alphanumeric and basic punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Convert to lowercase
            text = text.lower()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise
    
    def extract_skills(self, text: str) -> list[str]:
        """
        Extract skills from text using NLP techniques.
        
        Args:
            text: Input text
        
        Returns:
            List of extracted skills
        """
        try:
            skills = []
            
            if nlp:
                # Use spaCy for named entity recognition
                doc = nlp(text)
                
                # Extract technical skills (nouns and proper nouns)
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        len(token.text) > 2 and 
                        token.text.lower() not in self.stop_words):
                        skills.append(token.text.lower())
            else:
                # Fallback to simple keyword extraction
                words = word_tokenize(text)
                skills = [
                    word.lower() for word in words
                    if (len(word) > 2 and 
                        word.lower() not in self.stop_words and
                        word.isalpha())
                ]
            
            # Remove duplicates and return
            return list(set(skills))
            
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []
    
    def extract_education(self, text: str) -> list[str]:
        """
        Extract education information from text.
        
        Args:
            text: Input text
        
        Returns:
            List of education entries
        """
        try:
            education_keywords = [
                'bachelor', 'master', 'phd', 'doctorate', 'degree',
                'university', 'college', 'institute', 'school',
                'bsc', 'msc', 'mba', 'ba', 'ma'
            ]
            
            sentences = sent_tokenize(text)
            education_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in education_keywords):
                    education_sentences.append(sentence.strip())
            
            return education_sentences
            
        except Exception as e:
            logger.error(f"Error extracting education: {e}")
            return []
    
    def extract_experience(self, text: str) -> list[str]:
        """
        Extract work experience from text.
        
        Args:
            text: Input text
        
        Returns:
            List of experience entries
        """
        try:
            experience_keywords = [
                'experience', 'worked', 'job', 'position', 'role',
                'company', 'employer', 'years', 'responsibilities'
            ]
            
            sentences = sent_tokenize(text)
            experience_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in experience_keywords):
                    experience_sentences.append(sentence.strip())
            
            return experience_sentences
            
        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return []
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about the processed text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with text statistics
        """
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            # Remove stop words for meaningful word count
            meaningful_words = [
                word for word in words 
                if word.lower() not in self.stop_words and word.isalpha()
            ]
            
            return {
                "total_words": len(words),
                "meaningful_words": len(meaningful_words),
                "sentences": len(sentences),
                "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
                "unique_words": len(set(meaningful_words)),
                "readability_score": self._calculate_readability(text)
            }
            
        except Exception as e:
            logger.error(f"Error calculating text statistics: {e}")
            return {}
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate a simple readability score.
        
        Args:
            text: Input text
        
        Returns:
            Readability score (0-100)
        """
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not sentences or not words:
                return 0.0
            
            # Simple readability formula
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            
            # Simplified Flesch Reading Ease
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
