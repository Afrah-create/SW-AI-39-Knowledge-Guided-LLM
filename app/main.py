#!/usr/bin/env python3
"""
Agricultural Recommendation System with AI Integration
Modern AI-powered interface for crop recommendations in Uganda
"""

import os
import json
import logging
import re
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_file
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import transformers for fine-tuned model
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    logger.info(" Transformers library imported successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("️ Transformers library not available. Fine-tuned model features will be disabled.")

# Import huggingface_hub for model downloads
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
    logger.info(" Hugging Face Hub library imported successfully")
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("️ Hugging Face Hub library not available. Will try to load models locally.")

# Import Gemini API for LLM integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info(" Google Generative AI library imported successfully")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("️ Google Generative AI library not available. LLM features will be disabled.")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(" Environment variables loaded from .env file")
except ImportError:
    logger.debug("python-dotenv not available, using system environment variables only")

# Configure Gemini API
llm_model = None
if GEMINI_AVAILABLE:
    try:
        # Try to get API key from environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"API key found: {'Yes' if api_key else 'No'}")
        if api_key:
            genai.configure(api_key=api_key)
            # Try different model names for compatibility - updated for current API
            model_names_to_try = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-1.0-pro',
                'gemini-pro',
                'gemini-1.5-flash-002',
                'gemini-1.5-pro-002'
            ]
            
            llm_model = None
            for model_name in model_names_to_try:
                try:
                    llm_model = genai.GenerativeModel(model_name)
                    logger.info(f" Gemini API configured successfully with {model_name}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load {model_name}: {e}")
                    continue
            
            if llm_model is None:
                # List available models for debugging
                try:
                    models = genai.list_models()
                    available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
                    logger.info(f"Available Gemini models: {available_models}")
                    logger.warning("️ Failed to initialize any predefined Gemini model. Using fallback expert analysis.")
                    
                    # Try to use the first available model
                    if available_models:
                        try:
                            first_model = available_models[0]
                            llm_model = genai.GenerativeModel(first_model)
                            logger.info(f" Successfully initialized with first available model: {first_model}")
                        except Exception as e4:
                            logger.error(f" Failed to initialize with first available model: {e4}")
                            logger.warning("️ Gemini API not available. Using fallback expert analysis.")
                except Exception as e3:
                    logger.error(f"❌ Failed to list models: {e3}")
                    llm_model = None
        else:
            logger.warning("️ GEMINI_API_KEY not found in environment variables")
    except Exception as e:
        logger.error(f" Error configuring Gemini API: {e}")
        llm_model = None

app = Flask(__name__)

class DataLoader:
    """Load and manage agricultural data"""
    
    def __init__(self, data_dir=None, processed_dir=None):
        # Get absolute paths based on the location of this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        
        self.data_dir = data_dir or os.path.join(parent_dir, "data")
        self.processed_dir = processed_dir or os.path.join(parent_dir, "processed")
        self.knowledge_graph = None
        self.dataset_triples = None
        self.literature_triples = None
        self.ugandan_data = None
        
    def load_data(self):
        """Load all agricultural data files - optimized for memory"""
        try:
            logger.info("Loading agricultural data files...")
            
            # Skip large knowledge graph to reduce memory usage
            # Store file path for on-demand loading if needed
            kg_path = os.path.join(self.processed_dir, "unified_knowledge_graph.json")
            self.kg_path = kg_path
            if os.path.exists(kg_path):
                # Don't load the large knowledge graph - just create empty list
                # This prevents loading 175K triples into memory
                self.knowledge_graph = []
                # Get file size for logging
                import os as os_module
                size_mb = os_module.path.getsize(kg_path) / (1024 * 1024)
                logger.info(f" Knowledge graph file found ({size_mb:.1f} MB), loading deferred to save memory")
            else:
                logger.warning(f"️ Knowledge graph not found at {kg_path}")
                self.knowledge_graph = []
            
            # Don't load dataset triples to save memory
            dataset_triples_path = os.path.join(self.processed_dir, "dataset_triples.json")
            if os.path.exists(dataset_triples_path):
                self.dataset_triples = []  # Empty to save memory
                logger.info(" Dataset triples skipped to save memory")
            else:
                self.dataset_triples = []
            
            # Load literature triples (should be small)
            literature_triples_path = os.path.join(self.processed_dir, "literature_triples.json")
            if os.path.exists(literature_triples_path):
                with open(literature_triples_path, 'r', encoding='utf-8') as f:
                    self.literature_triples = json.load(f)
                logger.info(f" Loaded literature triples: {len(self.literature_triples)} triples")
            else:
                self.literature_triples = []
            
            # Load Ugandan dataset (CSV files are usually manageable)
            ugandan_data_path = os.path.join(self.processed_dir, "ugandan_data_cleaned.csv")
            if os.path.exists(ugandan_data_path):
                import pandas as pd
                # Load only a sample of the data to save memory
                self.ugandan_data = pd.read_csv(ugandan_data_path, nrows=1000)  # Only first 1000 rows
                logger.info(f" Loaded Ugandan dataset sample: {len(self.ugandan_data)} records")
            else:
                self.ugandan_data = None
            
            return True
            
        except Exception as e:
            logger.error(f" Error loading data: {e}")
            return False

class GCNModel(nn.Module):
    """Graph Convolutional Network for agricultural recommendations"""
    
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(GCNModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings (required by saved model)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # GCN layers with correct names to match saved model
        self.gcn1 = nn.Linear(embedding_dim, 200)  # First layer: 100 -> 200
        self.gcn2 = nn.Linear(200, embedding_dim)  # Second layer: 200 -> 100
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, entity_ids):
        # Get entity embeddings
        x = self.entity_embeddings(entity_ids)
        
        # Apply GCN layers with correct names
        x = torch.relu(self.gcn1(x))
        x = self.dropout(x)
        x = torch.relu(self.gcn2(x))
        x = self.dropout(x)
        
        # Return embeddings (no output layer in saved model)
        return x

class AgriculturalModelLoader:
    """Load and manage the trained GCN model from Hugging Face"""
    
    # Hugging Face repository ID for graph models
    HF_REPO_ID = "Awongo/soil-crop-recommendation-model"
    
    def __init__(self, models_dir=None):
        # Get absolute paths based on the location of this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        
        # Keep models_dir for fallback to local loading
        self.models_dir = models_dir or os.path.join(parent_dir, "processed", "trained_models")
        self.model = None
        self.model_metadata = None
        self.entity_to_id = None
        self.id_to_entity = None
        self.relation_to_id = None
        self.id_to_relation = None
        
    def load_model(self):
        """Load the trained model and metadata from Hugging Face"""
        try:
            logger.info("Loading trained agricultural model from Hugging Face...")
            
            # Try loading from Hugging Face first
            if HF_HUB_AVAILABLE:
                try:
                    logger.info(f" Downloading model metadata from {self.HF_REPO_ID}...")
                    metadata_path = hf_hub_download(
                        repo_id=self.HF_REPO_ID,
                        filename="model_metadata.json",
                        cache_dir=None
                    )
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                    
                    self.entity_to_id = self.model_metadata.get('entity_to_id', {})
                    self.id_to_entity = self.model_metadata.get('id_to_entity', {})
                    self.relation_to_id = self.model_metadata.get('relation_to_id', {})
                    self.id_to_relation = self.model_metadata.get('id_to_relation', {})
                    
                    logger.info(f" Loaded model metadata: {len(self.entity_to_id)} entities, {len(self.relation_to_id)} relations")
                    
                    # Download model weights
                    logger.info(" Downloading best_model.pth from Hugging Face...")
                    model_path = hf_hub_download(
                        repo_id=self.HF_REPO_ID,
                        filename="best_model.pth",
                        cache_dir=None
                    )
                    
                    # Create model instance
                    if self.model_metadata:
                        self.model = GCNModel(
                            num_entities=self.model_metadata.get('num_entities', 2513),
                            num_relations=self.model_metadata.get('num_relations', 15),
                            embedding_dim=self.model_metadata.get('embedding_dim', 100)
                        )
                        
                        # Load weights
                        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        self.model.eval()
                        logger.info(" GCN model loaded successfully from Hugging Face")
                        logger.info(f"Model parameters: {self.model_metadata.get('num_entities', 2513)} entities, {self.model_metadata.get('num_relations', 15)} relations, {self.model_metadata.get('embedding_dim', 100)} dim")
                        return True
                    else:
                        logger.error(" Model metadata not loaded")
                        return False
                        
                except Exception as e:
                    logger.warning(f" Failed to load from Hugging Face: {e}")
                    logger.info(" Falling back to local model files...")
                    # Fall through to local loading
            
            # Fallback to local loading
            logger.info("Loading trained agricultural model from local files...")
            metadata_path = os.path.join(self.models_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                self.entity_to_id = self.model_metadata.get('entity_to_id', {})
                self.id_to_entity = self.model_metadata.get('id_to_entity', {})
                self.relation_to_id = self.model_metadata.get('relation_to_id', {})
                self.id_to_relation = self.model_metadata.get('id_to_relation', {})
                
                logger.info(f" Loaded model metadata: {len(self.entity_to_id)} entities, {len(self.relation_to_id)} relations")
                
                # Load model weights
                model_path = os.path.join(self.models_dir, "best_model.pth")
                if os.path.exists(model_path):
                    # Create model instance
                    if self.model_metadata:
                        self.model = GCNModel(
                            num_entities=self.model_metadata.get('num_entities', 2513),
                            num_relations=self.model_metadata.get('num_relations', 15),
                            embedding_dim=self.model_metadata.get('embedding_dim', 100)
                        )
                        
                        # Load weights
                        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        self.model.eval()
                        logger.info(" GCN model loaded successfully from local files")
                        logger.info(f"Model parameters: {self.model_metadata.get('num_entities', 2513)} entities, {self.model_metadata.get('num_relations', 15)} relations, {self.model_metadata.get('embedding_dim', 100)} dim")
                        return True
                    else:
                        logger.error(" Model metadata not loaded")
                        return False
                else:
                    logger.warning(f"️ Model file not found at: {model_path}")
                    return False
            else:
                logger.warning(f"️ Model metadata not found at {metadata_path}")
                return False
            
        except Exception as e:
            logger.error(f" Error loading model: {e}")
            return False

class SemanticRetriever:
    """
    Advanced semantic retrieval system for RAG pipeline
    """
    
    def __init__(self, triples_data, entity_to_id, id_to_entity):
        self.triples_data = triples_data
        self.entity_to_id = entity_to_id
        self.id_to_entity = id_to_entity
        
        # Create TF-IDF vectorizer for hybrid retrieval
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Prepare text representations of triples
        self.triple_texts = self._create_triple_texts()
        
        # Create TF-IDF matrix
        if self.triple_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.triple_texts)
            logger.info(f" TF-IDF matrix created: {self.tfidf_matrix.shape}")
    
    def _create_triple_texts(self):
        """Create text representations of triples for TF-IDF"""
        texts = []
        for triple in self.triples_data:
            # Create descriptive text for each triple
            text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            if 'evidence' in triple:
                text += f" {triple['evidence']}"
            texts.append(text)
        return texts
    
    def hybrid_retrieve(self, query, top_k=15):
        """
        Hybrid retrieval using TF-IDF similarity
        """
        results = []
        
        # TF-IDF similarity
        if hasattr(self, 'tfidf_matrix'):
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        else:
            tfidf_similarities = np.zeros(len(self.triples_data))
        
        # Get top-k results
        top_indices = np.argsort(tfidf_similarities)[::-1][:top_k]
        
        for idx in top_indices:
            if idx < len(self.triples_data) and tfidf_similarities[idx] > 0.05:
                results.append({
                    'triple': self.triples_data[idx],
                    'score': tfidf_similarities[idx],
                    'tfidf_score': tfidf_similarities[idx]
                })
        
        # If no results found, return top results regardless of threshold
        if not results:
            for idx in top_indices[:top_k]:
                if idx < len(self.triples_data):
                    results.append({
                        'triple': self.triples_data[idx],
                        'score': tfidf_similarities[idx],
                        'tfidf_score': tfidf_similarities[idx]
                    })
        
        return results

class AgriculturalConstraintEngine:
    """Rule-based agricultural constraint engine"""
    
    def __init__(self):
        self.crop_constraints = {
            'maize': {
                'pH_range': (5.5, 7.5),
                'organic_matter_min': 1.0,
                'temperature_range': (18, 30),
                'rainfall_range': (500, 1500),
                'soil_textures': ['loam', 'clay_loam', 'sandy_loam'],
                'nitrogen_range': (50, 200),
                'phosphorus_range': (10, 50),
                'potassium_range': (80, 300)
            },
            'rice': {
                'pH_range': (5.0, 7.0),
                'organic_matter_min': 2.0,
                'temperature_range': (20, 35),
                'rainfall_range': (1000, 2500),
                'soil_textures': ['clay', 'clay_loam'],
                'nitrogen_range': (60, 250),
                'phosphorus_range': (15, 60),
                'potassium_range': (100, 400)
            },
            'beans': {
                'pH_range': (6.0, 7.5),
                'organic_matter_min': 1.5,
                'temperature_range': (15, 25),
                'rainfall_range': (600, 1200),
                'soil_textures': ['loam', 'sandy_loam', 'clay_loam'],
                'nitrogen_range': (40, 150),
                'phosphorus_range': (20, 80),
                'potassium_range': (60, 200)
            },
            'cassava': {
                'pH_range': (4.5, 8.0),
                'organic_matter_min': 0.5,
                'temperature_range': (20, 30),
                'rainfall_range': (800, 2000),
                'soil_textures': ['sandy', 'sandy_loam', 'loam'],
                'nitrogen_range': (30, 120),
                'phosphorus_range': (5, 30),
                'potassium_range': (40, 150)
            },
            'sweet_potato': {
                'pH_range': (5.0, 7.5),
                'organic_matter_min': 1.0,
                'temperature_range': (18, 28),
                'rainfall_range': (600, 1500),
                'soil_textures': ['sandy_loam', 'loam'],
                'nitrogen_range': (40, 180),
                'phosphorus_range': (10, 40),
                'potassium_range': (80, 250)
            },
            'coffee': {
                'pH_range': (5.5, 6.5),
                'organic_matter_min': 2.0,
                'temperature_range': (18, 24),
                'rainfall_range': (1200, 2000),
                'soil_textures': ['loam', 'clay_loam'],
                'nitrogen_range': (80, 200),
                'phosphorus_range': (15, 50),
                'potassium_range': (120, 300)
            },
            'cotton': {
                'pH_range': (5.5, 8.0),
                'organic_matter_min': 1.0,
                'temperature_range': (20, 35),
                'rainfall_range': (500, 1200),
                'soil_textures': ['loam', 'sandy_loam', 'clay_loam'],
                'nitrogen_range': (60, 180),
                'phosphorus_range': (10, 40),
                'potassium_range': (80, 200)
            },
            'sugarcane': {
                'pH_range': (5.5, 8.0),
                'organic_matter_min': 1.5,
                'temperature_range': (20, 30),
                'rainfall_range': (1000, 2000),
                'soil_textures': ['loam', 'clay_loam'],
                'nitrogen_range': (100, 300),
                'phosphorus_range': (20, 60),
                'potassium_range': (150, 400)
            }
        }
    
    def evaluate_crop_suitability(self, crop_name, soil_properties, climate_conditions):
        """Evaluate crop suitability based on constraints"""
        if crop_name not in self.crop_constraints:
            return {'suitable': False, 'violations': ['Unknown crop'], 'recommendations': []}
        
        constraints = self.crop_constraints[crop_name]
        violations = []
        recommendations = []
        
        # Check pH
        soil_ph = soil_properties.get('pH', 0)
        ph_min, ph_max = constraints['pH_range']
        if not (ph_min <= soil_ph <= ph_max):
            violations.append(f"pH {soil_ph} outside optimal range ({ph_min}-{ph_max})")
            if soil_ph < ph_min:
                recommendations.append("Add lime to increase soil pH")
            else:
                recommendations.append("Add sulfur to decrease soil pH")
        
        # Check organic matter
        organic_matter = soil_properties.get('organic_matter', 0)
        if organic_matter < constraints['organic_matter_min']:
            violations.append(f"Organic matter {organic_matter}% below minimum {constraints['organic_matter_min']}%")
            recommendations.append("Add compost or organic fertilizers")
        
        # Check soil texture
        texture = soil_properties.get('texture_class', '')
        if texture not in constraints['soil_textures']:
            violations.append(f"Soil texture '{texture}' not optimal for {crop_name}")
            recommendations.append(f"Consider soil amendments for better texture")
        
        # Check temperature
        temperature = climate_conditions.get('temperature_mean', 0)
        temp_min, temp_max = constraints['temperature_range']
        if not (temp_min <= temperature <= temp_max):
            violations.append(f"Temperature {temperature}°C outside optimal range ({temp_min}-{temp_max}°C)")
        
        # Check rainfall
        rainfall = climate_conditions.get('rainfall_mean', 0)
        rain_min, rain_max = constraints['rainfall_range']
        if not (rain_min <= rainfall <= rain_max):
            violations.append(f"Rainfall {rainfall}mm outside optimal range ({rain_min}-{rain_max}mm)")
            if rainfall < rain_min:
                recommendations.append("Consider irrigation systems")
            else:
                recommendations.append("Ensure proper drainage")
        
        # Check soil nutrients
        nitrogen = soil_properties.get('nitrogen', 0)
        phosphorus = soil_properties.get('phosphorus', 0)
        potassium = soil_properties.get('potassium', 0)
        
        if 'nitrogen_range' in constraints:
            n_min, n_max = constraints['nitrogen_range']
            if not (n_min <= nitrogen <= n_max):
                violations.append(f"Nitrogen {nitrogen}ppm outside optimal range ({n_min}-{n_max}ppm)")
                if nitrogen < n_min:
                    recommendations.append("Add nitrogen fertilizer")
                else:
                    recommendations.append("Reduce nitrogen application")
        
        if 'phosphorus_range' in constraints:
            p_min, p_max = constraints['phosphorus_range']
            if not (p_min <= phosphorus <= p_max):
                violations.append(f"Phosphorus {phosphorus}ppm outside optimal range ({p_min}-{p_max}ppm)")
                if phosphorus < p_min:
                    recommendations.append("Add phosphorus fertilizer")
                else:
                    recommendations.append("Reduce phosphorus application")
        
        if 'potassium_range' in constraints:
            k_min, k_max = constraints['potassium_range']
            if not (k_min <= potassium <= k_max):
                violations.append(f"Potassium {potassium}ppm outside optimal range ({k_min}-{k_max}ppm)")
                if potassium < k_min:
                    recommendations.append("Add potassium fertilizer")
                else:
                    recommendations.append("Reduce potassium application")
        
        # Calculate suitability score
        total_checks = 8  # Updated to include nutrients
        violations_count = len(violations)
        suitability_score = max(0, (total_checks - violations_count) / total_checks)
        
        return {
            'suitable': violations_count <= 2,  # Allow up to 2 violations
            'violations': violations,
            'recommendations': recommendations,
            'suitability_score': suitability_score
        }

class FineTunedLLM:
    """Fine-tuned agricultural LLM loaded from Hugging Face"""
    
    # Hugging Face repository ID for fine-tuned LLM
    HF_REPO_ID = "Awongo/agricultural-llm-finetuned"
    # Base model for tokenizer fallback (DialoGPT-small)
    BASE_MODEL_ID = "microsoft/DialoGPT-small"
    
    def __init__(self, model_path=None):
        # Use Hugging Face repo ID if no local path provided
        self.model_path = model_path or self.HF_REPO_ID
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model from Hugging Face"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("️ Transformers not available, cannot load fine-tuned model")
                return False
                
            logger.info(f" Loading fine-tuned model from Hugging Face: {self.model_path}")
            
            # Try multiple loading strategies for compatibility
            try:
                # Strategy 1: Load with trust_remote_code and use_fast=False for compatibility
                logger.info(" Attempting to load tokenizer with standard settings...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                logger.info(" Tokenizer loaded successfully")
            except Exception as tokenizer_error:
                logger.warning(f" Standard tokenizer loading failed: {tokenizer_error}")
                try:
                    # Strategy 2: Load without use_fast
                    logger.info(" Attempting alternative tokenizer loading...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        use_fast=False
                    )
                    logger.info(" Tokenizer loaded with alternative method")
                except Exception as tokenizer_error2:
                    logger.error(f" Alternative tokenizer loading also failed: {tokenizer_error2}")
                    # Try minimal loading
                    try:
                        logger.info(" Attempting minimal tokenizer loading from config...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_path,
                            local_files_only=False,
                            use_fast=False,
                            trust_remote_code=True
                        )
                    except Exception as e:
                        logger.error(f" All tokenizer loading methods from fine-tuned repo failed: {e}")
                        # Final fallback: Use base model tokenizer
                        logger.info(f" Attempting fallback to base model tokenizer: {self.BASE_MODEL_ID}")
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.BASE_MODEL_ID,
                                use_fast=False
                            )
                            logger.info(" Successfully loaded base model tokenizer as fallback")
                        except Exception as base_error:
                            logger.error(f" Base model tokenizer loading also failed: {base_error}")
                            raise
            
            # Load model with memory optimizations
            try:
                logger.info(" Loading model weights...")
                # Use float16 to reduce memory usage if supported, otherwise float32
                # Chiesa: For CPU, we stick with float32 but use low_cpu_mem_usage
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,  # Optimize memory usage
                    device_map='cpu'  # Explicitly use CPU
                )
                self.model.eval()  # Set to evaluation mode to disable dropout and save memory
                logger.info(" Model weights loaded successfully")
            except MemoryError as mem_error:
                logger.error(f" Memory error loading model weights: {mem_error}")
                raise
            except Exception as model_error:
                logger.error(f" Model loading failed: {model_error}")
                raise
            
            # Configure tokenizer pad token if needed
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(" Set pad_token to eos_token")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info(" Added [PAD] token")
            
            logger.info(" Fine-tuned model loaded successfully from Hugging Face!")
            return True
        except Exception as e:
            logger.error(f" Error loading fine-tuned model: {e}")
            import traceback
            logger.error(f" Full traceback: {traceback.format_exc()}")
            logger.warning(" Fine-tuned model will not be available, using fallback analysis")
            return False
    
    def generate_response(self, prompt, max_length=None, max_new_tokens=None, temperature=0.7, num_beams=1, repetition_penalty=1.0):
        """Generate response using fine-tuned model with improved parameters"""
        try:
            if self.model is None or self.tokenizer is None:
                return "Fine-tuned model not available."
            
            # Limit prompt length to avoid slow processing
            max_prompt_tokens = 256
            # Tokenize input with truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_prompt_tokens
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            logger.info(f" Generating with input length: {input_ids.shape[1]} tokens")
            
            with torch.no_grad():
                # Optimize for speed - use greedy decoding
                generation_params = {
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'repetition_penalty': repetition_penalty,
                    'no_repeat_ngram_size': 2,  # Reduced from 3 for speed
                    'do_sample': False  # Greedy decoding - fastest
                }
                
                # Use max_new_tokens if specified (better for long prompts)
                # Aggressively reduce for CPU inference speed
                if max_new_tokens is not None:
                    generation_params['max_new_tokens'] = min(max_new_tokens, 50)  # Cap at 50 for CPU speed
                elif max_length is not None:
                    generation_params['max_length'] = min(max_length, input_ids.shape[1] + 50)
                else:
                    generation_params['max_new_tokens'] = 40  # Reduced to 40 for faster CPU generation
                
                # Force greedy decoding for speed (ignore num_beams > 1)
                if num_beams > 1 and max_new_tokens and max_new_tokens <= 60:
                    # Only use beam search for very short generations
                    generation_params.update({
                        'num_beams': min(num_beams, 2),  # Max 2 beams
                        'early_stopping': True,
                        'do_sample': False
                    })
                    logger.info(" Using beam search (limited)")
                else:
                    # Always use greedy decoding for speed
                    generation_params['do_sample'] = False
                    logger.info(" Using greedy decoding for speed")
                
                logger.info(f" Generation params: max_new_tokens={generation_params.get('max_new_tokens', 'N/A')}")
                logger.info(" Starting model.generate()...")
                
                # Add early stopping and max time limit
                start_time = datetime.now()
                
                # Use torch.no_grad() and memory-efficient inference
                with torch.no_grad():
                    # Enable inference mode for better memory efficiency
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            **generation_params
                        )
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f" Generation complete in {elapsed:.2f}s, output length: {outputs.shape[1]}")
                
                # Clear cache after generation to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    # Force garbage collection for CPU memory
                    import gc
                    gc.collect()
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            logger.info(f" Generation complete, response length: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f" Error generating response: {e}")
            import traceback
            logger.error(f" Traceback: {traceback.format_exc()}")
            return None

class AgriculturalAPI:
    """Agricultural recommendation API with real data integration"""
    
    def __init__(self):
        self.models_loaded = False
        self.data_loaded = False
        self.rag_loaded = False
        
        # Initialize constraint engine (always available)
        self.constraint_engine = AgriculturalConstraintEngine()
        
        # Initialize model loader (for GCN embeddings)
        self.model_loader = AgriculturalModelLoader()
        
        # Don't load heavy data files or knowledge graph
        self.data_loader = None
        self.semantic_retriever = None
        self.finetuned_llm = None
        
        # Load model lazily on first request to save startup memory
        self._initialized = False
        
        logger.info(" AgriculturalAPI initialized (lazy model loading)")
    
    def _ensure_loaded(self):
        """Lazy load models on first request"""
        if not self._initialized:
            logger.info(" Loading GCN model on first request...")
            # Load only the GCN model (small, just weights)
            if self.model_loader.load_model():
                self.models_loaded = True
                logger.info(" GCN model loaded successfully")
            else:
                logger.warning(" GCN model not available, using constraint-based recommendations only")
                self.models_loaded = False
            
            # Try to load fine-tuned LLM (optional, can be disabled via environment variable for memory constraints)
            # Check if fine-tuned model is disabled via environment variable
            disable_finetuned = os.getenv('DISABLE_FINETUNED_MODEL', 'true').lower() == 'true'  # Default to disabled for memory
            
            if disable_finetuned:
                logger.info(" Fine-tuned LLM disabled via DISABLE_FINETUNED_MODEL environment variable (memory optimization)")
                self.finetuned_llm = None
            elif TRANSFORMERS_AVAILABLE and self.finetuned_llm is None:
                try:
                    logger.info(" Attempting to load fine-tuned LLM from Hugging Face...")
                    # Try loading but catch memory errors
                    self.finetuned_llm = FineTunedLLM()  # Will use Hugging Face repo by default
                    if self.finetuned_llm.model is None:
                        logger.warning(" Fine-tuned LLM not available, will use fallback analysis")
                        self.finetuned_llm = None
                    else:
                        logger.info(" Fine-tuned LLM loaded successfully")
                except MemoryError as mem_error:
                    logger.error(f" Memory error loading fine-tuned LLM: {mem_error}. Disabling to save memory.")
                    self.finetuned_llm = None
                except Exception as e:
                    logger.warning(f" Failed to load fine-tuned LLM: {e}. Using fallback analysis.")
                    self.finetuned_llm = None
            else:
                self.finetuned_llm = None
            
            self.data_loaded = True  # Mark as loaded (even though we skip heavy data)
            self._initialized = True
    
    def get_recommendation(self, soil_properties, climate_conditions, **kwargs):
        """Get crop recommendation based on soil and climate data"""
        # Lazy load model if not already loaded
        self._ensure_loaded()
        
        try:
            logger.info(f"Generating recommendation for soil: {soil_properties}, climate: {climate_conditions}")
            
            # Get farming conditions
            farming_conditions = kwargs.get('farming_conditions', {})
            available_land = kwargs.get('available_land', 0)
            
            # Get all available crops
            available_crops = list(self.constraint_engine.crop_constraints.keys())
            
            # Evaluate each crop
            suitable_crops = []
            for crop in available_crops:
                evaluation = self.constraint_engine.evaluate_crop_suitability(
                    crop, soil_properties, climate_conditions
                )
                
                # Apply additional farming condition filters
                if self._evaluate_farming_conditions(crop, farming_conditions):
                    suitable_crops.append({
                        'crop': crop,
                        'suitability_score': evaluation['suitability_score'],
                        'recommendations': evaluation['recommendations'],
                        'violations': evaluation['violations'],
                        'farming_factors': self._get_farming_factors(crop, farming_conditions)
                    })
            
            # Enhance with GCN model if available
            if self.models_loaded and self.model_loader.model is not None:
                logger.info(" Enhancing with GCN model embeddings...")
                suitable_crops = self._enhance_with_gcn_model(suitable_crops, soil_properties, climate_conditions)
                enhanced_count = sum(1 for c in suitable_crops if c.get('model_enhanced', False))
                logger.info(f" GCN model enhanced {enhanced_count}/{len(suitable_crops)} crops")
            else:
                logger.warning(" GCN model not available - using constraint-based recommendations only")
            
            # Sort by suitability score
            suitable_crops.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            # Limit to top 6 crops for cleaner display
            suitable_crops = suitable_crops[:6]
            
            # Generate land allocation if available land is provided
            land_allocation = None
            if available_land > 0:
                land_allocation = self._generate_land_allocation(suitable_crops, available_land)
            
            # Generate evaluation scores
            evaluation_scores = self._generate_evaluation(suitable_crops)
            
            # Generate recommendation text
            recommendation_text = self._generate_recommendation_text(suitable_crops, soil_properties, climate_conditions)
            
            # Get data sources information
            data_sources = self._get_data_sources()
            
            return {
                'suitable_crops': suitable_crops,
                'land_allocation': land_allocation,
                'evaluation_scores': evaluation_scores,
                'recommendation_text': recommendation_text,
                'recommendation_sections': self._get_structured_recommendation_sections(suitable_crops, soil_properties, climate_conditions),
                'data_sources': data_sources,
                'farming_conditions': farming_conditions
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendation: {e}")
            raise e
    
    def _enhance_with_gcn_model(self, suitable_crops, soil_properties, climate_conditions):
        """Enhance recommendations with GCN model embeddings"""
        try:
            if not self.model_loader.model or not self.model_loader.model_metadata:
                logger.warning(" GCN model or metadata not available for enhancement")
                return suitable_crops  # Return as-is if no model
            
            enhanced_crops = []
            entity_to_id = self.model_loader.entity_to_id
            model = self.model_loader.model
            
            logger.info(f" Attempting to enhance {len(suitable_crops)} crops with GCN model")
            logger.info(f" Entity mappings available: {len(entity_to_id)} entities")
            
            # Sample some entity names to understand the format
            sample_entities = list(entity_to_id.keys())[:10]
            logger.info(f" Sample entity names: {sample_entities}")
            
            for crop in suitable_crops:
                enhanced_crop = crop.copy()
                
                # Try to get entity ID for this crop
                crop_name = crop['crop'].lower()
                
                # Look for crop in entity mappings (flexible matching)
                crop_id = None
                crop_entity = None
                search_variations = [crop_name]
                
                # Try exact match first
                if crop_name in entity_to_id:
                    crop_id = entity_to_id[crop_name]
                    crop_entity = crop_name
                else:
                    # Try flexible matching with underscore replacement
                    search_variations = [
                        crop_name,
                        crop_name.replace('_', ' '),
                        crop_name.replace('_', '-'),
                        crop_name.replace(' ', '_')
                    ]
                    
                    for variation in search_variations:
                        if variation in entity_to_id:
                            crop_id = entity_to_id[variation]
                            crop_entity = variation
                            break
                
                # If still not found, try matching crop URLs (http://example.org/agrokg/crop/CROP_NAME)
                if crop_id is None:
                    crop_url = f"http://example.org/agrokg/crop/{crop_name}"
                    if crop_url in entity_to_id:
                        crop_id = entity_to_id[crop_url]
                        crop_entity = crop_url
                
                # If still not found, try partial matching
                if crop_id is None:
                    for entity, eid in entity_to_id.items():
                        entity_lower = entity.lower()
                        if crop_name in entity_lower or entity_lower in crop_name:
                            crop_id = eid
                            crop_entity = entity
                            break
                
                if crop_id is not None:
                    try:
                        # Get enhanced embedding from GCN model
                        with torch.no_grad():
                            entity_tensor = torch.tensor([crop_id], dtype=torch.long)
                            embedding = model(entity_tensor)
                            
                            # Use embedding magnitude to adjust score
                            embedding_score = float(torch.norm(embedding).item())
                            # Normalize to 0-1 range and add small boost
                            normalized_score = min(1.0, embedding_score / 10.0) * 0.1
                            enhanced_crop['suitability_score'] = min(1.0, enhanced_crop['suitability_score'] + normalized_score)
                            enhanced_crop['model_enhanced'] = True
                            logger.info(f" ✅ Enhanced {crop['crop']} → {crop_entity} (boost: {normalized_score:.4f})")
                    except Exception as e:
                        logger.warning(f" Could not enhance {crop['crop']} with model: {e}")
                        enhanced_crop['model_enhanced'] = False
                else:
                    enhanced_crop['model_enhanced'] = False
                    logger.debug(f" ❌ Crop '{crop['crop']}' not found in entity mappings (searched: {search_variations})")
                
                enhanced_crops.append(enhanced_crop)
            
            return enhanced_crops
            
        except Exception as e:
            logger.error(f" Error enhancing with GCN model: {e}")
            return suitable_crops  # Return original if enhancement fails
    
    def _generate_land_allocation(self, suitable_crops, available_land):
        """Generate optimal land allocation plan"""
        if not suitable_crops or available_land <= 0:
            return None
        
        # Simple allocation based on suitability scores
        total_score = sum(crop['suitability_score'] for crop in suitable_crops)
        crop_details = []
        remaining_land = available_land
                
        for crop in suitable_crops[:5]:  # Top 5 crops
            if remaining_land <= 0:
                break
                
            # Allocate land proportional to suitability score
            allocation_ratio = crop['suitability_score'] / total_score
            allocated_land = min(remaining_land * allocation_ratio, remaining_land)
                
            if allocated_land > 0.1:  # Minimum 0.1 hectares
                crop_details.append({
                    'crop': crop['crop'],
                    'land_allocated': round(allocated_land, 2),
                    'suitability_score': crop['suitability_score']
                })
                remaining_land -= allocated_land
        
        return {
            'total_land_used': available_land - remaining_land,
            'crop_details': crop_details
        }
    
    def _generate_evaluation(self, suitable_crops):
        """Generate evaluation scores"""
        if not suitable_crops:
            return {
                'overall_score': 0.0,
                'dimension_scores': {
                    'economic': 0.0,
                    'environmental': 0.0,
                    'social': 0.0,
                    'risk': 0.0
                }
            }
        
        # Calculate dimension scores
        avg_suitability = sum(crop['suitability_score'] for crop in suitable_crops) / len(suitable_crops)
        
        # Economic score based on high-value crops
        high_value_crops = ['coffee', 'cotton', 'sugarcane']
        economic_score = 0.5 + (0.3 if any(crop['crop'] in high_value_crops for crop in suitable_crops) else 0.0)
        
        # Environmental score based on crop diversity
        environmental_score = 0.5 + (0.2 if len(suitable_crops) >= 3 else 0.0)
        
        # Social score based on staple crops
        staple_crops = ['maize', 'rice', 'cassava', 'sweet_potato', 'beans']
        social_score = 0.5 + (0.3 if any(crop['crop'] in staple_crops for crop in suitable_crops) else 0.0)
        
        # Risk score based on diversification
        risk_score = 0.5 + (0.2 if len(suitable_crops) >= 3 else 0.0)
        
        # Overall score
        overall_score = (economic_score + environmental_score + social_score + risk_score) / 4
        
        return {
            'overall_score': overall_score,
            'dimension_scores': {
                'economic': economic_score,
                'environmental': environmental_score,
                'social': social_score,
                'risk': risk_score
            }
        }
    
    def _generate_recommendation_text(self, suitable_crops, soil_properties, climate_conditions):
        """Generate unified, well-structured recommendation combining all AI sources"""
        if not suitable_crops:
            return "No suitable crops found for your current conditions. Consider soil amendments and management practices."
        
        # Create unified recommendation structure
        unified_recommendation = self._create_unified_recommendation(suitable_crops, soil_properties, climate_conditions)
        
        return unified_recommendation
    
    def _create_unified_recommendation(self, suitable_crops, soil_properties, climate_conditions):
        """Create a unified, well-structured recommendation"""
        top_crop = suitable_crops[0]

        # Create clean, professional recommendation
        recommendation = f"AGRICULTURAL RECOMMENDATION REPORT\n\n"
        
        # Executive summary with better formatting
        recommendation += f"PRIMARY RECOMMENDATION: {top_crop['crop'].title()}\n"
        recommendation += f"SUITABILITY SCORE: {top_crop['suitability_score']:.1%}\n\n"
        
        recommendation += f"Based on comprehensive analysis of your soil and climate conditions, {top_crop['crop'].title()} emerges as the most suitable crop for your farm. This recommendation is derived from detailed evaluation of soil properties, climate conditions, and agricultural best practices specific to Uganda's farming environment.\n\n"

        # AI analysis section
        ai_insights = self._get_ai_insights(suitable_crops, soil_properties, climate_conditions)
        if ai_insights:
            recommendation += f"\n**AI ANALYSIS**\n"
            recommendation += f"{ai_insights}\n\n"

        # Expert analysis section
        expert_analysis = self._get_expert_analysis(suitable_crops, soil_properties, climate_conditions)
        if expert_analysis:
            recommendation += f"\n**EXPERT ANALYSIS**\n"
            recommendation += f"{expert_analysis}\n\n"

        # Technical analysis section
        technical_analysis = self._get_technical_analysis(suitable_crops, soil_properties, climate_conditions)
        recommendation += f"\n**TECHNICAL ANALYSIS**\n"
        recommendation += f"{technical_analysis}\n\n"

        # Implementation plan section
        implementation_plan = self._get_implementation_plan(suitable_crops, soil_properties, climate_conditions)
        recommendation += f"\n**IMPLEMENTATION PLAN**\n"
        recommendation += f"{implementation_plan}\n\n"

        return recommendation
    
    def _get_ai_insights(self, suitable_crops, soil_properties, climate_conditions):
        """Get AI-generated insights from fine-tuned model with timeout protection"""
        if self.finetuned_llm and self.finetuned_llm.model is not None:
            try:
                # Try to get insights from fine-tuned model with timeout
                # Use threading for timeout (works on all platforms)
                import threading
                result = [None]
                exception = [None]
                
                def generate_with_timeout():
                    try:
                        result[0] = self._generate_finetuned_recommendation(suitable_crops, soil_properties, climate_conditions)
                    except Exception as e:
                        exception[0] = e
                
                # Start generation in separate thread
                thread = threading.Thread(target=generate_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(timeout=10)  # 10 second timeout (reduced from 15 for faster fallback)
                
                if thread.is_alive():
                    logger.warning(" Fine-tuned model generation timed out after 10s, using fallback")
                    # Model generation is still running but we'll use fallback
                    return None
                
                if exception[0]:
                    raise exception[0]
                
                if result[0]:
                    logger.info(" Fine-tuned model provided insights")
                    return result[0]
                else:
                    logger.warning(" Fine-tuned model returned empty result")
                    return None
                    
            except Exception as e:
                logger.warning(f"Fine-tuned model generation failed: {e}")
                import traceback
                logger.debug(f" Traceback: {traceback.format_exc()}")
        
        # Improved fallback to structured AI insights with contextual analysis
        insights_parts = []
        
        # Crop recommendation
        if suitable_crops:
            top_crop = suitable_crops[0]
            crop_name = top_crop['crop'].title()
            score = top_crop['suitability_score']
            
            # Determine suitability level
            if score >= 0.80:
                suitability_desc = "excellent compatibility"
            elif score >= 0.60:
                suitability_desc = "strong suitability"
            elif score >= 0.40:
                suitability_desc = "moderate suitability"
            else:
                suitability_desc = "acceptable compatibility"
            
            insights_parts.append(f"Analysis indicates {crop_name} shows {suitability_desc} ({(score*100):.0f}%) with your conditions. ")
            
            if len(suitable_crops) > 1:
                alt_count = len(suitable_crops) - 1
                insights_parts.append(f"Additionally, {alt_count} alternative crop{'s' if alt_count > 1 else ''} show{'s' if alt_count == 1 else ''} strong potential. ")
            else:
                insights_parts.append("Multiple crops are suitable for your conditions with proper management. ")
        
        # Soil pH analysis
        ph = soil_properties.get('pH')
        if ph:
            if ph < 5.5:
                insights_parts.append(f"Soil pH is highly acidic ({ph}), requiring urgent liming. Apply 2-3 tons of agricultural lime per hectare to raise pH to 6.0-6.5 for optimal nutrient availability. ")
            elif ph < 6.0:
                insights_parts.append(f"Soil pH is moderately acidic ({ph}). Apply 1-2 tons of lime per hectare and incorporate organic matter to improve soil buffering capacity. ")
            elif 6.0 <= ph <= 7.5:
                insights_parts.append(f"Soil pH ({ph}) is within the optimal range for most crops, promoting efficient nutrient uptake. ")
            elif ph <= 8.0:
                insights_parts.append(f"Soil pH is slightly alkaline ({ph}). Organic matter addition and sulfur application can help lower pH if needed for specific crops. ")
            else:
                insights_parts.append(f"Soil pH is strongly alkaline ({ph}), which may limit micronutrient availability. Consider acidifying amendments and focus on alkaline-tolerant crops. ")
        
        # Organic matter analysis
        om = soil_properties.get('organic_matter')
        if om:
            if om < 2.0:
                insights_parts.append(f"Organic matter is low ({om}%), indicating poor soil health. Implement composting, cover cropping, and reduced tillage to build organic matter to at least 3-4%. ")
            elif om < 3.5:
                insights_parts.append(f"Organic matter ({om}%) is below optimal. Incorporate crop residues and manure to enhance soil structure and water retention. ")
            else:
                insights_parts.append(f"Organic matter ({om}%) is adequate, supporting good soil structure and microbial activity. ")
        
        # Nutrient analysis (NPK)
        nitrogen = soil_properties.get('nitrogen', 0)
        phosphorus = soil_properties.get('phosphorus', 0)
        potassium = soil_properties.get('potassium', 0)
        
        nutrient_issues = []
        if nitrogen < 20:
            nutrient_issues.append("N-deficient")
        elif nitrogen > 80:
            nutrient_issues.append("excessive N")
        
        if phosphorus < 10:
            nutrient_issues.append("P-deficient")
        elif phosphorus > 60:
            nutrient_issues.append("high P")
        
        if potassium < 100:
            nutrient_issues.append("K-deficient")
        elif potassium > 400:
            nutrient_issues.append("excessive K")
        
        if nutrient_issues:
            insights_parts.append(f"Nutrient status shows {', '.join(nutrient_issues).replace('-', ' ')} with levels of N:{nitrogen}, P:{phosphorus}, K:{potassium} ppm. ")
        else:
            insights_parts.append(f"Nutrient levels are balanced (N:{nitrogen}, P:{phosphorus}, K:{potassium} ppm), supporting healthy crop growth. ")
        
        # Climate analysis
        temp = climate_conditions.get('temperature_mean')
        rainfall = climate_conditions.get('rainfall_mean')
        
        if temp and rainfall:
            if temp < 15:
                insights_parts.append(f"Cool temperatures ({temp}°C) favor cool-season crops and early planting is essential. ")
            elif temp > 30:
                insights_parts.append(f"High temperatures ({temp}°C) require heat-tolerant varieties and adequate irrigation. ")
            else:
                insights_parts.append(f"Temperature ({temp}°C) is favorable for most tropical crops. ")
            
            if rainfall < 500:
                insights_parts.append(f"Low rainfall ({rainfall}mm) necessitates irrigation planning and drought-resistant varieties. ")
            elif rainfall > 1500:
                insights_parts.append(f"High rainfall ({rainfall}mm) requires good drainage and disease-resistant cultivars. ")
            else:
                insights_parts.append(f"Rainfall pattern ({rainfall}mm) is suitable for diverse cropping. ")
        
        # Soil texture analysis
        texture = soil_properties.get('texture_class', '').lower()
        if texture:
            texture_benefits = {
                'clay': 'high water retention but requires drainage management',
                'loamy': 'excellent structure and nutrient-holding capacity',
                'sandy': 'good drainage but needs frequent irrigation and organic matter',
                'silty': 'good water-holding with moderate drainage'
            }
            if texture in texture_benefits:
                insights_parts.append(f"Soil texture is {texture}, providing {texture_benefits[texture]}. ")
        
        # Management recommendations based on analysis
        recommendations = []
        if ph and ph < 6.0:
            recommendations.append("liming")
        if om and om < 3.0:
            recommendations.append("organic matter enhancement")
        if nitrogen < 30 or phosphorus < 15 or potassium < 150:
            recommendations.append("balanced fertilization")
        if rainfall and rainfall < 600:
            recommendations.append("water conservation practices")
        
        if recommendations:
            insights_parts.append(f"Key priorities: implement {' and '.join(recommendations)} to optimize production potential. ")
        
        # Combine all insights
        insights = ''.join(insights_parts)
        
        # Remove any trailing spaces and ensure proper ending
        insights = insights.strip()
        if not insights.endswith('.'):
            insights += '.'
        
        return insights
    
    def _get_expert_analysis(self, suitable_crops, soil_properties, climate_conditions):
        """Get expert analysis from Gemini API or fallback"""
        if llm_model:
            try:
                analysis = self._generate_llm_recommendation(suitable_crops, soil_properties, climate_conditions)
                logger.info(" Gemini API provided expert analysis")
                return analysis
            except Exception as e:
                logger.warning(f"Gemini API generation failed: {e}")
        
        # Use expert fallback
        return self._generate_expert_fallback(suitable_crops, soil_properties, climate_conditions)
    
    def _get_technical_analysis(self, suitable_crops, soil_properties, climate_conditions):
        """Get detailed technical analysis in paragraph format"""
        analysis = f"Comprehensive soil analysis reveals a pH level of {soil_properties.get('pH', 'Unknown')} with organic matter content of {soil_properties.get('organic_matter', 'Unknown')}%. The soil profile is characterized as {soil_properties.get('texture_class', 'Unknown').title()} texture, exhibiting nutrient concentrations of {soil_properties.get('nitrogen', 'Unknown')} ppm nitrogen, {soil_properties.get('phosphorus', 'Unknown')} ppm phosphorus, and {soil_properties.get('potassium', 'Unknown')} ppm potassium. "
        
        analysis += f"Environmental conditions present an average temperature of {climate_conditions.get('temperature_mean', 'Unknown')}°C with annual precipitation of {climate_conditions.get('rainfall_mean', 'Unknown')}mm, creating a microclimate conducive to agricultural productivity. "
        
        if suitable_crops:
            top_crop = suitable_crops[0]
            analysis += f"Through advanced algorithmic analysis, {top_crop['crop'].title()} demonstrates superior compatibility with your specific conditions, achieving a suitability rating of {top_crop['suitability_score']:.1%}. "
            
            if len(suitable_crops) > 1:
                alternatives = []
                for crop in suitable_crops[1:4]:
                    alternatives.append(f"{crop['crop'].title()} ({crop['suitability_score']:.1%})")
                analysis += f"Secondary crop recommendations include {', '.join(alternatives)}, providing viable alternatives for crop rotation and diversification strategies."
        else:
            analysis += "Through advanced algorithmic analysis, multiple crop options are available for your specific conditions."
        
        return analysis
    
    def _get_implementation_plan(self, suitable_crops, soil_properties, climate_conditions):
        """Get implementation plan in paragraph format"""
        if suitable_crops:
            top_crop = suitable_crops[0]
            plan = f"To achieve optimal {top_crop['crop'].title()} cultivation success, a comprehensive implementation strategy is essential. "
            
            if top_crop['recommendations']:
                recommendations_text = ', '.join(top_crop['recommendations'][:3])
                plan += f"Priority actions include {recommendations_text} to establish optimal growing conditions. "
            
            if top_crop['violations']:
                violations_text = ', '.join(top_crop['violations'][:3])
                plan += f"Critical management areas requiring immediate attention include {violations_text} to prevent yield limitations. "
        else:
            plan = "To achieve optimal agricultural success, a comprehensive implementation strategy is essential. "
        
        plan += f"Implementing a systematic monitoring protocol for soil health, nutrient levels, and crop development will ensure sustained productivity. Establishing a strategic crop rotation schedule will enhance soil fertility, minimize pest pressure, and optimize long-term agricultural sustainability."
        
        return plan
    
    def _is_similar_content(self, text1, text2, threshold=0.7):
        """Check if two texts are similar to avoid duplication"""
        if not text1 or not text2:
            return False
        
        # Simple similarity check based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if len(union) > 0 else 0
        return similarity > threshold
    
    def _clean_repetitive_text(self, text):
        """Clean repetitive phrases from fine-tuned model output"""
        if not text:
            return text
        
        # Common repetitive phrases to remove
        repetitive_phrases = [
            "This recommendation is based on agricultural research and local conditions in Uganda.",
            "based on agricultural research and local conditions in Uganda",
            "agricultural research and local conditions in Uganda",
            "local conditions in Uganda",
            "in Uganda",
            "This recommendation is based on agricultural",
            "based on agricultural",
            "agricultural research",
            "research and local conditions",
            "local conditions",
            "conditions in Uganda",
            "Uganda.",
            "Uganda",
        ]
        
        # Remove repetitive phrases
        cleaned_text = text
        for phrase in repetitive_phrases:
            # Remove multiple occurrences
            while phrase in cleaned_text:
                cleaned_text = cleaned_text.replace(phrase, "").strip()
        
        # Remove incomplete sentences at the end
        sentences = cleaned_text.split('.')
        if len(sentences) > 1:
            # Check if last sentence is incomplete (less than 10 characters)
            last_sentence = sentences[-1].strip()
            if len(last_sentence) < 10:
                sentences = sentences[:-1]
            cleaned_text = '. '.join(sentences).strip()
        
        # Remove bullet points and convert to paragraph format
        cleaned_text = cleaned_text.replace("- ", "").replace("• ", "").replace("* ", "")
        
        # Remove extra whitespace and periods
        cleaned_text = " ".join(cleaned_text.split())
        cleaned_text = cleaned_text.replace("..", ".").replace("  ", " ")
        
        # Ensure it ends with a period
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'
        
        return cleaned_text
    
    def generate_pdf_report(self, suitable_crops, soil_properties, climate_conditions, recommendation_text):
        """Generate a professionally formatted PDF report"""
        try:
            # Create a BytesIO buffer to hold the PDF
            buffer = io.BytesIO()
            
            # Create PDF document
            doc = SimpleDocTemplate(buffer, pagesize=A4, 
                                  rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkgreen
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leftIndent=0,
                rightIndent=0
            )
            
            # Build the PDF content
            story = []
            
            # Title
            story.append(Paragraph("AGRICULTURAL RECOMMENDATION REPORT", title_style))
            story.append(Spacer(1, 20))
            
            # Date and farm info
            current_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"<b>Report Date:</b> {current_date}", body_style))
            story.append(Spacer(1, 10))
            
            # Primary recommendation
            top_crop = None
            if suitable_crops:
                top_crop = suitable_crops[0]
                story.append(Paragraph(f"<b>PRIMARY RECOMMENDATION:</b> {top_crop['crop'].title()}", heading_style))
                story.append(Paragraph(f"<b>SUITABILITY SCORE:</b> {top_crop['suitability_score']:.1%}", body_style))
                story.append(Spacer(1, 15))
            
            # Executive summary
            story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
            if top_crop:
                summary_text = f"Based on comprehensive analysis of your soil and climate conditions, {top_crop['crop'].title()} emerges as the most suitable crop for your farm. This recommendation is derived from detailed evaluation of soil properties, climate conditions, and agricultural best practices specific to Uganda's farming environment."
            else:
                summary_text = "Based on comprehensive analysis of your soil and climate conditions, agricultural recommendations have been generated. This recommendation is derived from detailed evaluation of soil properties, climate conditions, and agricultural best practices specific to Uganda's farming environment."
            story.append(Paragraph(summary_text, body_style))
            story.append(Spacer(1, 15))
            
            # Soil and climate data table
            story.append(Paragraph("FARM CONDITIONS ANALYSIS", heading_style))
            
            # Create data table
            data = [
                ['Parameter', 'Value', 'Status'],
                ['Soil pH', f"{soil_properties.get('pH', 'N/A')}", 'Optimal' if 6.0 <= soil_properties.get('pH', 0) <= 7.5 else 'Needs Attention'],
                ['Organic Matter', f"{soil_properties.get('organic_matter', 'N/A')}%", 'Good' if soil_properties.get('organic_matter', 0) >= 2.0 else 'Low'],
                ['Soil Texture', f"{soil_properties.get('texture_class', 'N/A').title()}", 'Suitable'],
                ['Nitrogen', f"{soil_properties.get('nitrogen', 'N/A')} ppm", 'Adequate' if soil_properties.get('nitrogen', 0) >= 50 else 'Low'],
                ['Phosphorus', f"{soil_properties.get('phosphorus', 'N/A')} ppm", 'Adequate' if 10 <= soil_properties.get('phosphorus', 0) <= 50 else 'Needs Adjustment'],
                ['Potassium', f"{soil_properties.get('potassium', 'N/A')} ppm", 'Adequate' if soil_properties.get('potassium', 0) >= 100 else 'Low'],
                ['Temperature', f"{climate_conditions.get('temperature_mean', 'N/A')}°C", 'Optimal'],
                ['Rainfall', f"{climate_conditions.get('rainfall_mean', 'N/A')} mm", 'Adequate' if 500 <= climate_conditions.get('rainfall_mean', 0) <= 2000 else 'Needs Irrigation']
            ]
            
            table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Parse and add recommendation sections
            sections = recommendation_text.split('**')
            current_section = ""
            
            for i, part in enumerate(sections):
                if i % 2 == 0:  # Even index = content
                    if part.strip():
                        story.append(Paragraph(part.strip(), body_style))
                        story.append(Spacer(1, 10))
                else:  # Odd index = section heading
                    if part.strip():
                        story.append(Paragraph(part.strip(), heading_style))
            
            # Alternative crops section
            if len(suitable_crops) > 1:
                story.append(Paragraph("ALTERNATIVE CROP OPTIONS", heading_style))
                alt_text = "Secondary crop recommendations include "
                alternatives = []
                for crop in suitable_crops[1:4]:
                    alternatives.append(f"{crop['crop'].title()} ({crop['suitability_score']:.1%})")
                alt_text += ", ".join(alternatives) + ", providing viable alternatives for crop rotation and diversification strategies."
                story.append(Paragraph(alt_text, body_style))
                story.append(Spacer(1, 15))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by AgriAI - Smart Agricultural Assistant", 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                             fontSize=9, alignment=TA_CENTER, 
                                             textColor=colors.grey)))
            story.append(Paragraph("Powered by Advanced AI • Optimized for Uganda Agriculture", 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                             fontSize=8, alignment=TA_CENTER, 
                                             textColor=colors.grey)))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF content
            buffer.seek(0)
            pdf_content = buffer.getvalue()
            buffer.close()
            
            return pdf_content
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return None
    
    def _generate_finetuned_recommendation(self, suitable_crops, soil_properties, climate_conditions):
        """Generate recommendation using fine-tuned model with RAG integration"""
        try:
            # Get top crop
            top_crop = suitable_crops[0]['crop'] if suitable_crops else 'maize'
            
            # Retrieve RAG evidence from knowledge graph
            rag_evidence_text = self._get_rag_evidence_for_model(
                top_crop, soil_properties, climate_conditions, suitable_crops
            )
            
            # Build structured prompt with RAG evidence
            prompt = self._build_structured_prompt_with_rag(
                top_crop, soil_properties, climate_conditions, rag_evidence_text, suitable_crops
            )
            
            # Generate response with optimized parameters for speed
            logger.info(" Calling fine-tuned model generation...")
            try:
                response = self.finetuned_llm.generate_response(
                    prompt, 
                    max_new_tokens=30,   # Very short for CPU speed (30 tokens ~ 20-25 words)
                    temperature=0.7,     # Standard (not used with greedy)
                    num_beams=1,         # Greedy decoding - much faster
                    repetition_penalty=1.1  # Lower penalty for speed
                )
            except Exception as gen_error:
                logger.error(f" Generation error in _generate_finetuned_recommendation: {gen_error}")
                response = None
            
            # Clean up response and validate quality
            if response is None:
                logger.warning(" Fine-tuned model returned None, using fallback")
                return self._get_fallback_llm_recommendation(top_crop, soil_properties, climate_conditions)
            
            # Handle empty string responses
            if not response or not isinstance(response, str):
                logger.warning(f" Fine-tuned model returned invalid response type: {type(response)}")
                return self._get_fallback_llm_recommendation(top_crop, soil_properties, climate_conditions)
                
            if len(response.strip()) > 15:
                # Remove the prompt from response if it's included
                if prompt in response:
                    response = response.split(prompt, 1)[-1].strip()
                
                # Remove repetitive phrases
                response = self._clean_repetitive_text(response)
                
                # Check if response is meaningful (not just random numbers/text)
                if self._is_meaningful_response(response):
                    return response.strip()
            
            # Fallback to template-based recommendation if model fails
            return self._get_fallback_llm_recommendation(top_crop, soil_properties, climate_conditions)
                
        except Exception as e:
            logger.error(f"Error generating fine-tuned recommendation: {e}")
            return self._get_fallback_llm_recommendation(top_crop, soil_properties, climate_conditions)
    
    def _is_meaningful_response(self, text):
        """Check if response is meaningful (not just random numbers or gibberish)"""
        if not text or len(text) < 20:  # Increased minimum length
            return False
        
        # Check for random numbers like "6. 52008855°C" or "91. 905003745mm"
        if re.search(r'\d+\.\s*\d{7,}', text):  # Long decimal numbers with spaces
            return False
        
        # Check for too many numbers with decimals (gibberish)
        words = text.split()
        decimal_count = sum(1 for w in words if '.' in w and any(c.isdigit() for c in w))
        if decimal_count > 2:  # More than 2 decimal numbers is suspicious
            return False
        
        # Check for too many consecutive punctuation marks
        if '..' in text or '...' in text:
            return False
        
        # Check for actual agricultural words
        agricultural_keywords = ['soil', 'crop', 'plant', 'fertilizer', 'irrigation', 'yield', 
                                 'cultivation', 'management', 'pH', 'rainfall', 'temperature']
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in agricultural_keywords if keyword in text_lower)
        
        if keyword_count == 0:  # Must have at least one agricultural keyword
            return False
        
        # Check for actual sentences (must have proper words)
        has_words = any(len(w) > 2 and w.isalpha() for w in words)
        if not has_words:
            return False
        
        return True
    
    def _get_fallback_llm_recommendation(self, crop_name, soil_properties, climate_conditions):
        """Fallback recommendation when fine-tuned model fails or produces poor output"""
        recommendations = []
        
        # Add soil-specific advice
        ph = soil_properties.get('pH', 7)
        if ph < 6.0:
            recommendations.append("Apply lime to raise soil pH to optimal range (6.0-7.0)")
        elif ph > 7.5:
            recommendations.append("Add organic matter to improve soil structure and fertility")
        
        om = soil_properties.get('organic_matter', 2)
        if om < 2.0:
            recommendations.append("Incorporate organic matter (compost or manure) to improve soil health")
        
        # Add climate-specific advice
        rainfall = climate_conditions.get('rainfall_mean', 1000)
        if rainfall < 600:
            recommendations.append("Implement irrigation system for reliable water supply")
        elif rainfall > 1500:
            recommendations.append("Ensure proper drainage to prevent waterlogging")
        
        # Add general recommendations
        recommendations.append(f"Prepare well-drained beds for {crop_name}")
        recommendations.append("Follow recommended spacing and planting dates for optimal yields")
        
        # Build response
        response = f"For {crop_name.title()} cultivation in Uganda:\n"
        response += "\n".join(f"• {rec}" for rec in recommendations[:4])
        response += f"\n\nExpected yield: Good with proper management practices."
        
        return response
    
    def _generate_template_recommendation(self, suitable_crops, soil_properties, climate_conditions):
        """Generate template-based recommendation as fallback"""
        if not suitable_crops:
            return "No suitable crops found for your current conditions."
            
        top_crop = suitable_crops[0]
        
        # Create detailed recommendation
        recommendation = f"**Crop Recommendation:** {top_crop['crop'].title()}\n\n"
        recommendation += f"**Suitability Score:** {top_crop['suitability_score']:.1%}\n\n"
        
        # Add specific conditions
        recommendation += f"**Your Conditions:**\n"
        recommendation += f"- Soil pH: {soil_properties.get('pH', 'Unknown')}\n"
        recommendation += f"- Temperature: {climate_conditions.get('temperature_mean', 'Unknown')}°C\n"
        recommendation += f"- Rainfall: {climate_conditions.get('rainfall_mean', 'Unknown')}mm\n"
        recommendation += f"- Soil Texture: {soil_properties.get('texture_class', 'Unknown').title()}\n\n"
        
        # Add other recommendations
        if len(suitable_crops) > 1:
            other_crops = [crop['crop'].title() for crop in suitable_crops[1:4]]  # Show up to 3 alternatives
            recommendation += f"**Alternative Crops:** {', '.join(other_crops)}\n\n"
        
        # Add specific recommendations
        if top_crop['recommendations']:
            recommendation += f"**Key Recommendations:**\n"
            for rec in top_crop['recommendations'][:3]:
                recommendation += f"- {rec}\n"
        
        return recommendation
    
    def _get_rag_evidence_for_model(self, crop_name, soil_properties, climate_conditions, suitable_crops):
        """Retrieve RAG evidence from knowledge graph for fine-tuned model"""
        rag_evidence_parts = []
        
        # Check if semantic retriever is available
        if hasattr(self, 'semantic_retriever') and self.semantic_retriever:
            try:
                # Build query for crop-specific evidence
                query_parts = [
                    crop_name,
                    f"pH {soil_properties.get('pH', 'unknown')}",
                    f"organic matter {soil_properties.get('organic_matter', 'unknown')}",
                    f"{soil_properties.get('texture_class', 'unknown')} soil",
                    f"temperature {climate_conditions.get('temperature_mean', 'unknown')}",
                    f"rainfall {climate_conditions.get('rainfall_mean', 'unknown')}"
                ]
                
                query = " ".join(query_parts)
                
                # Retrieve relevant evidence (only top 2 to keep prompt short)
                evidence_results = self.semantic_retriever.hybrid_retrieve(query, top_k=2)
                
                if evidence_results:
                    # Format evidence for prompt (keep it concise)
                    for i, result in enumerate(evidence_results[:2], 1):
                        triple = result['triple']
                        if triple and isinstance(triple, dict):
                            subject = triple.get('subject', '')
                            predicate = triple.get('predicate', '')
                            obj = triple.get('object', '')
                            
                            if subject and predicate and obj:
                                # Keep it short: just the triple, no extra evidence
                                evidence_text = f"{subject} {predicate} {obj}"
                                rag_evidence_parts.append(evidence_text)
            except Exception as e:
                logger.warning(f"RAG evidence retrieval failed: {e}")
        
        return "\n".join(rag_evidence_parts) if rag_evidence_parts else ""
    
    def _build_structured_prompt_with_rag(self, crop_name, soil_properties, climate_conditions, rag_evidence, suitable_crops):
        """Build balanced agricultural prompt with RAG evidence"""
        
        # Extract key properties
        ph = soil_properties.get('pH', 'Unknown')
        texture = soil_properties.get('texture_class', 'Unknown')
        om = soil_properties.get('organic_matter', 'Unknown')
        temp = climate_conditions.get('temperature_mean', 'Unknown')
        rainfall = climate_conditions.get('rainfall_mean', 'Unknown')
        
        # Build prompt with better structure
        prompt_parts = [
            f"Agricultural advice for {crop_name.title()} in Uganda:",
            f"Soil: pH {ph}, {texture.title()} with {om}% organic matter",
            f"Climate: {temp}°C temperature, {rainfall}mm annual rainfall"
        ]
        
        # Add RAG evidence if available
        if rag_evidence:
            prompt_parts.append(f"Based on: {rag_evidence}")
        
        # Better instruction
        prompt_parts.append("Provide 3-4 specific cultivation tips (soil prep, planting, management):")
        
        return "\n".join(prompt_parts)
    
    def _generate_llm_recommendation(self, suitable_crops, soil_properties, climate_conditions):
        """Generate recommendation using Gemini API with enhanced context"""
        # Prepare enhanced context for LLM
        context = self._prepare_llm_context(suitable_crops, soil_properties, climate_conditions)
        
        # Create focused prompt for Gemini API
        prompt = f"""
You are an expert agricultural advisor specializing in Uganda/East Africa. Based on the provided soil and climate conditions, provide a comprehensive crop recommendation.

{context}

Please provide:
1. A clear recommendation for the most suitable crop(s) with reasoning
2. Specific management practices for the recommended crop(s)
3. Soil improvement suggestions if needed
4. Climate adaptation strategies
5. Economic considerations for Ugandan farmers

Keep the response practical, actionable, and suitable for smallholder farmers in Uganda. Focus on sustainable and cost-effective practices.
"""
        
        try:
            response = llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API generation error: {e}")
            # Return a helpful expert analysis fallback
            return self._generate_expert_fallback(suitable_crops, soil_properties, climate_conditions)
    
    def _generate_expert_fallback(self, suitable_crops, soil_properties, climate_conditions):
        """Generate expert analysis fallback in paragraph format"""
        if not suitable_crops:
            return "No suitable crops identified for analysis."
        
        top_crop = suitable_crops[0]
        analysis = f"Expert agricultural analysis reveals that your soil pH of {soil_properties.get('pH', 0)} presents "
        
        # Soil analysis
        ph = soil_properties.get('pH', 0)
        if ph < 5.5:
            analysis += f"acidic conditions requiring immediate lime application to elevate pH to the optimal range of 6.0-7.0 for enhanced nutrient availability and crop productivity. "
        elif ph > 8.0:
            analysis += f"alkaline conditions that would significantly benefit from sulfur application to reduce pH and improve nutrient uptake efficiency. "
        else:
            analysis += f"favorable conditions within the acceptable range for most agricultural crops, providing an excellent foundation for cultivation. "
        
        # Climate analysis
        rainfall = climate_conditions.get('rainfall_mean', 0)
        if rainfall < 500:
            analysis += f"Given the limited annual rainfall of {rainfall}mm, implementing a comprehensive irrigation system will be critical for ensuring consistent crop production and yield optimization. "
        elif rainfall > 2000:
            analysis += f"The substantial rainfall of {rainfall}mm necessitates robust drainage infrastructure to prevent waterlogging and maintain optimal soil conditions throughout the growing season. "
        else:
            analysis += f"The {rainfall}mm annual rainfall provides excellent growing conditions with adequate moisture for sustained crop development. "
        
        # Crop-specific advice
        analysis += f"Based on comprehensive evaluation, {top_crop['crop'].title()} demonstrates exceptional suitability with a confidence score of {top_crop['suitability_score']:.1%}. "
        
        if top_crop['recommendations']:
            recommendations_text = ', '.join(top_crop['recommendations'][:3])
            analysis += f"Essential management practices include {recommendations_text} to maximize yield potential and ensure sustainable agricultural success."
        
        return analysis
    
    def _prepare_llm_context(self, suitable_crops, soil_properties, climate_conditions):
        """Prepare context for LLM with RAG evidence"""
        context_parts = []
        
        # Add soil and climate conditions
        context_parts.append("AGRICULTURAL CONDITIONS:")
        context_parts.append(f"Soil pH: {soil_properties.get('pH', 'Unknown')}")
        context_parts.append(f"Organic Matter: {soil_properties.get('organic_matter', 'Unknown')}%")
        context_parts.append(f"Soil Texture: {soil_properties.get('texture_class', 'Unknown')}")
        context_parts.append(f"Temperature: {climate_conditions.get('temperature_mean', 'Unknown')}°C")
        context_parts.append(f"Rainfall: {climate_conditions.get('rainfall_mean', 'Unknown')}mm")
        context_parts.append("")
        
        # Add crop suitability analysis
        context_parts.append("CROP SUITABILITY ANALYSIS:")
        for i, crop_info in enumerate(suitable_crops[:5]):
            context_parts.append(f"{i+1}. {crop_info['crop'].title()}: Suitability Score {crop_info['suitability_score']:.2f}")
            if crop_info['recommendations']:
                context_parts.append(f"   Recommendations: {'; '.join(crop_info['recommendations'])}")
            if crop_info['violations']:
                context_parts.append(f"   Constraints: {'; '.join(crop_info['violations'])}")
        context_parts.append("")
        
        # Add RAG evidence if available
        if hasattr(self, 'semantic_retriever') and self.semantic_retriever:
            try:
                # Create query for evidence retrieval
                query_parts = []
                query_parts.append(f"pH {soil_properties.get('pH', 'unknown')}")
                query_parts.append(f"organic matter {soil_properties.get('organic_matter', 'unknown')}")
                query_parts.append(f"{soil_properties.get('texture_class', 'unknown')} soil")
                query_parts.append(f"temperature {climate_conditions.get('temperature_mean', 'unknown')}")
                query_parts.append(f"rainfall {climate_conditions.get('rainfall_mean', 'unknown')}")
                
                # Add crop names to query
                crop_names = [crop['crop'] for crop in suitable_crops[:3]]
                query_parts.extend(crop_names)
                
                query = " ".join(query_parts)
                
                # Retrieve relevant evidence
                evidence_results = self.semantic_retriever.hybrid_retrieve(query, top_k=10)
                
                if evidence_results:
                    context_parts.append("EVIDENCE FROM AGRICULTURAL KNOWLEDGE GRAPH:")
                    for i, result in enumerate(evidence_results[:5]):
                        triple = result['triple']
                        context_parts.append(f"{i+1}. {triple['subject']} {triple['predicate']} {triple['object']}")
                        context_parts.append(f"   Relevance Score: {result['score']:.3f}")
                        if 'evidence' in triple:
                            context_parts.append(f"   Evidence: {triple['evidence']}")
                    context_parts.append("")
            except Exception as e:
                logger.warning(f"RAG evidence retrieval failed: {e}")
        
        return "\n".join(context_parts)
    
    def _get_structured_recommendation_sections(self, suitable_crops, soil_properties, climate_conditions):
        """Return recommendation sections in structured format for better UI display"""
        sections = {}
        
        # Get top crop info
        if suitable_crops:
            top_crop = suitable_crops[0]
            sections['primary_recommendation'] = {
                'crop': top_crop['crop'].title(),
                'score': top_crop['suitability_score']
            }
            
            # Extract action items from recommendations
            sections['action_items'] = self._extract_action_items(suitable_crops, soil_properties)
            
            # Get summary metrics
            sections['summary_metrics'] = self._extract_summary_metrics(suitable_crops, soil_properties, climate_conditions)
        else:
            sections['primary_recommendation'] = {'crop': 'No suitable crops found', 'score': 0.0}
            sections['action_items'] = []
            sections['summary_metrics'] = {}
        
        # Get AI insights
        ai_insights = self._get_ai_insights(suitable_crops, soil_properties, climate_conditions)
        sections['ai_analysis'] = ai_insights if ai_insights else ""
        
        # Get expert analysis
        expert_analysis = self._get_expert_analysis(suitable_crops, soil_properties, climate_conditions)
        sections['expert_analysis'] = expert_analysis if expert_analysis else ""
        
        # Get technical analysis
        technical_analysis = self._get_technical_analysis(suitable_crops, soil_properties, climate_conditions)
        sections['technical_analysis'] = technical_analysis if technical_analysis else ""
        
        # Get implementation plan
        implementation_plan = self._get_implementation_plan(suitable_crops, soil_properties, climate_conditions)
        sections['implementation_plan'] = implementation_plan if implementation_plan else ""
        
        return sections
    
    def _extract_action_items(self, suitable_crops, soil_properties):
        """Extract action items as structured list"""
        action_items = []
        
        if suitable_crops:
            top_crop = suitable_crops[0]
            
            # Add recommendations as action items
            if top_crop.get('recommendations'):
                for rec in top_crop['recommendations']:
                    action_items.append({
                        'type': 'recommendation',
                        'priority': 'high',
                        'text': rec
                    })
            
            # Add constraint violations as urgent action items
            if top_crop.get('violations'):
                for violation in top_crop['violations']:
                    action_items.append({
                        'type': 'critical',
                        'priority': 'urgent',
                        'text': violation
                    })
        
        return action_items[:10]  # Limit to top 10
    
    def _extract_summary_metrics(self, suitable_crops, soil_properties, climate_conditions):
        """Extract key metrics for summary cards"""
        metrics = {}
        
        if suitable_crops:
            top_crop = suitable_crops[0]
            metrics['suitability_score'] = top_crop['suitability_score']
            metrics['total_recommendations'] = len(top_crop.get('recommendations', []))
            metrics['critical_issues'] = len(top_crop.get('violations', []))
        
        # Soil health indicators
        ph = soil_properties.get('pH', 0)
        if ph < 6.0 or ph > 7.5:
            metrics['soil_ph_status'] = 'needs_adjustment'
        else:
            metrics['soil_ph_status'] = 'optimal'
        
        # Climate suitability
        temp = climate_conditions.get('temperature_mean', 0)
        if 15 <= temp <= 30:
            metrics['temperature_status'] = 'optimal'
        else:
            metrics['temperature_status'] = 'needs_monitoring'
        
        return metrics
    
    def _get_data_sources(self):
        """Get information about data sources used"""
        sources = {
            "constraint_engine": "active",
            "gcn_model_loaded": self.models_loaded,
            "knowledge_graph_triples": 0,  # Not loading large KG
            "dataset_triples": 0,  # Not loading
            "literature_triples": 0,
            "ugandan_data_points": 0,
            "rag_pipeline_active": self.rag_loaded,
            "llm_model_available": llm_model is not None,
            "mode": "lightweight_with_model"
        }
        
        return sources
    
    def _evaluate_farming_conditions(self, crop, farming_conditions):
        """Evaluate if crop is suitable based on farming conditions"""
        # Basic farming condition checks
        irrigation = farming_conditions.get('irrigation', '')
        fertilizer_access = farming_conditions.get('fertilizer_access', '')
        labor_availability = farming_conditions.get('labor_availability', '')
        budget_range = farming_conditions.get('budget_range', '')
        
        # High-value crops need better resources
        high_value_crops = ['coffee', 'cotton', 'sugarcane']
        if crop in high_value_crops:
            if irrigation == 'none' and fertilizer_access in ['none', 'limited']:
                return False
            if budget_range == 'low':
                return False
        
        # Labor-intensive crops need adequate labor
        labor_intensive_crops = ['rice', 'cotton', 'sugarcane']
        if crop in labor_intensive_crops and labor_availability == 'low':
            return False
        
        return True
    
    def _get_farming_factors(self, crop, farming_conditions):
        """Get farming factors that affect crop suitability"""
        factors = []
        
        irrigation = farming_conditions.get('irrigation', '')
        fertilizer_access = farming_conditions.get('fertilizer_access', '')
        labor_availability = farming_conditions.get('labor_availability', '')
        market_access = farming_conditions.get('market_access', '')
        budget_range = farming_conditions.get('budget_range', '')
        
        if irrigation == 'none':
            factors.append("Requires irrigation setup")
        elif irrigation == 'abundant':
            factors.append("Good irrigation available")
            
        if fertilizer_access == 'none':
            factors.append("Limited fertilizer access")
        elif fertilizer_access == 'good':
            factors.append("Good fertilizer access")
            
        if labor_availability == 'low':
            factors.append("Low labor availability")
        elif labor_availability == 'high':
            factors.append("High labor availability")
            
        if market_access == 'poor':
            factors.append("Poor market access")
        elif market_access == 'excellent':
            factors.append("Excellent market access")
            
        if budget_range == 'low':
            factors.append("Low budget constraints")
        elif budget_range == 'high':
            factors.append("High budget available")
        
        return factors

# Initialize the API - this will be shared across all workers
# Note: We accept that this will use memory, but Railway's restart policy will handle crashes
try:
    api = AgriculturalAPI()
    logger.info(" ✅ AgriculturalAPI initialized successfully")
except Exception as e:
    logger.error(f" Failed to initialize AgriculturalAPI: {e}")
    api = None

def get_api():
    """Get the API instance"""
    global api
    if api is None:
        logger.error(" ❌ API not initialized")
        raise RuntimeError("AgriculturalAPI not initialized")
    return api

@app.route('/')
def home():
    """Serve the main web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriAI - Intelligent Agricultural Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* AI System Interface */
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            margin: 0;
            padding: 0;
            color: #e0e6ed;
            line-height: 1.6;
            scroll-behavior: smooth;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: rgba(15, 15, 35, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        
        
        .ai-status {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
            position: relative;
            z-index: 1;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            border-radius: 15px;
            font-size: 0.8rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff96;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Main Content Area */
        .main-content {
            padding: 25px;
            background: rgba(15, 15, 35, 0.8);
        }
        
        .ai-chat-interface {
            background: rgba(20, 20, 40, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        .chat-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 15px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .chat-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(0, 255, 150, 0.05) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        
        .chat-title-section {
            flex: 1;
            position: relative;
            z-index: 1;
        }
        
        .system-status {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 10px;
            position: relative;
            z-index: 1;
        }
        
        .status-indicators {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            border-radius: 20px;
            font-size: 0.85rem;
            color: #00ff96;
            transition: all 0.3s ease;
        }
        
        .status-item:hover {
            background: rgba(0, 255, 150, 0.2);
            transform: translateY(-2px);
        }
        
        .status-item i {
            font-size: 0.9rem;
        }
        
        .system-tagline {
            font-size: 0.9rem;
            color: #a0a0a0;
            text-align: right;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        
        .chat-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 5px;
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .chat-subtitle {
            font-size: 1rem;
            color: #a0a0a0;
            margin-bottom: 15px;
            font-weight: 400;
        }
        
        .ai-status {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            border-radius: 15px;
            font-size: 0.8rem;
            color: #ffffff;
            font-weight: 500;
        }
        
        .ai-avatar {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #000;
            animation: glow 2s infinite alternate;
            flex-shrink: 0;
            position: relative;
            z-index: 1;
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 10px rgba(0, 255, 150, 0.5); }
            to { box-shadow: 0 0 20px rgba(0, 255, 150, 0.8); }
        }
        
        /* Input Form Styles */
        .input-form {
            background: rgba(25, 25, 50, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 150, 0.1);
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 18px;
            margin-bottom: 25px;
        }
        
        .form-group {
            position: relative;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #ffffff;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid rgba(0, 255, 150, 0.2);
            border-radius: 10px;
            background: rgba(15, 15, 35, 0.8);
            color: #ffffff;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #00ff96;
            box-shadow: 0 0 15px rgba(0, 255, 150, 0.3);
            background: rgba(15, 15, 35, 0.9);
        }
        
        .form-group input::placeholder {
            color: #666;
        }
        
        /* AI Action Button */
        .ai-submit-btn {
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            color: #000;
            border: none;
            padding: 12px 30px;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0 auto;
            box-shadow: 0 4px 15px rgba(0, 255, 150, 0.3);
        }
        
        .ai-submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 255, 150, 0.4);
            background: linear-gradient(135deg, #00d4aa, #00ff96);
        }
        
        .ai-submit-btn:active {
            transform: translateY(0);
        }
        
        .ai-submit-btn i {
            font-size: 1rem;
        }
        
        /* Results Display */
        .results-container {
            background: rgba(20, 20, 40, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            display: none;
        }
        
        .results-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0, 255, 150, 0.2);
        }
        
        .results-icon {
            width: 35px;
            height: 35px;
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: #000;
        }
        
        .results-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        /* Crop Cards */
        .crop-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .crop-card {
            background: rgba(25, 25, 50, 0.8);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }
        
        .crop-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 255, 150, 0.2);
            border-color: rgba(0, 255, 150, 0.4);
            background: rgba(30, 30, 60, 0.9);
        }
        
        .crop-card.expanded {
            border-color: #00ff96;
            box-shadow: 0 0 20px rgba(0, 255, 150, 0.2);
            background: rgba(30, 30, 60, 0.95);
        }
        
        .crop-details {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-in-out;
        }
        
        .crop-card.expanded .crop-details {
            max-height: 1000px;
            padding-top: 15px;
            margin-top: 10px;
            border-top: 1px solid rgba(0, 255, 150, 0.2);
        }
        
        .crop-toggle-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #00ff96;
            font-size: 0.85rem;
            margin-top: 10px;
            transition: color 0.3s ease;
        }
        
        .crop-card:hover .crop-toggle-btn {
            color: #00d4aa;
        }
        
        .crop-toggle-icon {
            transition: transform 0.3s ease;
        }
        
        .crop-card.expanded .crop-toggle-icon {
            transform: rotate(180deg);
        }
        
        .crop-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00ff96, #00d4aa);
        }
        
        .crop-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
        }
        
        .crop-score {
            font-size: 1rem;
            color: #00ff96;
            font-weight: 500;
            margin-bottom: 12px;
        }
        
        .crop-details {
            color: #a0a0a0;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        
        .crop-details ul {
            margin: 8px 0;
            padding-left: 15px;
        }
        
        .crop-details li {
            margin-bottom: 3px;
        }
        
        /* Loading Animation */
        .loading-container {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .ai-loader {
            width: 60px;
            height: 60px;
            border: 3px solid rgba(0, 255, 150, 0.2);
            border-top: 3px solid #00ff96;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #00ff96;
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                margin: 5px;
                border-radius: 10px;
            }
            
            .main-content {
                padding: 15px;
            }
            
            .ai-chat-interface {
                padding: 15px;
            }
            
            .chat-header {
                padding: 15px;
                gap: 12px;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .chat-title {
                font-size: 1.5rem;
            }
            
            .chat-subtitle {
                font-size: 0.9rem;
            }
            
            .ai-avatar {
                width: 45px;
                height: 45px;
                font-size: 1.3rem;
            }
            
            .system-status {
                align-items: center;
                margin-top: 10px;
            }
            
            .status-indicators {
                gap: 10px;
                justify-content: center;
            }
            
            .status-item {
                font-size: 0.8rem;
                padding: 6px 10px;
            }
            
            .system-tagline {
                text-align: center;
                font-size: 0.8rem;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }
            
            .ai-status {
                flex-direction: column;
                gap: 8px;
        }
        
        .status-indicator {
                padding: 5px 10px;
                font-size: 0.75rem;
            }
            
            .crop-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }
            
            .crop-card {
                padding: 12px;
            }
            
            .results-container {
                padding: 15px;
            }
        }
        
        @media (max-width: 480px) {
            
            .chat-title {
                font-size: 1rem;
            }
            
            .chat-subtitle {
                font-size: 0.75rem;
            }
            
            .ai-submit-btn {
                padding: 10px 25px;
                font-size: 0.9rem;
            }
        }
        
        /* Error Messages */
        .error-message {
            background: rgba(255, 59, 48, 0.1);
            border: 1px solid rgba(255, 59, 48, 0.3);
            color: #ff3b30;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        /* Success Messages */
        .success-message {
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            color: #00ff96;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        /* Real-time Validation Feedback */
        .form-group {
            position: relative;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            border-color: #00ff96;
            box-shadow: 0 0 0 3px rgba(0, 255, 150, 0.1);
        }
        
        .form-group.success input,
        .form-group.success select {
            border-color: #00ff96;
        }
        
        .form-group.error input,
        .form-group.error select {
            border-color: #ff3b30;
        }
        
        .form-group .validation-icon {
            position: absolute;
            right: 15px;
            top: 40px;
            font-size: 1.2rem;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .form-group.success .validation-icon {
            color: #00ff96;
            opacity: 1;
        }
        
        .form-group.error .validation-icon {
            color: #ff3b30;
            opacity: 1;
        }
        
        .form-group .error-text {
            color: #ff3b30;
            font-size: 0.8rem;
            margin-top: 5px;
            display: none;
        }
        
        .form-group.error .error-text {
            display: block;
        }
        
        /* Progress Bar for Loading */
        .loading-progress {
            width: 100%;
            height: 4px;
            background: rgba(0, 255, 150, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        .loading-progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00ff96, #00d4aa);
            border-radius: 2px;
            animation: loading-progress 2s infinite;
        }
        
        @keyframes loading-progress {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        /* Suitability Score Progress Bar */
        .score-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 8px 0;
        }
        
        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff96, #00d4aa);
            border-radius: 10px;
            transition: width 0.8s ease;
            position: relative;
        }
        
        .score-bar-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Improved Crop Cards */
        .crop-card-header-score {
            font-size: 1.2rem;
            font-weight: 700;
            color: #00ff96;
            margin-left: auto;
        }
        
        .crop-card.expanded {
            border-color: #00ff96;
            box-shadow: 0 0 20px rgba(0, 255, 150, 0.2);
        }
        
        .crop-card-toggle {
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .crop-card.expanded .crop-card-toggle i {
            transform: rotate(180deg);
        }
        
        /* Tooltip for Help Text */
        .help-tooltip {
            position: relative;
            display: inline-block;
            margin-left: 8px;
            cursor: help;
        }
        
        .help-tooltip-icon {
            color: #a0a0a0;
            font-size: 0.9rem;
        }
        
        .help-tooltip:hover .help-tooltip-icon {
            color: #00ff96;
        }
        
        .help-tooltip-text {
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(20, 20, 40, 0.98);
            color: #e0e6ed;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.85rem;
            width: 250px;
            text-align: left;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 255, 150, 0.3);
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
            pointer-events: none;
        }
        
        .help-tooltip:hover .help-tooltip-text {
            opacity: 1;
            visibility: visible;
        }
        
        .help-tooltip-text::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: rgba(20, 20, 40, 0.98);
        }
        
        /* Chart Styles */
        #landAllocationChart {
            max-width: 100%;
            height: auto !important;
        }
        
        .chart-legend {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .chart-legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        
        .legend-info {
            flex: 1;
            color: #e0e6ed;
            font-size: 0.9rem;
        }
        
        .legend-crop {
            font-weight: 600;
            color: #ffffff;
            text-transform: capitalize;
        }
        
        .legend-area {
            color: #a0a0a0;
            font-size: 0.85rem;
        }
        
        /* Chart Container - Responsive */
        .chart-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            align-items: center;
        }
        
        @media (max-width: 768px) {
            .chart-container {
                grid-template-columns: 1fr !important;
            }
            
            #landAllocationChart {
                margin: 0 auto;
            }
            
            .chart-legend {
                margin-top: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- AI Chat Interface -->
            <div class="ai-chat-interface">
                <div class="chat-header">
                    <div class="ai-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="chat-title-section">
                        <div class="chat-title">AgriAI</div>
                        <div class="chat-subtitle">Smart Agricultural Assistant</div>
                    </div>
                    <div class="system-status">
                        <div class="status-indicators">
                            <div class="status-item">
                                <i class="fas fa-brain"></i>
                                <span>AI Model Active</span>
                            </div>
                            <div class="status-item">
                                <i class="fas fa-database"></i>
                                <span>Knowledge Based</span>
                            </div>
                            <div class="status-item">
                                <i class="fas fa-seedling"></i>
                                <span>Ready for Analysis</span>
                            </div>
                        </div>
                        <div class="system-tagline">
                            Powered by Advanced AI • Optimized for Uganda Agriculture
                        </div>
                    </div>
                </div>
                    
                <form id="agriculturalForm" class="input-form">
                    <div class="form-grid">
                        <!-- Soil Properties -->
                        <div class="form-group">
                            <label for="soil_ph">
                                <i class="fas fa-flask"></i> Soil pH
                                <span class="help-tooltip">
                                    <i class="fas fa-question-circle help-tooltip-icon"></i>
                                    <span class="help-tooltip-text">pH measures soil acidity. Most crops grow best at pH 6.0-7.5. Lower pH is acidic, higher is alkaline.</span>
                                </span>
                            </label>
                            <input type="number" id="soil_ph" name="soil_ph" step="0.1" min="0" max="14" placeholder="Enter soil pH (0-14, e.g., 6.5)">
                            <span class="validation-icon">
                                <i class="fas fa-check-circle"></i>
                            </span>
                            <div class="error-text">Please enter a valid pH value between 0 and 14</div>
                        </div>
                        <div class="form-group">
                            <label for="organic_matter">
                                <i class="fas fa-leaf"></i> Organic Matter (%)
                                <span class="help-tooltip">
                                    <i class="fas fa-question-circle help-tooltip-icon"></i>
                                    <span class="help-tooltip-text">Organic matter improves soil structure and nutrient availability. Aim for 3-5% for healthy crops.</span>
                                </span>
                            </label>
                            <input type="number" id="organic_matter" name="organic_matter" step="0.1" min="0" max="20" placeholder="Enter organic matter %">
                            <span class="validation-icon">
                                <i class="fas fa-check-circle"></i>
                            </span>
                            <div class="error-text">Please enter a value between 0 and 20</div>
                        </div>
                        <div class="form-group">
                            <label for="texture_class">
                                <i class="fas fa-mountain"></i> Soil Texture
                                <span class="help-tooltip">
                                    <i class="fas fa-question-circle help-tooltip-icon"></i>
                                    <span class="help-tooltip-text">Soil texture affects water retention and drainage. Loam is ideal for most crops.</span>
                                </span>
                            </label>
                            <select id="texture_class" name="texture_class">
                                <option value="">Select soil texture</option>
                                <option value="clay">Clay</option>
                                <option value="clay_loam">Clay Loam</option>
                                <option value="loam">Loam</option>
                                <option value="sandy_loam">Sandy Loam</option>
                                <option value="sandy_clay_loam">Sandy Clay Loam</option>
                                <option value="silt_loam">Silt Loam</option>
                                <option value="sandy">Sandy</option>
                            </select>
                            <span class="validation-icon">
                                <i class="fas fa-check-circle"></i>
                            </span>
                            <div class="error-text">Please select a soil texture</div>
                        </div>
                        
                        <!-- Soil Nutrients -->
                        <div class="form-group">
                            <label for="nitrogen"><i class="fas fa-atom"></i> Nitrogen (ppm)</label>
                            <input type="number" id="nitrogen" name="nitrogen" step="1" min="0" max="500" placeholder="Enter nitrogen content">
                        </div>
                        <div class="form-group">
                            <label for="phosphorus"><i class="fas fa-atom"></i> Phosphorus (ppm)</label>
                            <input type="number" id="phosphorus" name="phosphorus" step="1" min="0" max="200" placeholder="Enter phosphorus content">
                        </div>
                        <div class="form-group">
                            <label for="potassium"><i class="fas fa-atom"></i> Potassium (ppm)</label>
                            <input type="number" id="potassium" name="potassium" step="1" min="0" max="1000" placeholder="Enter potassium content">
                    </div>
                    
                        <!-- Climate Conditions -->
                        <div class="form-group">
                            <label for="temperature_mean"><i class="fas fa-thermometer-half"></i> Temperature (°C)</label>
                            <input type="number" id="temperature_mean" name="temperature_mean" step="0.1" min="10" max="40" placeholder="Enter average temperature">
                        </div>
                            <div class="form-group">
                            <label for="rainfall_mean"><i class="fas fa-cloud-rain"></i> Rainfall (mm)</label>
                            <input type="number" id="rainfall_mean" name="rainfall_mean" step="1" min="200" max="3000" placeholder="Enter annual rainfall">
                            </div>
                            
                        <!-- Essential Farming Info -->
                            <div class="form-group">
                            <label for="available_land"><i class="fas fa-map"></i> Available Land (hectares)</label>
                            <input type="number" id="available_land" name="available_land" step="0.1" min="0.1" max="1000" placeholder="Enter land area">
                            </div>
                    </div>
                    
                    <button type="submit" class="ai-submit-btn">
                        <i class="fas fa-brain"></i>
                        Generate Recommendations
                    </button>
                </form>
                
                <!-- Loading Animation -->
                <div class="loading-container" id="loadingContainer">
                    <div class="ai-loader"></div>
                    <div class="loading-text">AI analyzing conditions...</div>
                    <div class="loading-progress">
                        <div class="loading-progress-bar"></div>
                    </div>
                    <div class="loading-steps" style="margin-top: 20px; color: #a0a0a0; font-size: 0.9rem;">
                        <div><i class="fas fa-check" style="color: #00ff96;"></i> Analyzing soil properties</div>
                        <div style="margin-top: 8px;"><i class="fas fa-check" style="color: #00ff96;"></i> Evaluating crop suitability</div>
                        <div style="margin-top: 8px;"><i class="fas fa-spinner fa-spin" style="color: #00ff96;"></i> Loading AI insights</div>
                    </div>
            </div>
            
                <!-- Error/Success Messages -->
                <div class="error-message" id="errorMessage"></div>
                <div class="success-message" id="successMessage"></div>
            </div>
            
            <!-- Results Container -->
            <div class="results-container" id="resultsContainer">
                <div class="results-header">
                    <div class="results-icon">
                        <i class="fas fa-chart-line"></i>
                </div>
                    <div class="results-title">AI Recommendations</div>
                    </div>
                <div id="resultsContent"></div>
                    </div>
        </div>
    </div>
    
    <script>
        // AI System JavaScript
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('agriculturalForm');
            const loadingContainer = document.getElementById('loadingContainer');
            const resultsContainer = document.getElementById('resultsContainer');
            const errorMessage = document.getElementById('errorMessage');
            const successMessage = document.getElementById('successMessage');
            const resultsContent = document.getElementById('resultsContent');
            
            // Real-time validation function
            function validateField(fieldName, value) {
                const field = document.querySelector(`#${fieldName}`);
                const formGroup = field.closest('.form-group');
                
                // Remove previous validation states
                formGroup.classList.remove('success', 'error');
                
                if (!value || value === '') {
                    formGroup.classList.add('error');
                    return false;
                }
                
                // Field-specific validation
                let isValid = true;
                switch(fieldName) {
                    case 'soil_ph':
                        const ph = parseFloat(value);
                        isValid = ph >= 0 && ph <= 14;
                        break;
                    case 'organic_matter':
                        const om = parseFloat(value);
                        isValid = om >= 0 && om <= 20;
                        break;
                    case 'texture_class':
                        isValid = value !== '';
                        break;
                    case 'nitrogen':
                    case 'phosphorus':
                    case 'potassium':
                        const nutrient = parseFloat(value);
                        isValid = !isNaN(nutrient) && nutrient >= 0;
                        break;
                    case 'temperature_mean':
                        const temp = parseFloat(value);
                        isValid = temp >= 10 && temp <= 40;
                        break;
                    case 'rainfall_mean':
                        const rain = parseFloat(value);
                        isValid = rain >= 200 && rain <= 3000;
                        break;
                    case 'available_land':
                        const land = parseFloat(value);
                        isValid = land > 0 && land <= 1000;
                        break;
                }
                
                if (isValid) {
                    formGroup.classList.add('success');
                } else {
                    formGroup.classList.add('error');
                }
                
                return isValid;
            }
            
            // Add real-time validation to all input fields
            const formFields = form.querySelectorAll('input, select');
            formFields.forEach(field => {
                field.addEventListener('blur', function() {
                    validateField(this.name, this.value);
                });
                
                field.addEventListener('input', function() {
                    if (this.value) {
                        validateField(this.name, this.value);
                    }
                });
            });
            
            // Form submission with AI loading animation
            form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
                // Show loading animation
                loadingContainer.style.display = 'block';
                resultsContainer.style.display = 'none';
                errorMessage.style.display = 'none';
                successMessage.style.display = 'none';
                
                // Collect form data
                const formData = new FormData(form);
                const flatData = Object.fromEntries(formData);
                
                // Structure data for API
                const data = {
                    soil_properties: {
                        pH: parseFloat(flatData.soil_ph) || 0,
                        organic_matter: parseFloat(flatData.organic_matter) || 0,
                        texture_class: flatData.texture_class || '',
                        nitrogen: parseFloat(flatData.nitrogen) || 0,
                        phosphorus: parseFloat(flatData.phosphorus) || 0,
                        potassium: parseFloat(flatData.potassium) || 0
                    },
                    climate_conditions: {
                        temperature_mean: parseFloat(flatData.temperature_mean) || 0,
                        rainfall_mean: parseFloat(flatData.rainfall_mean) || 0
                    },
                    farming_conditions: {
                        available_land: parseFloat(flatData.available_land) || 0
                    }
                };
                
                try {
                    // Simulate AI processing time
                    await new Promise(resolve => setTimeout(resolve, 2000));
                
                    // Make API call
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    // Hide loading animation
                    loadingContainer.style.display = 'none';
                    
                    if (response.ok) {
                        displayResults(result);
                        successMessage.textContent = 'Analysis complete!';
                        successMessage.style.display = 'block';
                } else {
                        throw new Error(result.error || 'Analysis failed');
                }
                    
            } catch (error) {
                    loadingContainer.style.display = 'none';
                    errorMessage.textContent = `AI Error: ${error.message}`;
                    errorMessage.style.display = 'block';
                }
            });
            
            // Display AI results
            function displayResults(result) {
                let html = '';
                
                // Primary Recommendation Header
                if (result.recommendation_sections && result.recommendation_sections.primary_recommendation) {
                    const primary = result.recommendation_sections.primary_recommendation;
                    html += `
                        <div style="background: linear-gradient(135deg, rgba(0, 255, 150, 0.2), rgba(0, 255, 150, 0.05)); padding: 20px; border-radius: 12px; margin-bottom: 25px; border: 2px solid rgba(0, 255, 150, 0.3);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <div>
                                    <h2 style="color: #00ff96; margin: 0; font-size: 1.4rem; font-weight: 700; display: flex; align-items: center; gap: 10px;">
                                        <i class="fas fa-seedling"></i> ${primary.crop}
                                    </h2>
                                    <div style="color: #ffffff; font-size: 0.95rem; margin-top: 5px; opacity: 0.9;">
                                        Suitability Score: ${(primary.score * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <button id="downloadPdfBtn" style="background: linear-gradient(135deg, #00ff96, #00cc77); color: #000; border: none; padding: 10px 20px; border-radius: 25px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 15px rgba(0, 255, 150, 0.2);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(0, 255, 150, 0.4)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0, 255, 150, 0.2)'">
                                    <i class="fas fa-download"></i> Download PDF
                                </button>
                            </div>
                        </div>
                    `;
                }
                
                // Summary Metrics Cards
                if (result.recommendation_sections && result.recommendation_sections.summary_metrics) {
                    const metrics = result.recommendation_sections.summary_metrics;
                    html += `
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px;">
                            <div style="background: rgba(0, 255, 150, 0.15); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 255, 150, 0.3);">
                                <div style="color: #00ff96; font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${(metrics.suitability_score * 100).toFixed(0)}%</div>
                                <div style="color: #ffffff; font-size: 0.85rem; opacity: 0.8;">Suitability Score</div>
                            </div>
                            <div style="background: rgba(255, 107, 107, 0.15); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 107, 107, 0.3);">
                                <div style="color: #ff6b6b; font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${metrics.critical_issues || 0}</div>
                                <div style="color: #ffffff; font-size: 0.85rem; opacity: 0.8;">Critical Issues</div>
                            </div>
                            <div style="background: rgba(255, 165, 0, 0.15); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 165, 0, 0.3);">
                                <div style="color: #ffa500; font-size: 2rem; font-weight: 700; margin-bottom: 5px;">${metrics.total_recommendations || 0}</div>
                                <div style="color: #ffffff; font-size: 0.85rem; opacity: 0.8;">Action Items</div>
                            </div>
                            <div style="background: ${metrics.soil_ph_status === 'optimal' ? 'rgba(0, 255, 150, 0.15)' : 'rgba(255, 107, 107, 0.15)'}; padding: 15px; border-radius: 10px; border: 1px solid ${metrics.soil_ph_status === 'optimal' ? 'rgba(0, 255, 150, 0.3)' : 'rgba(255, 107, 107, 0.3)'};">
                                <div style="color: ${metrics.soil_ph_status === 'optimal' ? '#00ff96' : '#ff6b6b'}; font-size: 1.2rem; font-weight: 700; margin-bottom: 5px; text-transform: capitalize;">${metrics.soil_ph_status || 'Unknown'}</div>
                                <div style="color: #ffffff; font-size: 0.85rem; opacity: 0.8;">Soil pH Status</div>
                            </div>
                        </div>
                    `;
                }
                
                // Action Items Section
                if (result.recommendation_sections && result.recommendation_sections.action_items && result.recommendation_sections.action_items.length > 0) {
                    html += `
                        <div style="background: rgba(25, 25, 50, 0.8); padding: 18px; border-radius: 10px; margin-bottom: 18px; border-left: 4px solid #ffa500;">
                            <h3 style="color: #ffa500; margin: 0 0 15px 0; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-list-check"></i> Priority Actions
                            </h3>
                            <div style="display: flex; flex-direction: column; gap: 10px;">
                    `;
                    
                    result.recommendation_sections.action_items.forEach((item, index) => {
                        const priorityColor = item.priority === 'urgent' ? '#ff6b6b' : '#ffa500';
                        const priorityIcon = item.priority === 'urgent' ? 'fa-exclamation-circle' : 'fa-check-circle';
                        
                        html += `
                            <div style="display: flex; align-items: start; gap: 10px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 6px; border-left: 3px solid ${priorityColor};">
                                <i class="fas ${priorityIcon}" style="color: ${priorityColor}; margin-top: 2px;"></i>
                                <span style="color: #ffffff; font-size: 0.9rem; line-height: 1.4;">${item.text}</span>
                            </div>
                        `;
                    });
                    
                    html += `</div></div>`;
                }
                
                // Structured Sections
                if (result.recommendation_sections) {
                    // AI Analysis Section
                    if (result.recommendation_sections.ai_analysis) {
                        html += `
                            <div style="background: rgba(25, 25, 50, 0.8); padding: 18px; border-radius: 10px; margin-bottom: 18px; border-left: 4px solid #00a8ff;">
                                <h3 style="color: #00a8ff; margin: 0 0 12px 0; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                    <i class="fas fa-brain"></i> AI Analysis
                                </h3>
                                <p style="color: #ffffff; line-height: 1.7; font-size: 0.95rem; margin: 0;">${result.recommendation_sections.ai_analysis}</p>
                            </div>
                        `;
                    }
                    
                    // Expert Analysis Section
                    if (result.recommendation_sections.expert_analysis) {
                        html += `
                            <div style="background: rgba(25, 25, 50, 0.8); padding: 18px; border-radius: 10px; margin-bottom: 18px; border-left: 4px solid #ff6b6b;">
                                <h3 style="color: #ff6b6b; margin: 0 0 12px 0; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                    <i class="fas fa-user-tie"></i> Expert Analysis
                                </h3>
                                <p style="color: #ffffff; line-height: 1.7; font-size: 0.95rem; margin: 0;">${result.recommendation_sections.expert_analysis}</p>
                            </div>
                        `;
                    }
                    
                    // Technical Analysis Section
                    if (result.recommendation_sections.technical_analysis) {
                        html += `
                            <div style="background: rgba(25, 25, 50, 0.8); padding: 18px; border-radius: 10px; margin-bottom: 18px; border-left: 4px solid #ffa500;">
                                <h3 style="color: #ffa500; margin: 0 0 12px 0; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                    <i class="fas fa-chart-line"></i> Technical Analysis
                                </h3>
                                <p style="color: #ffffff; line-height: 1.7; font-size: 0.95rem; margin: 0;">${result.recommendation_sections.technical_analysis}</p>
                            </div>
                        `;
                    }
                    
                    // Implementation Plan Section
                    if (result.recommendation_sections.implementation_plan) {
                        html += `
                            <div style="background: rgba(25, 25, 50, 0.8); padding: 18px; border-radius: 10px; margin-bottom: 18px; border-left: 4px solid #7b68ee;">
                                <h3 style="color: #7b68ee; margin: 0 0 12px 0; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                    <i class="fas fa-tasks"></i> Implementation Plan
                                </h3>
                                <p style="color: #ffffff; line-height: 1.7; font-size: 0.95rem; margin: 0;">${result.recommendation_sections.implementation_plan}</p>
                            </div>
                        `;
                    }
                }
                
                // Crop recommendations
                if (result.suitable_crops && result.suitable_crops.length > 0) {
                    html += '<div class="crop-grid">';
                    result.suitable_crops.forEach(crop => {
                        const suitabilityPercent = (crop.suitability_score * 100).toFixed(1);
                        html += `
                            <div class="crop-card" onclick="toggleCropDetails(this)">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <div class="crop-name">${crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1)}</div>
                                    <div class="crop-card-header-score">${suitabilityPercent}%</div>
                                </div>
                                <div class="score-bar">
                                    <div class="score-bar-fill" style="width: ${suitabilityPercent}%"></div>
                                </div>
                                <div class="crop-toggle-btn">
                                    <span>View Details</span>
                                    <i class="fas fa-chevron-down crop-toggle-icon"></i>
                                </div>
                                <div class="crop-details">
                                    ${crop.recommendations && crop.recommendations.length > 0 ? `
                                        <div style="margin-bottom: 12px;">
                                            <strong style="color: #00ff96; display: block; margin-bottom: 8px;">
                                                <i class="fas fa-lightbulb"></i> Recommendations:
                                            </strong>
                                            <ul style="margin: 0; padding-left: 20px; color: #e0e6ed;">
                                            ${crop.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                        </ul>
                                        </div>
                                    ` : ''}
                                    ${crop.violations && crop.violations.length > 0 ? `
                                        <div style="margin-bottom: 12px;">
                                            <strong style="color: #ff6b6b; display: block; margin-bottom: 8px;">
                                                <i class="fas fa-exclamation-triangle"></i> Constraints:
                                            </strong>
                                            <ul style="margin: 0; padding-left: 20px; color: #e0e6ed;">
                                            ${crop.violations.map(viol => `<li>${viol}</li>`).join('')}
                                        </ul>
                                        </div>
                                    ` : ''}
                                    ${crop.farming_factors && crop.farming_factors.length > 0 ? `
                                        <div style="margin-bottom: 12px;">
                                            <strong style="color: #00d4aa; display: block; margin-bottom: 8px;">
                                                <i class="fas fa-seedling"></i> Farming Factors:
                                            </strong>
                                            <ul style="margin: 0; padding-left: 20px; color: #e0e6ed;">
                                            ${crop.farming_factors.map(factor => `<li>${factor}</li>`).join('')}
                                        </ul>
                                        </div>
                                    ` : ''}
                    </div>
                    </div>
                        `;
                    });
                    html += '</div>';
                }
                
                // Land allocation if available
                if (result.land_allocation && result.land_allocation.crop_details && result.land_allocation.crop_details.length > 0) {
                    html += `
                        <div style="background: rgba(25, 25, 50, 0.8); padding: 20px; border-radius: 12px; margin-top: 20px;">
                            <h3 style="color: #ffffff; margin-bottom: 15px; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-chart-pie"></i> Land Allocation Plan
                            </h3>
                            <div class="chart-container">
                                <div>
                                    <canvas id="landAllocationChart"></canvas>
                    </div>
                                <div class="chart-legend" id="landChartLegend"></div>
                    </div>
                    </div>
                `;
                }
                
                resultsContent.innerHTML = html;
                resultsContainer.style.display = 'block';
                
                // Render pie chart for land allocation
                if (result.land_allocation && result.land_allocation.crop_details && result.land_allocation.crop_details.length > 0) {
                    renderLandAllocationChart(result.land_allocation.crop_details);
                }
                
                // Smooth scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
                
                // Add PDF download functionality
                const downloadBtn = document.getElementById('downloadPdfBtn');
                if (downloadBtn) {
                    downloadBtn.addEventListener('click', async function() {
                        try {
                            // Show loading state
                            downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
                            downloadBtn.disabled = true;
                            
                            // Get form data for PDF generation
                            const formData = new FormData(form);
                            const flatData = Object.fromEntries(formData);
                            
                            // Structure data for PDF API
                            const structuredData = {
                                soil_properties: {
                                    pH: parseFloat(flatData.soil_ph) || 0,
                                    organic_matter: parseFloat(flatData.organic_matter) || 0,
                                    texture_class: flatData.texture_class || '',
                                    nitrogen: parseFloat(flatData.nitrogen) || 0,
                                    phosphorus: parseFloat(flatData.phosphorus) || 0,
                                    potassium: parseFloat(flatData.potassium) || 0
                                },
                                climate_conditions: {
                                    temperature_mean: parseFloat(flatData.temperature_mean) || 0,
                                    rainfall_mean: parseFloat(flatData.rainfall_mean) || 0
                                },
                                farming_conditions: {
                                    available_land: parseFloat(flatData.available_land) || 0
                                }
                            };
                            
                            // Make PDF download request
                            const response = await fetch('/api/download_pdf', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(structuredData)
                            });
                            
                            if (response.ok) {
                                // Create blob and download
                                const blob = await response.blob();
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `Agricultural_Recommendation_Report_${new Date().toISOString().slice(0,10)}.pdf`;
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);
                                document.body.removeChild(a);
                                
                                // Show success message
                                downloadBtn.innerHTML = '<i class="fas fa-check"></i> PDF Downloaded!';
                                setTimeout(() => {
                                    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download PDF';
                                    downloadBtn.disabled = false;
                                }, 2000);
                            } else {
                                throw new Error('Failed to generate PDF');
                            }
                        } catch (error) {
                            console.error('PDF download error:', error);
                            downloadBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Download Failed';
                            setTimeout(() => {
                                downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download PDF';
                                downloadBtn.disabled = false;
                            }, 2000);
                        }
                    });
                }
            }
            
            // Add some AI-like typing effect for status updates
            function updateStatusIndicators() {
                const indicators = document.querySelectorAll('.status-indicator');
                indicators.forEach((indicator, index) => {
                    setTimeout(() => {
                        indicator.style.opacity = '1';
                        indicator.style.transform = 'scale(1)';
                    }, index * 200);
                });
            }
            
            // Initialize AI interface
            updateStatusIndicators();
        });
        
        // Toggle crop card expand/collapse
        function toggleCropDetails(card) {
            card.classList.toggle('expanded');
            const toggleBtn = card.querySelector('.crop-toggle-btn span');
            if (card.classList.contains('expanded')) {
                toggleBtn.textContent = 'Hide Details';
            } else {
                toggleBtn.textContent = 'View Details';
            }
        }
        
        // Render pie chart for land allocation
        function renderLandAllocationChart(cropDetails) {
            const canvas = document.getElementById('landAllocationChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            // Color palette for crops
            const cropColors = {
                'maize': '#FFD700',
                'rice': '#4169E1',
                'beans': '#228B22',
                'cassava': '#FF8C00',
                'sweet_potato': '#9370DB',
                'banana': '#FFFF00',
                'coffee': '#8B4513',
                'cotton': '#FF69B4',
                'sweet potato': '#9370DB',
                'red pepper': '#FF0000',
                'peas': '#90EE90',
                'groundnut': '#DAA520'
            };
            
            // Prepare data
            const labels = cropDetails.map(crop => crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1));
            const data = cropDetails.map(crop => parseFloat(crop.land_allocated));
            const colors = cropDetails.map(crop => cropColors[crop.crop] || '#00ff96');
            const suitabilityScores = cropDetails.map(crop => (crop.suitability_score * 100).toFixed(1) + '%');
            
            // Create legend
            const legendHtml = cropDetails.map((crop, index) => `
                <div class="chart-legend-item" style="border-left-color: ${colors[index]};">
                    <div class="legend-color" style="background-color: ${colors[index]};"></div>
                    <div class="legend-info">
                        <div class="legend-crop">${crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1)}</div>
                        <div class="legend-area">${crop.land_allocated} ha (${suitabilityScores[index]} suitable)</div>
                    </div>
                </div>
            `).join('');
            
            const legendContainer = document.getElementById('landChartLegend');
            if (legendContainer) {
                legendContainer.innerHTML = legendHtml;
            }
            
            // Create chart
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 2,
                        hoverOffset: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(20, 20, 40, 0.95)',
                            titleColor: '#ffffff',
                            bodyColor: '#e0e6ed',
                            borderColor: 'rgba(0, 255, 150, 0.5)',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.parsed || 0;
                                    const dataset = context.dataset;
                                    const percentage = ((value / dataset.data.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                                    return `${label}: ${value.toFixed(2)} ha (${percentage}%)`;
                                }
                            }
                        }
                    },
                    animation: {
                        animateRotate: true,
                        duration: 1000
                    }
                }
            });
            
            // Set chart size responsively
            const container = canvas.parentElement;
            const isMobile = window.innerWidth < 768;
            canvas.style.width = '100%';
            canvas.style.height = isMobile ? '250px' : '300px';
        }
    </script>
</body>
</html>
    """
    return html_content

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    """Get crop recommendation based on soil and climate data"""
    try:
        data = request.get_json()
        logger.info(f"Received recommendation request: {data}")
        
        # Validate input
        required_fields = ['soil_properties', 'climate_conditions']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate soil properties
        soil_props = data['soil_properties']
        required_soil_fields = ['pH', 'organic_matter', 'texture_class']
        for field in required_soil_fields:
            if field not in soil_props:
                logger.error(f"Missing required soil property: {field}")
                return jsonify({'error': f'Missing required soil property: {field}'}), 400
        
        # Validate climate conditions
        climate_conds = data['climate_conditions']
        required_climate_fields = ['temperature_mean', 'rainfall_mean']
        for field in required_climate_fields:
            if field not in climate_conds:
                logger.error(f"Missing required climate condition: {field}")
                return jsonify({'error': f'Missing required climate condition: {field}'}), 400
        
        # Get farming conditions (optional)
        farming_conds = data.get('farming_conditions', {})
        
        logger.info(f"Soil properties: {soil_props}")
        logger.info(f"Climate conditions: {climate_conds}")
        logger.info(f"Farming conditions: {farming_conds}")
        
        # Get recommendation from API
        api_instance = get_api()
        result = api_instance.get_recommendation(
            soil_properties=soil_props,
            climate_conditions=climate_conds,
            available_land=farming_conds.get('available_land', 0),
            farming_conditions=farming_conds
        )
        
        logger.info(f"Generated recommendation with {len(result['suitable_crops'])} suitable crops")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_pdf', methods=['POST'])
def download_pdf():
    """Download recommendation as PDF"""
    try:
        data = request.get_json()
        logger.info(f"Received PDF download request")
        
        # Validate input
        required_fields = ['soil_properties', 'climate_conditions']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract data
        soil_props = data['soil_properties']
        climate_conds = data['climate_conditions']
        farming_conds = data.get('farming_conditions', {})
        
        # Get recommendation and suitable crops
        api_instance = get_api()
        result = api_instance.get_recommendation(
            soil_properties=soil_props,
            climate_conditions=climate_conds,
            available_land=farming_conds.get('available_land', 0),
            farming_conditions=farming_conds
        )
        
        # Get suitable crops for PDF - use the crops from the result
        suitable_crops = result.get('suitable_crops', [])
        
        # Generate PDF
        pdf_content = api_instance.generate_pdf_report(
            suitable_crops, soil_props, climate_conds, result['recommendation_text']
        )
        
        if pdf_content:
            # Create a BytesIO object for the PDF
            pdf_buffer = io.BytesIO(pdf_content)
            pdf_buffer.seek(0)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Agricultural_Recommendation_Report_{timestamp}.pdf"
            
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate PDF'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in PDF download endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(" Starting Agricultural Recommendation System...")
    api_instance = get_api()
    logger.info(f" Data loaded: {api_instance.data_loaded if hasattr(api_instance, 'data_loaded') else 'N/A'}")
    logger.info(f" Models loaded: {api_instance.models_loaded if hasattr(api_instance, 'models_loaded') else 'N/A'}")
    logger.info(f" RAG pipeline: {api_instance.rag_loaded if hasattr(api_instance, 'rag_loaded') else 'N/A'}")
    logger.info(f" LLM available: {llm_model is not None}")
    
    # Use environment variable for port (required by cloud platforms)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
