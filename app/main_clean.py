#!/usr/bin/env python3
"""
Agricultural Recommendation System with AI Integration
Modern AI-powered interface for crop recommendations in Uganda
"""

import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Gemini API for LLM integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("✅ Google Generative AI library imported successfully")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ Google Generative AI library not available. LLM features will be disabled.")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment variables loaded from .env file")
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
            # Try different model names for compatibility
            model_names_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro',
                'gemini-1.0-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
                'models/gemini-pro'
            ]
            
            llm_model = None
            for model_name in model_names_to_try:
                try:
                    llm_model = genai.GenerativeModel(model_name)
                    logger.info(f"✅ Gemini API configured successfully with {model_name}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load {model_name}: {e}")
                    continue
            
            if llm_model is None:
                # List available models for debugging
                try:
                    models = genai.list_models()
                    available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
                    logger.info(f"Available models: {available_models}")
                    logger.error("❌ Failed to initialize any Gemini model. Please check available models above.")
                except Exception as e3:
                    logger.error(f"❌ Failed to list models: {e3}")
                    llm_model = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
    except Exception as e:
        logger.error(f"❌ Error configuring Gemini API: {e}")
        llm_model = None

app = Flask(__name__)

class DataLoader:
    """Load and manage agricultural data"""
    
    def __init__(self, data_dir="data", processed_dir="processed"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.knowledge_graph = None
        self.dataset_triples = None
        self.literature_triples = None
        self.ugandan_data = None
    
    def load_data(self):
        """Load all agricultural data files"""
        try:
            logger.info("Loading agricultural data files...")
            
            # Load unified knowledge graph from processed folder
            kg_path = os.path.join(self.processed_dir, "unified_knowledge_graph.json")
            if os.path.exists(kg_path):
                with open(kg_path, 'r', encoding='utf-8') as f:
                    self.knowledge_graph = json.load(f)
                logger.info(f"✅ Loaded knowledge graph: {len(self.knowledge_graph)} triples")
            else:
                logger.warning(f"⚠️ Knowledge graph not found at {kg_path}")
            
            # Load dataset triples
            dataset_triples_path = os.path.join(self.processed_dir, "dataset_triples.json")
            if os.path.exists(dataset_triples_path):
                with open(dataset_triples_path, 'r', encoding='utf-8') as f:
                    self.dataset_triples = json.load(f)
                logger.info(f"✅ Loaded dataset triples: {len(self.dataset_triples)} triples")
            
            # Load literature triples
            literature_triples_path = os.path.join(self.processed_dir, "literature_triples.json")
            if os.path.exists(literature_triples_path):
                with open(literature_triples_path, 'r', encoding='utf-8') as f:
                    self.literature_triples = json.load(f)
                logger.info(f"✅ Loaded literature triples: {len(self.literature_triples)} triples")
            
            # Load Ugandan dataset
            ugandan_data_path = os.path.join(self.processed_dir, "ugandan_data_cleaned.csv")
            if os.path.exists(ugandan_data_path):
                import pandas as pd
                self.ugandan_data = pd.read_csv(ugandan_data_path)
                logger.info(f"✅ Loaded Ugandan dataset: {len(self.ugandan_data)} records")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

class AgriculturalModelLoader:
    """Load and manage the trained GCN model"""
    
    def __init__(self, models_dir="processed/trained_models"):
        self.models_dir = models_dir
        self.model = None
        self.model_metadata = None
        self.entity_to_id = None
        self.id_to_entity = None
        self.relation_to_id = None
        self.id_to_relation = None
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            logger.info("Loading trained agricultural model...")
            
            # Load model metadata
            metadata_path = os.path.join(self.models_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                self.entity_to_id = self.model_metadata.get('entity_to_id', {})
                self.id_to_entity = self.model_metadata.get('id_to_entity', {})
                self.relation_to_id = self.model_metadata.get('relation_to_id', {})
                self.id_to_relation = self.model_metadata.get('id_to_relation', {})
                
                logger.info(f"✅ Loaded model metadata: {len(self.entity_to_id)} entities, {len(self.relation_to_id)} relations")
                return True
            else:
                logger.warning(f"⚠️ Model metadata not found at {metadata_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
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
            logger.info(f"✅ TF-IDF matrix created: {self.tfidf_matrix.shape}")
    
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
                'soil_textures': ['loam', 'clay_loam', 'sandy_loam']
            },
            'rice': {
                'pH_range': (5.0, 7.0),
                'organic_matter_min': 2.0,
                'temperature_range': (20, 35),
                'rainfall_range': (1000, 2500),
                'soil_textures': ['clay', 'clay_loam']
            },
            'beans': {
                'pH_range': (6.0, 7.5),
                'organic_matter_min': 1.5,
                'temperature_range': (15, 25),
                'rainfall_range': (600, 1200),
                'soil_textures': ['loam', 'sandy_loam', 'clay_loam']
            },
            'cassava': {
                'pH_range': (4.5, 8.0),
                'organic_matter_min': 0.5,
                'temperature_range': (20, 30),
                'rainfall_range': (800, 2000),
                'soil_textures': ['sandy', 'sandy_loam', 'loam']
            },
            'sweet_potato': {
                'pH_range': (5.0, 7.5),
                'organic_matter_min': 1.0,
                'temperature_range': (18, 28),
                'rainfall_range': (600, 1500),
                'soil_textures': ['sandy_loam', 'loam']
            },
            'coffee': {
                'pH_range': (5.5, 6.5),
                'organic_matter_min': 2.0,
                'temperature_range': (18, 24),
                'rainfall_range': (1200, 2000),
                'soil_textures': ['loam', 'clay_loam']
            },
            'cotton': {
                'pH_range': (5.5, 8.0),
                'organic_matter_min': 1.0,
                'temperature_range': (20, 35),
                'rainfall_range': (500, 1200),
                'soil_textures': ['loam', 'sandy_loam', 'clay_loam']
            },
            'sugarcane': {
                'pH_range': (5.5, 8.0),
                'organic_matter_min': 1.5,
                'temperature_range': (20, 30),
                'rainfall_range': (1000, 2000),
                'soil_textures': ['loam', 'clay_loam']
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
        
        # Calculate suitability score
        total_checks = 5
        violations_count = len(violations)
        suitability_score = max(0, (total_checks - violations_count) / total_checks)
        
        return {
            'suitable': violations_count <= 2,  # Allow up to 2 violations
            'violations': violations,
            'recommendations': recommendations,
            'suitability_score': suitability_score
        }

class AgriculturalAPI:
    """Agricultural recommendation API with real data integration"""
    
    def __init__(self):
        self.models_loaded = False
        self.data_loaded = False
        self.rag_loaded = False
        self.model_loader = AgriculturalModelLoader()
        self.data_loader = DataLoader()
        self.constraint_engine = AgriculturalConstraintEngine()
        self.semantic_retriever = None
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load ML models and data"""
        try:
            logger.info("Loading agricultural recommendation system...")
            
            # Load data first
            if self.data_loader.load_data():
                self.data_loaded = True
                logger.info("✅ Data loaded successfully")
            
            # Load GCN model
            if self.model_loader.load_model():
                self.models_loaded = True
                logger.info("✅ GCN model loaded successfully")
            else:
                logger.warning("⚠️ GCN model loading failed, using constraint-based recommendations only")
                self.models_loaded = False
            
            # Initialize RAG pipeline if knowledge graph is available
            if self.data_loaded and self.data_loader.knowledge_graph and self.model_loader.model_metadata:
                try:
                    self.semantic_retriever = SemanticRetriever(
                        triples_data=self.data_loader.knowledge_graph,
                        entity_to_id=self.model_loader.model_metadata['entity_to_id'],
                        id_to_entity=self.model_loader.model_metadata['id_to_entity']
                    )
                    self.rag_loaded = True
                    logger.info("✅ RAG pipeline initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ RAG pipeline initialization failed: {e}")
                    self.rag_loaded = False
            else:
                logger.warning("⚠️ RAG pipeline not initialized - missing knowledge graph or model metadata")
                self.rag_loaded = False
            
            logger.info("✅ Agricultural recommendation system initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            self.models_loaded = False
            self.data_loaded = False
    
    def get_recommendation(self, soil_properties, climate_conditions, **kwargs):
        """Get crop recommendation based on soil and climate data"""
        try:
            logger.info(f"Generating recommendation for soil: {soil_properties}, climate: {climate_conditions}")
            
            # Get all available crops
            available_crops = list(self.constraint_engine.crop_constraints.keys())
            
            # Evaluate each crop
            suitable_crops = []
            for crop in available_crops:
                evaluation = self.constraint_engine.evaluate_crop_suitability(
                    crop, soil_properties, climate_conditions
                )
                
                if evaluation['suitable']:
                    suitable_crops.append({
                        'crop': crop,
                        'suitability_score': evaluation['suitability_score'],
                        'recommendations': evaluation['recommendations'],
                        'violations': evaluation['violations']
                    })
            
            # Sort by suitability score
            suitable_crops.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            # Generate land allocation if available land is provided
            land_allocation = None
            if 'available_land' in kwargs and kwargs['available_land']:
                land_allocation = self._generate_land_allocation(suitable_crops, kwargs['available_land'])
            
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
                'data_sources': data_sources
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendation: {e}")
            raise e
    
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
        """Generate human-readable recommendation text using LLM when available"""
        if not suitable_crops:
            return "No suitable crops found for your current conditions. Consider soil amendments and management practices."
        
        # Try to use LLM for enhanced recommendations
        if llm_model:
            try:
                return self._generate_llm_recommendation(suitable_crops, soil_properties, climate_conditions)
            except Exception as e:
                logger.warning(f"LLM generation failed, falling back to template: {e}")
        
        # Fallback to simple template-based generation
        top_crop = suitable_crops[0]
        recommendation = f"Based on your soil and climate conditions, {top_crop['crop'].title()} is the most suitable crop with a {top_crop['suitability_score']:.1%} suitability score. "
        
        if len(suitable_crops) > 1:
            other_crops = [crop['crop'].title() for crop in suitable_crops[1:3]]
            recommendation += f"Other recommended crops include {', '.join(other_crops)}. "
        
        # Add specific recommendations
        if top_crop['recommendations']:
            recommendation += f"Key recommendations: {'; '.join(top_crop['recommendations'][:2])}."
        
        return recommendation
    
    def _generate_llm_recommendation(self, suitable_crops, soil_properties, climate_conditions):
        """Generate recommendation using LLM"""
        # Prepare context for LLM
        context = self._prepare_llm_context(suitable_crops, soil_properties, climate_conditions)
        
        # Create prompt for LLM
        prompt = f"""
You are an expert agricultural advisor specializing in Uganda/East Africa. Based on the provided soil and climate conditions and crop suitability analysis, provide a comprehensive crop recommendation.

{context}

Please provide:
1. A clear recommendation for the most suitable crop(s)
2. Specific management practices for the recommended crop(s)
3. Any important considerations or warnings
4. Suggestions for improving soil conditions if needed

Keep the response practical, actionable, and suitable for Ugandan farmers.
"""
        
        try:
            response = llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise e
    
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
    
    def _get_data_sources(self):
        """Get information about data sources used"""
        sources = {
            "knowledge_graph_triples": len(self.data_loader.knowledge_graph) if self.data_loader.knowledge_graph else 0,
            "dataset_triples": len(self.data_loader.dataset_triples) if self.data_loader.dataset_triples else 0,
            "literature_triples": len(self.data_loader.literature_triples) if self.data_loader.literature_triples else 0,
            "ugandan_data_points": len(self.data_loader.ugandan_data) if hasattr(self.data_loader, 'ugandan_data') and self.data_loader.ugandan_data is not None else 0,
            "rag_pipeline_active": self.rag_loaded,
            "llm_model_available": llm_model is not None
        }
        
        return sources

# Initialize the API
api = AgriculturalAPI()

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
        
        /* AI System Header */
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            padding: 30px 40px;
            text-align: center;
            position: relative;
            border-bottom: 2px solid rgba(0, 255, 150, 0.3);
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(0, 255, 150, 0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            z-index: 1;
        }
        
        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }
        
        .ai-status {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 20px;
            position: relative;
            z-index: 1;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            border-radius: 20px;
            font-size: 0.9rem;
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
            padding: 40px;
            background: rgba(15, 15, 35, 0.8);
        }
        
        .ai-chat-interface {
            background: rgba(20, 20, 40, 0.9);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .chat-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0, 255, 150, 0.2);
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
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 10px rgba(0, 255, 150, 0.5); }
            to { box-shadow: 0 0 20px rgba(0, 255, 150, 0.8); }
        }
        
        .chat-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .chat-subtitle {
            font-size: 0.9rem;
            color: #a0a0a0;
            margin-top: 5px;
        }
        
        /* Input Form Styles */
        .input-form {
            background: rgba(25, 25, 50, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(0, 255, 150, 0.1);
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
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
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid rgba(0, 255, 150, 0.2);
            border-radius: 10px;
            background: rgba(15, 15, 35, 0.8);
            color: #ffffff;
            font-size: 1rem;
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
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0 auto;
            box-shadow: 0 5px 20px rgba(0, 255, 150, 0.3);
        }
        
        .ai-submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 255, 150, 0.4);
            background: linear-gradient(135deg, #00d4aa, #00ff96);
        }
        
        .ai-submit-btn:active {
            transform: translateY(0);
        }
        
        .ai-submit-btn i {
            font-size: 1.2rem;
        }
        
        /* Results Display */
        .results-container {
            background: rgba(20, 20, 40, 0.9);
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: none;
        }
        
        .results-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0, 255, 150, 0.2);
        }
        
        .results-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #00ff96, #00d4aa);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: #000;
        }
        
        .results-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        /* Crop Cards */
        .crop-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }
        
        .crop-card {
            background: rgba(25, 25, 50, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 150, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .crop-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 255, 150, 0.2);
            border-color: rgba(0, 255, 150, 0.4);
        }
        
        .crop-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00ff96, #00d4aa);
        }
        
        .crop-name {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 10px;
        }
        
        .crop-score {
            font-size: 1.1rem;
            color: #00ff96;
            font-weight: 500;
            margin-bottom: 15px;
        }
        
        .crop-details {
            color: #a0a0a0;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .crop-details ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .crop-details li {
            margin-bottom: 5px;
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
                margin: 10px;
                border-radius: 15px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .ai-status {
                flex-direction: column;
                gap: 10px;
            }
            
            .crop-grid {
                grid-template-columns: 1fr;
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
    </style>
</head>
<body>
    <div class="container">
        <!-- AI System Header -->
        <div class="header">
            <h1><i class="fas fa-robot"></i> AgriAI</h1>
            <div class="subtitle">Intelligent Agricultural Assistant for Uganda</div>
            <div class="ai-status">
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>AI Models Active</span>
                </div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Knowledge Graph Loaded</span>
                </div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>RAG Pipeline Ready</span>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- AI Chat Interface -->
            <div class="ai-chat-interface">
                <div class="chat-header">
                    <div class="ai-avatar">
                        <i class="fas fa-seedling"></i>
                    </div>
                    <div>
                        <div class="chat-title">Agricultural Intelligence</div>
                        <div class="chat-subtitle">Powered by AI • Knowledge Graph • RAG Pipeline</div>
                    </div>
                </div>
                
                <form id="agriculturalForm" class="input-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="soil_ph"><i class="fas fa-flask"></i> Soil pH</label>
                            <input type="number" id="soil_ph" name="soil_ph" step="0.1" min="3" max="10" placeholder="Enter soil pH (e.g., 6.5)">
                        </div>
                        <div class="form-group">
                            <label for="organic_matter"><i class="fas fa-leaf"></i> Organic Matter (%)</label>
                            <input type="number" id="organic_matter" name="organic_matter" step="0.1" min="0" max="20" placeholder="Enter organic matter %">
                        </div>
                        <div class="form-group">
                            <label for="texture_class"><i class="fas fa-mountain"></i> Soil Texture</label>
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
                        </div>
                        <div class="form-group">
                            <label for="temperature_mean"><i class="fas fa-thermometer-half"></i> Temperature (°C)</label>
                            <input type="number" id="temperature_mean" name="temperature_mean" step="0.1" min="10" max="40" placeholder="Enter average temperature">
                        </div>
                        <div class="form-group">
                            <label for="rainfall_mean"><i class="fas fa-cloud-rain"></i> Rainfall (mm)</label>
                            <input type="number" id="rainfall_mean" name="rainfall_mean" step="1" min="200" max="3000" placeholder="Enter annual rainfall">
                        </div>
                        <div class="form-group">
                            <label for="available_land"><i class="fas fa-map"></i> Available Land (hectares)</label>
                            <input type="number" id="available_land" name="available_land" step="0.1" min="0.1" max="1000" placeholder="Enter land area">
                        </div>
                    </div>
                    
                    <button type="submit" class="ai-submit-btn">
                        <i class="fas fa-brain"></i>
                        Generate AI Recommendations
                    </button>
                </form>
                
                <!-- Loading Animation -->
                <div class="loading-container" id="loadingContainer">
                    <div class="ai-loader"></div>
                    <div class="loading-text">AI is analyzing your agricultural conditions...</div>
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
                    <div class="results-title">AI-Generated Recommendations</div>
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
                const data = Object.fromEntries(formData);
                
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
                        successMessage.textContent = 'AI analysis completed successfully!';
                        successMessage.style.display = 'block';
                    } else {
                        throw new Error(result.error || 'AI analysis failed');
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
                
                // Overall recommendation
                if (result.recommendation_text) {
                    html += `
                        <div class="ai-recommendation" style="background: rgba(0, 255, 150, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 25px; border-left: 4px solid #00ff96;">
                            <h3 style="color: #00ff96; margin-bottom: 15px;"><i class="fas fa-lightbulb"></i> AI Recommendation</h3>
                            <p style="color: #ffffff; line-height: 1.6;">${result.recommendation_text}</p>
                        </div>
                    `;
                }
                
                // Crop recommendations
                if (result.suitable_crops && result.suitable_crops.length > 0) {
                    html += '<div class="crop-grid">';
                    result.suitable_crops.forEach(crop => {
                        html += `
                            <div class="crop-card">
                                <div class="crop-name">${crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1)}</div>
                                <div class="crop-score">Suitability: ${(crop.suitability_score * 100).toFixed(1)}%</div>
                                <div class="crop-details">
                                    ${crop.recommendations && crop.recommendations.length > 0 ? `
                                        <strong>Recommendations:</strong>
                                        <ul>
                                            ${crop.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                        </ul>
                                    ` : ''}
                                    ${crop.violations && crop.violations.length > 0 ? `
                                        <strong>Constraints:</strong>
                                        <ul>
                                            ${crop.violations.map(viol => `<li>${viol}</li>`).join('')}
                                        </ul>
                                    ` : ''}
                                </div>
                            </div>
                        `;
                    });
                    html += '</div>';
                }
                
                // Land allocation if available
                if (result.land_allocation) {
                    html += `
                        <div style="background: rgba(25, 25, 50, 0.8); padding: 20px; border-radius: 15px; margin-top: 25px;">
                            <h3 style="color: #ffffff; margin-bottom: 15px;"><i class="fas fa-map"></i> Land Allocation Plan</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                                ${result.land_allocation.crop_details.map(crop => `
                                    <div style="background: rgba(0, 255, 150, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 255, 150, 0.2);">
                                        <div style="color: #00ff96; font-weight: 600;">${crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1)}</div>
                                        <div style="color: #ffffff; font-size: 0.9rem;">${crop.land_allocated} hectares</div>
                                        <div style="color: #a0a0a0; font-size: 0.8rem;">Score: ${(crop.suitability_score * 100).toFixed(1)}%</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
                
                resultsContent.innerHTML = html;
                resultsContainer.style.display = 'block';
                
                // Smooth scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
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
        
        logger.info(f"Soil properties: {soil_props}")
        logger.info(f"Climate conditions: {climate_conds}")
        
        # Get recommendation from API
        result = api.get_recommendation(
            soil_properties=soil_props,
            climate_conditions=climate_conds,
            available_land=data.get('available_land')
        )
        
        logger.info(f"Generated recommendation with {len(result['suitable_crops'])} suitable crops")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Agricultural Recommendation System...")
    logger.info(f"📊 Data loaded: {api.data_loaded}")
    logger.info(f"🤖 Models loaded: {api.models_loaded}")
    logger.info(f"🔍 RAG pipeline: {api.rag_loaded}")
    logger.info(f"🧠 LLM available: {llm_model is not None}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
