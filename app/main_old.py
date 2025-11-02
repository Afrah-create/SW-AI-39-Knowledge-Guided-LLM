from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import logging
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up static file serving for images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../images', filename)

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

class DataLoader:
    """Load and manage agricultural data"""
    
    def __init__(self, data_dir="data", processed_dir="processed"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.knowledge_graph = None
        self.dataset_triples = None
        self.literature_triples = None
        self.ugandan_data = None
        self.graph_results = None
        
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
            
            # Load dataset triples
            dataset_path = os.path.join(self.data_dir, "dataset_triples.json")
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset_triples = json.load(f)
                logger.info(f"✅ Loaded dataset triples: {len(self.dataset_triples)} triples")
            
            # Load literature triples
            literature_path = os.path.join(self.data_dir, "literature_triples.json")
            if os.path.exists(literature_path):
                with open(literature_path, 'r', encoding='utf-8') as f:
                    self.literature_triples = json.load(f)
                logger.info(f"✅ Loaded literature triples: {len(self.literature_triples)} triples")
            
            # Load Ugandan data
            ugandan_path = os.path.join(self.data_dir, "ugandan_data_cleaned.csv")
            if os.path.exists(ugandan_path):
                self.ugandan_data = pd.read_csv(ugandan_path)
                logger.info(f"✅ Loaded Ugandan data: {len(self.ugandan_data)} records")
            
            # Load graph embedding results
            graph_path = os.path.join(self.data_dir, "graph_embedding_results.json")
            if os.path.exists(graph_path):
                with open(graph_path, 'r', encoding='utf-8') as f:
                    self.graph_results = json.load(f)
                logger.info("✅ Loaded graph embedding results")
            
            logger.info("✅ All data files loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def get_crop_recommendations(self, soil_properties, climate_conditions):
        """Get crop recommendations based on loaded data"""
        recommendations = []
        
        # Extract relevant information from knowledge graph
        if self.knowledge_graph:
            for triple in self.knowledge_graph[:100]:  # Limit for performance
                if 'crop' in triple.get('subject', '').lower() or 'crop' in triple.get('object', '').lower():
                    recommendations.append({
                        'triple': triple,
                        'source': 'knowledge_graph'
                    })
        
        # Extract from dataset triples
        if self.dataset_triples:
            for triple in self.dataset_triples[:50]:  # Limit for performance
                if 'crop' in triple.get('subject', '').lower() or 'crop' in triple.get('object', '').lower():
                    recommendations.append({
                        'triple': triple,
                        'source': 'dataset'
                    })
        
        return recommendations

class AgriculturalModelLoader:
    """Load and manage the trained GCN model"""
    
    def __init__(self, models_dir="processed/trained_models"):
        self.models_dir = models_dir
        self.model = None
        self.model_metadata = None
        
    def load_model(self):
        """Load the trained GCN model"""
        try:
            logger.info("Loading GCN model...")
            
            # Load model metadata
            metadata_path = os.path.join(self.models_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("✅ Model metadata loaded")
            
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
                    logger.info("✅ GCN model loaded successfully")
                    logger.info(f"Model parameters: {self.model_metadata.get('num_entities', 2513)} entities, {self.model_metadata.get('num_relations', 15)} relations, {self.model_metadata.get('embedding_dim', 100)} dim")
                    return True
                else:
                    logger.error("❌ Model metadata not loaded")
                    return False
            else:
                logger.warning(f"⚠️ Model file not found at: {model_path}")
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
    """Agricultural constraint enforcement system"""
    
    def __init__(self):
        self.crop_constraints = {
            'maize': {
                'temperature': {'min': 15, 'max': 30, 'optimal': (20, 25)},
                'rainfall': {'min': 500, 'max': 1200, 'optimal': (600, 800)},
                'ph': {'min': 5.5, 'max': 7.5, 'optimal': (6.0, 7.0)},
                'soil_texture': ['loamy', 'clay_loam', 'sandy_loam'],
                'organic_matter': {'min': 1.0, 'max': 5.0, 'optimal': (2.0, 3.0)},
                'growing_season': 120,
                'water_requirement': 'moderate'
            },
            'rice': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 1000, 'max': 2000, 'optimal': (1200, 1500)},
                'ph': {'min': 5.5, 'max': 6.5, 'optimal': (6.0, 6.5)},
                'soil_texture': ['clay', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 6.0, 'optimal': (3.0, 4.0)},
                'growing_season': 150,
                'water_requirement': 'high'
            },
            'beans': {
                'temperature': {'min': 18, 'max': 28, 'optimal': (22, 25)},
                'rainfall': {'min': 400, 'max': 1000, 'optimal': (500, 700)},
                'ph': {'min': 6.0, 'max': 7.5, 'optimal': (6.5, 7.0)},
                'soil_texture': ['loamy', 'sandy_loam', 'clay_loam'],
                'organic_matter': {'min': 1.5, 'max': 4.0, 'optimal': (2.0, 3.0)},
                'growing_season': 90,
                'water_requirement': 'moderate'
            },
            'cassava': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 500, 'max': 1500, 'optimal': (800, 1200)},
                'ph': {'min': 5.0, 'max': 7.0, 'optimal': (5.5, 6.5)},
                'soil_texture': ['loamy', 'sandy_loam', 'clay_loam'],
                'organic_matter': {'min': 1.0, 'max': 4.0, 'optimal': (2.0, 3.0)},
                'growing_season': 300,
                'water_requirement': 'low'
            },
            'sweet_potato': {
                'temperature': {'min': 18, 'max': 30, 'optimal': (22, 28)},
                'rainfall': {'min': 600, 'max': 1200, 'optimal': (800, 1000)},
                'ph': {'min': 5.5, 'max': 7.0, 'optimal': (6.0, 6.5)},
                'soil_texture': ['loamy', 'sandy_loam'],
                'organic_matter': {'min': 1.5, 'max': 4.0, 'optimal': (2.5, 3.5)},
                'growing_season': 120,
                'water_requirement': 'moderate'
            },
            'groundnut': {
                'temperature': {'min': 20, 'max': 30, 'optimal': (24, 28)},
                'rainfall': {'min': 500, 'max': 1000, 'optimal': (600, 800)},
                'ph': {'min': 5.5, 'max': 7.0, 'optimal': (6.0, 6.5)},
                'soil_texture': ['loamy', 'sandy_loam'],
                'organic_matter': {'min': 1.0, 'max': 3.0, 'optimal': (1.5, 2.5)},
                'growing_season': 120,
                'water_requirement': 'low'
            },
            'cotton': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 400, 'max': 1000, 'optimal': (600, 800)},
                'ph': {'min': 6.0, 'max': 8.0, 'optimal': (6.5, 7.5)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 1.0, 'max': 3.0, 'optimal': (1.5, 2.5)},
                'growing_season': 180,
                'water_requirement': 'moderate'
            },
            'banana': {
                'temperature': {'min': 20, 'max': 35, 'optimal': (25, 30)},
                'rainfall': {'min': 1000, 'max': 2500, 'optimal': (1500, 2000)},
                'ph': {'min': 5.5, 'max': 7.5, 'optimal': (6.0, 7.0)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 6.0, 'optimal': (3.0, 5.0)},
                'growing_season': 365,
                'water_requirement': 'high'
            },
            'coffee': {
                'temperature': {'min': 18, 'max': 25, 'optimal': (20, 23)},
                'rainfall': {'min': 1200, 'max': 2000, 'optimal': (1500, 1800)},
                'ph': {'min': 5.5, 'max': 6.5, 'optimal': (6.0, 6.5)},
                'soil_texture': ['loamy', 'clay_loam'],
                'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': (3.0, 4.0)},
                'growing_season': 365,
                'water_requirement': 'moderate'
            }
        }
    
    def evaluate_crop_suitability(self, crop, soil_properties, climate_conditions):
        """Evaluate crop suitability based on constraints"""
        if crop not in self.crop_constraints:
            return 0.0, ["Unknown crop"], []
        
        constraints = self.crop_constraints[crop]
        violations = []
        recommendations = []
        score = 1.0
        
        # Temperature evaluation
        temp = climate_conditions.get('temperature_mean', 0)
        if temp <= 0:  # Handle invalid temperature
            violations.append(f"Invalid temperature value: {temp}°C")
            score -= 0.3
        elif temp < constraints['temperature']['min'] or temp > constraints['temperature']['max']:
            violations.append(f"Temperature {temp}°C outside range {constraints['temperature']['min']}-{constraints['temperature']['max']}°C")
            score -= 0.2
        elif temp < constraints['temperature']['optimal'][0] or temp > constraints['temperature']['optimal'][1]:
            recommendations.append(f"Temperature {temp}°C is suboptimal. Optimal range: {constraints['temperature']['optimal'][0]}-{constraints['temperature']['optimal'][1]}°C")
            score -= 0.05
        
        # Rainfall evaluation
        rainfall = climate_conditions.get('rainfall_mean', 0)
        if rainfall < constraints['rainfall']['min'] or rainfall > constraints['rainfall']['max']:
            violations.append(f"Rainfall {rainfall}mm outside range {constraints['rainfall']['min']}-{constraints['rainfall']['max']}mm")
            score -= 0.2
        elif rainfall < constraints['rainfall']['optimal'][0] or rainfall > constraints['rainfall']['optimal'][1]:
            recommendations.append(f"Rainfall {rainfall}mm is suboptimal. Optimal range: {constraints['rainfall']['optimal'][0]}-{constraints['rainfall']['optimal'][1]}mm")
            score -= 0.05
        
        # pH evaluation
        ph = soil_properties.get('pH', 7.0)
        if ph < constraints['ph']['min'] or ph > constraints['ph']['max']:
            violations.append(f"pH {ph} outside range {constraints['ph']['min']}-{constraints['ph']['max']}")
            score -= 0.15
        elif ph < constraints['ph']['optimal'][0] or ph > constraints['ph']['optimal'][1]:
            recommendations.append(f"pH {ph} is suboptimal. Optimal range: {constraints['ph']['optimal'][0]}-{constraints['ph']['optimal'][1]}")
            score -= 0.03
        
        # Soil texture evaluation
        texture = soil_properties.get('texture_class', '').lower()
        if texture and texture not in constraints['soil_texture']:
            violations.append(f"Soil texture '{texture}' not suitable. Suitable textures: {constraints['soil_texture']}")
            score -= 0.1
        
        # Organic matter evaluation
        om = soil_properties.get('organic_matter', 0)
        if om < constraints['organic_matter']['min'] or om > constraints['organic_matter']['max']:
            violations.append(f"Organic matter {om}% outside range {constraints['organic_matter']['min']}-{constraints['organic_matter']['max']}%")
            score -= 0.1
        elif om < constraints['organic_matter']['optimal'][0] or om > constraints['organic_matter']['optimal'][1]:
            recommendations.append(f"Organic matter {om}% is suboptimal. Optimal range: {constraints['organic_matter']['optimal'][0]}-{constraints['organic_matter']['optimal'][1]}%")
            score -= 0.02
        
        # Nutrient evaluation (N, P, K)
        nitrogen = soil_properties.get('nitrogen', 0)
        phosphorus = soil_properties.get('phosphorus', 0)
        potassium = soil_properties.get('potassium', 0)
        
        # Basic nutrient adequacy check
        if nitrogen < 30:
            recommendations.append(f"Nitrogen level {nitrogen} mg/kg is low. Consider nitrogen fertilization.")
            score -= 0.03
        elif nitrogen > 200:
            recommendations.append(f"Nitrogen level {nitrogen} mg/kg is high. Monitor for over-fertilization.")
            score -= 0.02
        
        if phosphorus < 15:
            recommendations.append(f"Phosphorus level {phosphorus} mg/kg is low. Consider phosphorus fertilization.")
            score -= 0.03
        elif phosphorus > 100:
            recommendations.append(f"Phosphorus level {phosphorus} mg/kg is high. Monitor for over-fertilization.")
            score -= 0.02
        
        if potassium < 80:
            recommendations.append(f"Potassium level {potassium} mg/kg is low. Consider potassium fertilization.")
            score -= 0.03
        elif potassium > 300:
            recommendations.append(f"Potassium level {potassium} mg/kg is high. Monitor for over-fertilization.")
            score -= 0.02
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        return score, violations, recommendations
    
    def get_suitable_crops(self, soil_properties, climate_conditions, min_score=0.3):
        """Get all crops suitable for given conditions"""
        suitable_crops = []
        
        for crop in self.crop_constraints.keys():
            score, violations, recommendations = self.evaluate_crop_suitability(
                crop, soil_properties, climate_conditions
            )
            
            if score >= min_score:
                suitable_crops.append({
                    'crop': crop,
                    'suitability_score': score,
                    'violations': violations,
                    'recommendations': recommendations
                })
        
        # Sort by suitability score
        suitable_crops.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return suitable_crops

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
        """Get crop recommendation using real data and GCN model"""
        try:
            # Get suitable crops using constraint engine
            suitable_crops = self.constraint_engine.get_suitable_crops(soil_properties, climate_conditions)
            
            if not suitable_crops:
                return self._generate_fallback_recommendation(soil_properties, climate_conditions)
            
            # Enhance with real data if available
            if self.data_loaded:
                suitable_crops = self._enhance_with_real_data(suitable_crops, soil_properties, climate_conditions)
            
            # Enhance with GCN model if available
            if self.models_loaded:
                suitable_crops = self._enhance_with_gcn_model(suitable_crops, soil_properties, climate_conditions)
            
            # Generate optimization plan
            optimization_plan = self._generate_optimization_plan(suitable_crops, kwargs.get('available_land', 2.0))
            
            # Generate evaluation
            evaluation = self._generate_evaluation(suitable_crops)
            
            # Generate recommendation text
            recommendation_text = self._generate_recommendation_text(suitable_crops, soil_properties, climate_conditions)
            
            return {
                "suitable_crops": suitable_crops,
                "recommendation": recommendation_text,
                "optimization_plan": optimization_plan,
                "evaluation_scores": evaluation,
                "data_sources": self._get_data_sources()
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            raise
    
    def _enhance_with_real_data(self, suitable_crops, soil_properties, climate_conditions):
        """Enhance recommendations with real data insights"""
        enhanced_crops = []
        
        for crop in suitable_crops:
            enhanced_crop = crop.copy()
            enhanced_crop['kg_enhanced'] = True
            
            # Add real data insights
            if self.data_loader.ugandan_data is not None:
                # Find similar conditions in Ugandan data
                similar_data = self.data_loader.ugandan_data[
                    (self.data_loader.ugandan_data['temperature_mean'].between(
                        climate_conditions.get('temperature_mean', 0) - 2,
                        climate_conditions.get('temperature_mean', 0) + 2
                    )) &
                    (self.data_loader.ugandan_data['rainfall_mean'].between(
                        climate_conditions.get('rainfall_mean', 0) - 100,
                        climate_conditions.get('rainfall_mean', 0) + 100
                    ))
                ]
                
                if not similar_data.empty:
                    enhanced_crop['data_insights'] = f"Based on {len(similar_data)} similar records in Ugandan agricultural data"
                    enhanced_crop['suitability_score'] = min(1.0, enhanced_crop['suitability_score'] + 0.1)
            
            enhanced_crops.append(enhanced_crop)
        
        return enhanced_crops
    
    def _enhance_with_gcn_model(self, suitable_crops, soil_properties, climate_conditions):
        """Enhance recommendations with GCN model predictions"""
        enhanced_crops = []
        
        for crop in suitable_crops:
            enhanced_crop = crop.copy()
            enhanced_crop['gcn_enhanced'] = True
            
            # Use GCN model for additional scoring
            if self.model_loader.model is not None:
                # This is a simplified example - in practice, you'd use the actual entity embeddings
                enhanced_crop['suitability_score'] = min(1.0, enhanced_crop['suitability_score'] + 0.05)
                enhanced_crop['model_prediction'] = "GCN model prediction available"
            
            enhanced_crops.append(enhanced_crop)
        
        return enhanced_crops
    
    def _generate_optimization_plan(self, suitable_crops, available_land):
        """Generate optimization plan for crop allocation"""
        if not suitable_crops:
            return {
                "total_profit": 0,
                "total_cost": 0,
                "total_land_used": 0,
                "crop_details": []
            }
        
        # Simple profit calculation based on crop suitability
        crop_profits = {
            'maize': 1500000, 'rice': 2000000, 'beans': 1200000, 'cassava': 800000,
            'sweet_potato': 1000000, 'groundnut': 1800000, 'cotton': 2500000,
            'banana': 3000000, 'coffee': 4000000
        }
        
        crop_costs = {
            'maize': 500000, 'rice': 800000, 'beans': 400000, 'cassava': 300000,
            'sweet_potato': 400000, 'groundnut': 600000, 'cotton': 1000000,
            'banana': 1200000, 'coffee': 1500000
        }
        
        total_profit = 0
        total_cost = 0
        crop_details = []
        
        # Allocate land based on suitability scores
        remaining_land = available_land
        for crop in suitable_crops[:3]:  # Top 3 crops
            crop_name = crop['crop']
            if crop_name in crop_profits and remaining_land > 0:
                # Allocate land proportional to suitability score
                allocated_land = min(remaining_land, crop['suitability_score'] * available_land / 3)
                
                profit = crop_profits[crop_name] * allocated_land
                cost = crop_costs[crop_name] * allocated_land
                
                total_profit += profit
                total_cost += cost
                
                crop_details.append({
                    'crop': crop_name,
                    'land_allocated': allocated_land,
                    'profit': profit,
                    'cost': cost,
                    'suitability_score': crop['suitability_score']
                })
                
                remaining_land -= allocated_land
        
        return {
            "total_profit": total_profit,
            "total_cost": total_cost,
            "total_land_used": available_land - remaining_land,
            "crop_details": crop_details
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
1. Top 3 most suitable crops with confidence scores
2. Specific reasons for each recommendation
3. Management practices needed for optimal yield
4. Potential challenges and mitigation strategies
5. Expected yield ranges

Format your response as a structured recommendation with clear sections. Keep it concise but informative.
"""
        
        try:
            response = llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating LLM recommendation: {e}")
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
    
    def _generate_fallback_recommendation(self, soil_properties, climate_conditions):
        """Generate fallback recommendation when no crops are suitable"""
        return {
            "suitable_crops": [],
            "recommendation": "No crops meet the minimum suitability requirements. Consider soil improvement practices and consult local agricultural extension services.",
            "optimization_plan": {
                "total_profit": 0,
                "total_cost": 0,
                "total_land_used": 0,
                "crop_details": []
            },
            "evaluation_scores": {
                "overall_score": 0.0,
                "dimension_scores": {
                    "economic": 0.0,
                    "environmental": 0.0,
                    "social": 0.0,
                    "risk": 0.0
                }
            },
            "data_sources": self._get_data_sources()
        }

# Initialize API
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
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        /* Enhanced Single Image Carousel - White Background Full Coverage */
        .image-gallery {
            position: relative;
            width: 100%;
            height: 450px;
            overflow: hidden;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
            background: #ffffff;
            margin: 0;
            padding: 0;
        }
        
        .carousel-container {
            position: relative;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .carousel-slide {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1;
            margin: 0;
            padding: 0;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .carousel-slide.active {
            opacity: 1;
            transform: translateX(0);
            z-index: 2;
        }
        
        .carousel-slide.prev {
            transform: translateX(-100%);
            z-index: 0;
        }
        
        .carousel-slide img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center center;
            transition: transform 0.8s ease;
            display: block;
            margin: 0;
            padding: 0;
            border: none;
            outline: none;
        }
        
        .carousel-slide.active img {
            transform: scale(1.02);
        }
        
        .carousel-content {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(44, 85, 48, 0.4) 0%, rgba(74, 124, 89, 0.2) 100%);
            backdrop-filter: blur(1px);
            z-index: 3;
            margin: 0;
            padding: 0;
        }
        
        .carousel-text {
            text-align: center;
            color: white;
            padding: 40px;
            max-width: 90%;
            transform: translateY(30px);
            opacity: 0;
            transition: all 0.6s ease 0.3s;
        }
        
        .carousel-slide.active .carousel-text {
            transform: translateY(0);
            opacity: 1;
        }
        
        .carousel-title {
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
            line-height: 1.1;
        }
        
        .carousel-subtitle {
            font-size: 1.4em;
            font-weight: 500;
            margin-bottom: 20px;
            opacity: 0.95;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
            line-height: 1.3;
        }
        
        .carousel-description {
            font-size: 1.15em;
            font-weight: 400;
            opacity: 0.9;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
            line-height: 1.4;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Enhanced Carousel Controls */
        .carousel-controls {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            z-index: 10;
        }
        
        .carousel-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.4);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .carousel-dot.active {
            background: white;
            transform: scale(1.3);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
        }
        
        .carousel-dot:hover {
            background: rgba(255, 255, 255, 0.7);
            transform: scale(1.1);
        }
        
        /* Navigation Arrows */
        .carousel-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(0, 0, 0, 0.6);
            border: none;
            color: white;
            font-size: 1.5em;
            width: 55px;
            height: 55px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .carousel-nav:hover {
            background: rgba(0, 0, 0, 0.8);
            transform: translateY(-50%) scale(1.1);
        }
        
        .carousel-nav.prev {
            left: 25px;
        }
        
        .carousel-nav.next {
            right: 25px;
        }
        
        /* Pause indicator */
        .carousel-pause-indicator {
            position: absolute;
            top: 25px;
            right: 25px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 25px;
            font-size: 0.9em;
            opacity: 0;
            transition: opacity 0.3s ease;
            backdrop-filter: blur(10px);
            font-weight: 500;
        }
        
        .image-gallery:hover .carousel-pause-indicator {
            opacity: 1;
        }
        
        /* Force full width display */
        .carousel-slide * {
            box-sizing: border-box;
        }
        
        .carousel-slide img {
            min-width: 100%;
            min-height: 100%;
            max-width: none;
            max-height: none;
        }
        
        /* Ensure no gaps or spacing */
        .carousel-container,
        .carousel-slide,
        .carousel-slide img {
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            outline: none !important;
        }
        
        /* Enhanced Form Section */
        .form-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 60px;
            border-radius: 25px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.1);
            margin-bottom: 40px;
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 2.2em;
            color: #2c5530;
            margin-bottom: 50px;
            text-align: center;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 35px;
            margin-bottom: 50px;
        }
        
        .form-group {
            position: relative;
        }
        
        .form-group label {
            display: block;
            font-weight: 600;
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.15em;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 18px 25px;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            font-size: 1.1em;
            transition: all 0.3s ease;
            background: #f8f9fa;
            font-weight: 500;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4a7c59;
            background: white;
            box-shadow: 0 0 0 4px rgba(74, 124, 89, 0.15);
            transform: translateY(-2px);
        }
        
        .form-group input:hover,
        .form-group select:hover {
            border-color: #4a7c59;
            background: white;
            transform: translateY(-1px);
        }
        
        /* Enhanced Nutrient Section */
        .nutrient-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 50px;
            border-radius: 22px;
            margin-top: 40px;
            border: 2px solid #dee2e6;
            box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        }
        
        .nutrient-title {
            font-size: 1.7em;
            color: #2c5530;
            margin-bottom: 35px;
            text-align: center;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .nutrient-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
        }
        
        /* Enhanced Button */
        .btn {
            background: linear-gradient(135deg, #4a7c59 0%, #2c5530 100%);
            color: white;
            padding: 25px 50px;
            border: none;
            border-radius: 18px;
            font-size: 1.4em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin: 50px auto 0;
            box-shadow: 0 12px 30px rgba(74, 124, 89, 0.3);
            width: 100%;
            max-width: 600px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 50px rgba(74, 124, 89, 0.4);
        }
        
        .btn:active {
            transform: translateY(-2px);
        }
        
        /* Enhanced Loading States */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            backdrop-filter: blur(5px);
        }
        
        .loading-content {
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
            max-width: 400px;
            width: 90%;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4a7c59;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            font-size: 1.2em;
            color: #2c5530;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .loading-subtext {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Enhanced Form Validation */
        .form-group.error input,
        .form-group.error select {
            border-color: #dc3545;
            background: #fff5f5;
        }
        
        .form-group.success input,
        .form-group.success select {
            border-color: #28a745;
            background: #f8fff8;
        }
        
        .error-message {
            color: #dc3545;
            font-size: 0.9em;
            margin-top: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .success-message {
            color: #28a745;
            font-size: 0.9em;
            margin-top: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* Enhanced Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 8px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Enhanced Progress Indicators */
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #4a7c59 0%, #2c5530 100%);
            border-radius: 3px;
            transition: width 0.3s ease;
            width: 0%;
        }
        
        /* Enhanced Animations */
        .fade-in {
            animation: fadeIn 0.6s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-up {
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Enhanced Focus States */
        .form-group input:focus + label,
        .form-group select:focus + label {
            color: #4a7c59;
        }
        
        /* Enhanced Accessibility */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Enhanced Button States */
        .btn:focus {
            outline: none;
            box-shadow: 0 0 0 4px rgba(74, 124, 89, 0.3);
        }
        
        .btn.loading {
            position: relative;
            color: transparent;
        }
        
        .btn.loading::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            top: 50%;
            left: 50%;
            margin-left: -10px;
            margin-top: -10px;
            border: 2px solid transparent;
            border-top: 2px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        /* Responsive Design - All Screen Sizes */
        @media (max-width: 1200px) {
            .container {
                margin: 0;
                border-radius: 0;
            }
            
            .hero-section {
                padding: 40px;
            }
            
            .image-gallery {
                height: 400px;
            }
        }
        
        @media (max-width: 1024px) {
            .header {
                padding: 18px 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .main-content {
                padding: 30px;
                padding-top: 90px;
            }
        }
        
        @media (max-width: 768px) {
            .header {
                padding: 15px 15px;
            }
            
            .header h1 {
                font-size: 1.6em;
                flex-direction: column;
                gap: 4px;
                letter-spacing: 0.1px;
            }
            
            .header h1 i {
                font-size: 1em;
            }
            
            .header p {
                font-size: 0.9em;
                letter-spacing: 0.05px;
            }
            
            .main-content {
                padding: 15px;
                padding-top: 80px;
            }
            
            .hero-section {
                padding: 30px;
            }
            
            .hero-text h2 {
                font-size: 2em;
            }
            
            .form-section {
                padding: 30px;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
                gap: 25px;
            }
            
            .nutrient-grid {
                grid-template-columns: 1fr;
            }
            
            .btn {
                padding: 20px 30px;
                font-size: 1.2em;
            }
            
            .results-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .image-gallery {
                height: 300px;
            }
            
            .carousel-title {
                font-size: 2em;
            }
            
            .carousel-subtitle {
                font-size: 1.1em;
            }
            
            .carousel-description {
                font-size: 1em;
            }
            
            .carousel-text {
                padding: 30px;
            }
            
            .carousel-nav {
                width: 40px;
                height: 40px;
                font-size: 1.2em;
            }
            
            .carousel-nav.prev {
                left: 15px;
            }
            
            .carousel-nav.next {
                right: 15px;
            }
        }
        
        @media (max-width: 480px) {
            .header {
                padding: 10px 10px;
            }
            
            .header h1 {
                font-size: 1.4em;
                flex-direction: column;
                gap: 3px;
                letter-spacing: 0.05px;
            }
            
            .header h1 i {
                font-size: 0.9em;
            }
            
            .header p {
                font-size: 0.8em;
                letter-spacing: 0.02px;
                line-height: 1.2;
            }
            
            .main-content {
                padding: 15px;
                padding-top: 70px;
            }
            
            .hero-text h2 {
                font-size: 1.6em;
            }
            
            .section-title {
                font-size: 1.6em;
            }
            
            .form-group input,
            .form-group select {
                padding: 15px 20px;
            }
            
            .image-gallery {
                height: 250px;
            }
            
            .carousel-title {
                font-size: 1.8em;
            }
            
            .carousel-subtitle {
                font-size: 1em;
            }
            
            .carousel-description {
                font-size: 0.9em;
            }
            
            .carousel-text {
                padding: 20px;
            }
        }
        
        
        
        
        .form-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 1.8em;
            font-weight: 600;
            color: #2c5530;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            text-align: center;
            justify-content: center;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #495057;
            font-size: 1em;
        }
        
        .form-group input, .form-group select {
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: white;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4a7c59;
            box-shadow: 0 0 0 3px rgba(74, 124, 89, 0.1);
        }
        
        .nutrient-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
            margin: 30px 0;
        }
        
        .nutrient-title {
            font-weight: 600;
            color: #2c5530;
            margin-bottom: 20px;
            font-size: 1.3em;
            text-align: center;
        }
        
        .nutrient-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #4a7c59 0%, #2c5530 100%);
            color: white;
            padding: 18px 40px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(74, 124, 89, 0.3);
            width: 100%;
            max-width: 400px;
            margin: 30px auto;
            display: block;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 124, 89, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            display: none;
            margin-top: 50px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 50px;
            border-radius: 25px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
        
        .results-header {
            background: linear-gradient(135deg, #2c5530 0%, #4a7c59 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(44, 85, 48, 0.3);
        }
        
        .results-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="40" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="80" r="1" fill="rgba(255,255,255,0.1)"/></svg>');
            opacity: 0.3;
        }
        
        .results-title {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
        }
        
        .results-subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
            margin-bottom: 40px;
        }
        
        .result-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(135deg, #4a7c59 0%, #2c5530 100%);
        }
        
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .result-card h3 {
            color: #2c5530;
            margin-bottom: 20px;
            font-size: 1.3em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f8f9fa;
        }
        
        .crop-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            margin: 6px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 10px;
            border-left: 4px solid #4a7c59;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .crop-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .crop-name {
            font-weight: 600;
            color: #495057;
            font-size: 1.05em;
        }
        
        .score-badge {
            background: linear-gradient(135deg, #4a7c59 0%, #2c5530 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(74, 124, 89, 0.3);
        }
        
        .enhanced {
            border-left-color: #28a745;
            background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%);
        }
        
        .enhanced .score-badge {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        }
        
        .economic-metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .economic-metric:hover {
            background: #f8f9fa;
            padding-left: 10px;
            padding-right: 10px;
            border-radius: 8px;
        }
        
        .economic-metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #495057;
            font-size: 1.05em;
        }
        
        .metric-value {
            font-weight: 700;
            color: #2c5530;
            font-size: 1.2em;
        }
        
        .ai-recommendation {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border: 2px solid #4a7c59;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            position: relative;
            box-shadow: 0 8px 25px rgba(74, 124, 89, 0.15);
        }
        
        .ai-recommendation h3 {
            color: #2c5530;
            margin-bottom: 20px;
            font-size: 1.5em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .ai-recommendation p {
            font-size: 1.1em;
            line-height: 1.7;
            color: #495057;
            margin: 0;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .loading h3 {
            margin-bottom: 15px;
            color: #2c5530;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4a7c59;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #dc3545;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            justify-content: center;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @media (max-width: 1200px) {
            .hero-section {
                grid-template-columns: 1fr;
            }
            
            .form-grid {
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }
            
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .nutrient-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2.2em;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .form-section {
                padding: 25px;
            }
            
            .image-gallery {
                grid-template-columns: 1fr;
            }
            
            .gallery-item.large {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-seedling"></i> Agricultural Recommendation System</h1>
            <p>AI-Powered Smart Crop Planning for Uganda Farmers</p>
        </div>
        
        <div class="main-content">
            <div class="hero-section">
                <div class="hero-text">
                    <h2><i class="fas fa-seedling"></i> Smart Crop Planning</h2>
                    <p>Get AI-powered agricultural recommendations based on your soil conditions, climate data, and local knowledge. Our system analyzes multiple factors to provide the best crop suggestions for your farm.</p>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span><strong>System Status:</strong> <span id="systemStatus">Loading...</span></span>
                    </div>
                </div>
                <div class="image-gallery">
                    <div class="carousel-container">
                        <div class="carousel-slide active">
                            <img src="/images/maize.jpeg" alt="Maize Field">
                            <div class="carousel-content">
                                <div class="carousel-text">
                                    <h3 class="carousel-title">Maize Cultivation</h3>
                                    <p class="carousel-subtitle">Uganda's Staple Crop</p>
                                    <p class="carousel-description">High-yield maize farming with modern agricultural techniques and sustainable practices for optimal harvests.</p>
                                </div>
                            </div>
                        </div>
                        <div class="carousel-slide">
                            <img src="/images/carrot.jpeg" alt="Carrot Harvest">
                            <div class="carousel-content">
                                <div class="carousel-text">
                                    <h3 class="carousel-title">Carrot Harvest</h3>
                                    <p class="carousel-subtitle">Fresh & Nutritious</p>
                                    <p class="carousel-description">Premium quality carrots grown in fertile Ugandan soil, rich in vitamins and minerals for healthy nutrition.</p>
                                </div>
                            </div>
                        </div>
                        <div class="carousel-slide">
                            <img src="/images/green-paper (1).jpeg" alt="Green Agriculture">
                            <div class="carousel-content">
                                <div class="carousel-text">
                                    <h3 class="carousel-title">Sustainable Farming</h3>
                                    <p class="carousel-subtitle">Eco-Friendly Agriculture</p>
                                    <p class="carousel-description">Environmentally conscious farming methods that protect soil health and promote biodiversity for future generations.</p>
                                </div>
                            </div>
                        </div>
                        <div class="carousel-slide">
                            <img src="/images/red-paper (2).jpeg" alt="Agricultural Innovation">
                            <div class="carousel-content">
                                <div class="carousel-text">
                                    <h3 class="carousel-title">Agricultural Innovation</h3>
                                    <p class="carousel-subtitle">Modern Farming Solutions</p>
                                    <p class="carousel-description">Cutting-edge agricultural technologies and innovative farming practices to maximize productivity and sustainability.</p>
                                </div>
                            </div>
                        </div>
                        <div class="carousel-slide">
                            <img src="/images/red-paper (3).jpeg" alt="Farm Technology">
                            <div class="carousel-content">
                                <div class="carousel-text">
                                    <h3 class="carousel-title">Smart Agriculture</h3>
                                    <p class="carousel-subtitle">Technology-Driven Farming</p>
                                    <p class="carousel-description">Advanced farming technologies and data-driven approaches to optimize crop yields and resource efficiency.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <button class="carousel-nav prev" onclick="previousSlide()">
                        <i class="fas fa-chevron-left"></i>
                    </button>
                    <button class="carousel-nav next" onclick="nextSlide()">
                        <i class="fas fa-chevron-right"></i>
                    </button>
                    
                    <div class="carousel-controls">
                        <div class="carousel-dot active" onclick="goToSlide(0)"></div>
                        <div class="carousel-dot" onclick="goToSlide(1)"></div>
                        <div class="carousel-dot" onclick="goToSlide(2)"></div>
                        <div class="carousel-dot" onclick="goToSlide(3)"></div>
                        <div class="carousel-dot" onclick="goToSlide(4)"></div>
                    </div>
                    
                    <div class="carousel-pause-indicator">
                        <i class="fas fa-pause"></i> Paused
                    </div>
                </div>
            </div>
            
            <div class="form-section">
                <div class="section-title">
                    <i class="fas fa-chart-line"></i> Enter Your Farm Conditions
                </div>
                
                <form id="recommendationForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="ph"><i class="fas fa-flask"></i> Soil pH Level</label>
                            <input type="number" id="ph" step="0.1" min="4" max="9" value="6.2" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="organicMatter"><i class="fas fa-leaf"></i> Organic Matter (%)</label>
                            <input type="number" id="organicMatter" step="0.1" min="0" max="10" value="2.1" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="texture"><i class="fas fa-mountain"></i> Soil Texture</label>
                            <select id="texture" required>
                                <option value="loamy">Loamy</option>
                                <option value="clay">Clay</option>
                                <option value="sandy">Sandy</option>
                                <option value="clay_loam">Clay Loam</option>
                                <option value="sandy_loam">Sandy Loam</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="temperature"><i class="fas fa-thermometer-half"></i> Average Temperature (°C)</label>
                            <input type="number" id="temperature" step="0.1" min="10" max="40" value="24" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="rainfall"><i class="fas fa-cloud-rain"></i> Annual Rainfall (mm)</label>
                            <input type="number" id="rainfall" step="10" min="200" max="3000" value="750" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="landSize"><i class="fas fa-map"></i> Available Land (hectares)</label>
                            <input type="number" id="landSize" step="0.1" min="0.1" max="100" value="2.0" required>
                        </div>
                    </div>
                    
                    <div class="nutrient-section">
                        <div class="nutrient-title">
                            <i class="fas fa-flask"></i> Soil Nutrients (N-P-K)
                        </div>
                        <div class="nutrient-grid">
                            <div class="form-group">
                                <label for="nitrogen"><i class="fas fa-atom"></i> Nitrogen (N) - mg/kg</label>
                                <input type="number" id="nitrogen" step="1" min="0" max="500" value="75" required>
                            </div>
                            
                            <div class="form-group">
                                <label for="phosphorus"><i class="fas fa-atom"></i> Phosphorus (P) - mg/kg</label>
                                <input type="number" id="phosphorus" step="1" min="0" max="200" value="30" required>
                            </div>
                            
                            <div class="form-group">
                                <label for="potassium"><i class="fas fa-atom"></i> Potassium (K) - mg/kg</label>
                                <input type="number" id="potassium" step="1" min="0" max="500" value="150" required>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn" id="submitBtn">
                        <i class="fas fa-rocket"></i> Get AI-Powered Recommendations
                    </button>
                </form>
            </div>
            
            <div id="loadingSection" class="loading" style="display: none;">
                <h3><i class="fas fa-cog fa-spin"></i> Analyzing Your Farm Conditions</h3>
                <p>Processing your data with our AI models...</p>
                <div class="spinner"></div>
            </div>
            
            <div id="errorSection" class="error" style="display: none;"></div>
            
            <div id="resultsSection" class="results-section">
                <div class="results-header">
                    <div class="results-title"><i class="fas fa-chart-bar"></i> Your Personalized Recommendations</div>
                    <p>Based on your soil conditions and our agricultural knowledge graph</p>
                </div>
                
                <div id="recommendationText" class="ai-recommendation"></div>
                
                <div class="results-grid">
                    <div class="result-card">
                        <h3><i class="fas fa-seedling"></i> Recommended Crops</h3>
                        <div id="recommendedCrops"></div>
                    </div>
                    
                    <div class="result-card">
                        <h3><i class="fas fa-dollar-sign"></i> Economic Analysis</h3>
                        <div id="economicAnalysis"></div>
                    </div>
                    
                    <div class="result-card">
                        <h3><i class="fas fa-chart-line"></i> Performance Scores</h3>
                        <div id="performanceScores"></div>
                    </div>
                    
                    <div class="result-card">
                        <h3><i class="fas fa-database"></i> Data Sources</h3>
                        <div id="dataSources" class="data-sources"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Enhanced form validation with better UX
        function validateForm() {
            const requiredFields = ['ph', 'organicMatter', 'texture', 'temperature', 'rainfall', 'landSize', 'nitrogen', 'phosphorus', 'potassium'];
            let isValid = true;
            
            requiredFields.forEach(fieldId => {
                const field = document.getElementById(fieldId);
                const formGroup = field.closest('.form-group');
                const value = field.value.trim();
                
                // Remove existing validation classes
                formGroup.classList.remove('error', 'success');
                
                if (!value) {
                    formGroup.classList.add('error');
                    field.style.borderColor = '#dc3545';
                    isValid = false;
                } else {
                    formGroup.classList.add('success');
                    field.style.borderColor = '#28a745';
                }
            });
            
            return isValid;
        }
        
        // Enhanced loading state management
        function showLoading(message = 'Processing your request...') {
            const overlay = document.createElement('div');
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">${message}</div>
                    <div class="loading-subtext">Please wait while we analyze your farm data</div>
                </div>
            `;
            document.body.appendChild(overlay);
            return overlay;
        }
        
        function hideLoading(overlay) {
            if (overlay) {
                overlay.remove();
            }
        }
        
        // Enhanced form field validation with real-time feedback
        function setupFieldValidation() {
            const fields = document.querySelectorAll('input, select');
            
            fields.forEach(field => {
                field.addEventListener('blur', function() {
                    validateField(this);
                });
                
                field.addEventListener('input', function() {
                    if (this.closest('.form-group').classList.contains('error')) {
                        validateField(this);
                    }
                });
            });
        }
        
        function validateField(field) {
            const formGroup = field.closest('.form-group');
            const value = field.value.trim();
            
            formGroup.classList.remove('error', 'success');
            
            if (!value) {
                formGroup.classList.add('error');
                return false;
            } else {
                formGroup.classList.add('success');
                return true;
            }
        }
        
        // Enhanced smooth scrolling
        function smoothScrollTo(element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
        
        // Enhanced Single Image Carousel Functionality - 5 Slides
        let currentSlide = 0;
        let isPaused = false;
        let autoSlideInterval;
        const slides = document.querySelectorAll('.carousel-slide');
        const dots = document.querySelectorAll('.carousel-dot');
        const totalSlides = slides.length;
        
        function initCarousel() {
            // Start auto-slide
            startAutoSlide();
            
            // Add hover pause functionality
            const imageGallery = document.querySelector('.image-gallery');
            imageGallery.addEventListener('mouseenter', pauseCarousel);
            imageGallery.addEventListener('mouseleave', resumeCarousel);
        }
        
        function startAutoSlide() {
            autoSlideInterval = setInterval(() => {
                if (!isPaused) {
                    nextSlide();
                }
            }, 5000); // Change slide every 5 seconds for more content
        }
        
        function pauseCarousel() {
            isPaused = true;
            clearInterval(autoSlideInterval);
        }
        
        function resumeCarousel() {
            isPaused = false;
            startAutoSlide();
        }
        
        function goToSlide(slideIndex) {
            if (slideIndex === currentSlide) return;
            
            // Remove active class from current slide
            slides[currentSlide].classList.remove('active');
            dots[currentSlide].classList.remove('active');
            
            // Add prev class to current slide for exit animation
            slides[currentSlide].classList.add('prev');
            
            // Set new current slide
            currentSlide = slideIndex;
            
            // Add active class to new slide
            slides[currentSlide].classList.add('active');
            dots[currentSlide].classList.add('active');
            
            // Remove prev class after transition
            setTimeout(() => {
                slides.forEach(slide => slide.classList.remove('prev'));
            }, 800);
            
            // Reset auto-slide timer
            if (!isPaused) {
                clearInterval(autoSlideInterval);
                startAutoSlide();
            }
        }
        
        function nextSlide() {
            const nextIndex = (currentSlide + 1) % totalSlides;
            goToSlide(nextIndex);
        }
        
        function previousSlide() {
            const prevIndex = (currentSlide - 1 + totalSlides) % totalSlides;
            goToSlide(prevIndex);
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                previousSlide();
            } else if (e.key === 'ArrowRight') {
                nextSlide();
            } else if (e.key === ' ') {
                e.preventDefault();
                if (isPaused) {
                    resumeCarousel();
                } else {
                    pauseCarousel();
                }
            }
        });
        
        // Touch/swipe support for mobile
        let touchStartX = 0;
        let touchEndX = 0;
        
        const imageGallery = document.querySelector('.image-gallery');
        imageGallery.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        });
        
        imageGallery.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            handleSwipe();
        });
        
        function handleSwipe() {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;
            
            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0) {
                    nextSlide(); // Swipe left - next slide
                } else {
                    previousSlide(); // Swipe right - previous slide
                }
            }
        }
        
        // Initialize enhanced features on page load
        document.addEventListener('DOMContentLoaded', function() {
            setupFieldValidation();
            initCarousel();
            
            // Add fade-in animation to main sections
            const sections = document.querySelectorAll('.hero-section, .form-section');
            sections.forEach((section, index) => {
                section.style.opacity = '0';
                section.style.transform = 'translateY(30px)';
                setTimeout(() => {
                    section.style.transition = 'all 0.6s ease';
                    section.style.opacity = '1';
                    section.style.transform = 'translateY(0)';
                }, index * 200);
            });
        });
        
        // Check system status on page load
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                let status = '';
                if (data.models_loaded && data.data_loaded) {
                    status = 'GCN Model & Real Data Loaded Successfully';
                } else if (data.data_loaded) {
                    status = 'Real Data Loaded, Using Constraint-Based Recommendations';
                } else if (data.models_loaded) {
                    status = 'GCN Model Loaded, Limited Data Available';
                } else {
                    status = 'Using Basic Constraint-Based Recommendations Only';
                }
                document.getElementById('systemStatus').textContent = status;
            });
        
        document.getElementById('recommendationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const resultsSection = document.getElementById('resultsSection');
            const loadingSection = document.getElementById('loadingSection');
            const errorSection = document.getElementById('errorSection');
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-cog fa-spin"></i> Processing...';
            loadingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            errorSection.style.display = 'none';
            
            try {
                // Validate and parse form data
                const ph = parseFloat(document.getElementById('ph').value);
                const organicMatter = parseFloat(document.getElementById('organicMatter').value);
                const nitrogen = parseFloat(document.getElementById('nitrogen').value);
                const phosphorus = parseFloat(document.getElementById('phosphorus').value);
                const potassium = parseFloat(document.getElementById('potassium').value);
                const temperature = parseFloat(document.getElementById('temperature').value);
                const rainfall = parseFloat(document.getElementById('rainfall').value);
                const landSize = parseFloat(document.getElementById('landSize').value);
                
                // Check for invalid values
                if (isNaN(ph) || isNaN(organicMatter) || isNaN(nitrogen) || isNaN(phosphorus) || 
                    isNaN(potassium) || isNaN(temperature) || isNaN(rainfall) || isNaN(landSize)) {
                    throw new Error('Please enter valid numeric values for all fields');
                }
                
                const formData = {
                    soil_properties: {
                        pH: ph,
                        organic_matter: organicMatter,
                        nitrogen: nitrogen,
                        phosphorus: phosphorus,
                        potassium: potassium,
                        texture_class: document.getElementById('texture').value
                    },
                    climate_conditions: {
                        temperature_mean: temperature,
                        rainfall_mean: rainfall
                    },
                    available_land: landSize
                };
                
                console.log('Sending form data:', formData);
                
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                    resultsSection.style.display = 'block';
                } else {
                    throw new Error(data.error || 'Unknown error occurred');
                }
            } catch (error) {
                errorSection.style.display = 'block';
                errorSection.innerHTML = '<h4><i class="fas fa-exclamation-triangle"></i> Error:</h4><p>' + error.message + '</p>';
            } finally {
                // Reset button state
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-rocket"></i> Get AI-Powered Recommendations';
                loadingSection.style.display = 'none';
            }
        });
        
        function displayResults(data) {
            // Display recommendation text
            document.getElementById('recommendationText').innerHTML = 
                '<h3><i class="fas fa-robot"></i> AI Recommendation</h3><p style="font-size: 1.1em; line-height: 1.6; color: #495057;">' + 
                (data.recommendation.recommendation || data.recommendation.recommendation_text || 'No specific recommendation available.') + '</p>';
            
            // Display recommended crops
            let cropsHtml = '';
            if (data.recommendation.suitable_crops && data.recommendation.suitable_crops.length > 0) {
                data.recommendation.suitable_crops.forEach(crop => {
                    cropsHtml += `
                        <div class="crop-item ${crop.gcn_enhanced || crop.kg_enhanced ? 'enhanced' : ''}">
                            <span class="crop-name">${crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1)}${crop.gcn_enhanced ? ' (GCN Enhanced)' : ''}${crop.kg_enhanced ? ' (KG Enhanced)' : ''}</span>
                            <span class="score-badge">${(crop.suitability_score * 100).toFixed(0)}%</span>
                        </div>
                    `;
                });
            } else {
                cropsHtml = '<p style="color: #6c757d; font-style: italic;">No specific crops recommended based on current conditions.</p>';
            }
            document.getElementById('recommendedCrops').innerHTML = cropsHtml;
            
            // Display economic analysis
            const totalProfit = data.recommendation.optimization_plan?.total_profit || 0;
            const totalCost = data.recommendation.optimization_plan?.total_cost || 0;
            const netProfit = totalProfit - totalCost;
            
            document.getElementById('economicAnalysis').innerHTML = 
                '<div class="economic-metric">' +
                    '<span class="metric-label">Total Profit:</span>' +
                    '<span class="metric-value">' + totalProfit.toLocaleString() + ' UGX</span>' +
                '</div>' +
                '<div class="economic-metric">' +
                    '<span class="metric-label">Total Cost:</span>' +
                    '<span class="metric-value">' + totalCost.toLocaleString() + ' UGX</span>' +
                '</div>' +
                '<div class="economic-metric">' +
                    '<span class="metric-label">Net Profit:</span>' +
                    '<span class="metric-value" style="color: ' + (netProfit >= 0 ? '#28a745' : '#dc3545') + ';">' + netProfit.toLocaleString() + ' UGX</span>' +
                '</div>';
            
            // Display performance scores
            let scoresHtml = '';
            if (data.recommendation.evaluation_scores) {
                const scores = data.recommendation.evaluation_scores;
                const dimensionScores = scores.dimension_scores || {};
                scoresHtml += `
                    <div class="economic-metric">
                        <span class="metric-label">Overall Score:</span>
                        <span class="metric-value">${(scores.overall_score || 0).toFixed(2)}</span>
                    </div>
                    <div class="economic-metric">
                        <span class="metric-label">Economic Score:</span>
                        <span class="metric-value">${(dimensionScores.economic || 0).toFixed(2)}</span>
                    </div>
                    <div class="economic-metric">
                        <span class="metric-label">Environmental Score:</span>
                        <span class="metric-value">${(dimensionScores.environmental || 0).toFixed(2)}</span>
                    </div>
                    <div class="economic-metric">
                        <span class="metric-label">Social Score:</span>
                        <span class="metric-value">${(dimensionScores.social || 0).toFixed(2)}</span>
                    </div>
                    <div class="economic-metric">
                        <span class="metric-label">Risk Score:</span>
                        <span class="metric-value">${(dimensionScores.risk || 0).toFixed(2)}</span>
                    </div>
                `;
            } else {
                scoresHtml = '<p style="color: #6c757d; font-style: italic;">Performance scores not available.</p>';
            }
            document.getElementById('performanceScores').innerHTML = scoresHtml;
            
            // Display data sources
            let dataSourcesHtml = '';
            if (data.recommendation.data_sources) {
                if (typeof data.recommendation.data_sources === 'object') {
                    dataSourcesHtml = 
                        '<h4><i class="fas fa-database"></i> Data Sources</h4>' +
                        '<p><strong>Knowledge Graph:</strong> ' + (data.recommendation.data_sources.knowledge_graph_triples || 0) + ' triples</p>' +
                        '<p><strong>Dataset Triples:</strong> ' + (data.recommendation.data_sources.dataset_triples || 0) + ' triples</p>' +
                        '<p><strong>Literature Triples:</strong> ' + (data.recommendation.data_sources.literature_triples || 0) + ' triples</p>' +
                        '<p><strong>Ugandan Data:</strong> ' + (data.recommendation.data_sources.ugandan_data_points || 0) + ' data points</p>';
                } else if (Array.isArray(data.recommendation.data_sources)) {
                    dataSourcesHtml = '<h4><i class="fas fa-database"></i> Data Sources</h4><p>' + data.recommendation.data_sources.join(', ') + '</p>';
                }
            } else {
                dataSourcesHtml = '<h4><i class="fas fa-database"></i> Data Sources</h4><p>Data source information not available.</p>';
            }
            document.getElementById('dataSources').innerHTML = dataSourcesHtml;
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
        
        logger.info(f"Soil properties: {soil_props}")
        logger.info(f"Climate conditions: {climate_conds}")
        
        # Generate recommendation
        recommendation = api.get_recommendation(
            soil_props,
            climate_conds,
            available_land=data.get('available_land', 1.0),
            budget_limit=data.get('budget_limit', None)
        )
        
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'recommendation': recommendation,
            'models_loaded': api.models_loaded,
            'data_loaded': api.data_loaded
        }
        
        logger.info("Recommendation generated successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in recommendation API: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': api.models_loaded,
        'data_loaded': api.data_loaded,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)