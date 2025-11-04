// lib/services/recommendation_service.dart
import '../models/crop_models.dart';
import 'constraint_engine.dart';
import 'model_loader.dart';

class RecommendationService {
  final ConstraintEngine _constraintEngine = ConstraintEngine();
  final ModelLoader? _modelLoader;
  bool _initialized = false;

  RecommendationService({ModelLoader? modelLoader}) : _modelLoader = modelLoader;

  Future<void> initialize() async {
    if (_initialized) return;
    await _constraintEngine.initialize();
    await _modelLoader?.loadModel();
    _initialized = true;
  }

  Future<List<CropRecommendation>> getRecommendations({
    required SoilProperties soil,
    required ClimateConditions climate,
  }) async {
    if (!_initialized) await initialize();

    final crops = _constraintEngine.availableCrops;
    final recommendations = <CropRecommendation>[];

    for (final crop in crops) {
      final evaluation = _constraintEngine.evaluateCropSuitability(
        crop,
        soil,
        climate,
      );

      // Enhance with ML model if available
      double mlScore = evaluation.suitabilityScore;
      if (_modelLoader != null && _modelLoader!.isLoaded) {
        try {
          final embedding = _getEntityEmbedding(soil, climate);
          final prediction = await _modelLoader!.predict(embedding);
          // Combine constraint score (70%) with ML score (30%)
          mlScore = (evaluation.suitabilityScore * 0.7) + (prediction * 0.3);
        } catch (e) {
          // Fallback to constraint score if ML fails
          mlScore = evaluation.suitabilityScore;
        }
      }

      recommendations.add(CropRecommendation(
        crop: crop,
        score: mlScore.clamp(0, 100),
        violations: evaluation.violations,
        recommendations: evaluation.recommendations,
        suitable: evaluation.suitable,
      ));
    }

    // Sort by score (highest first)
    recommendations.sort((a, b) => b.score.compareTo(a.score));

    // Return top 6 crops
    return recommendations.take(6).toList();
  }

  List<double> _getEntityEmbedding(SoilProperties soil, ClimateConditions climate) {
    // Convert soil/climate to embedding vector
    // This should match your Python entity encoding
    return [
      soil.pH / 14.0, // Normalize pH (0-14) to 0-1
      soil.organicMatter / 10.0, // Normalize organic matter
      climate.temperatureMean / 50.0, // Normalize temperature
      climate.rainfallMean / 2000.0, // Normalize rainfall
      // Texture encoding (one-hot like)
      _encodeTexture(soil.textureClass),
      // Nutrients if available
      (soil.nitrogen ?? 0) / 300.0,
      (soil.phosphorus ?? 0) / 100.0,
      (soil.potassium ?? 0) / 500.0,
    ];
  }

  double _encodeTexture(String texture) {
    final textures = ['sandy', 'sandy_loam', 'loam', 'clay_loam', 'clay'];
    final index = textures.indexOf(texture.toLowerCase());
    return index >= 0 ? index / textures.length.toDouble() : 0.5;
  }
}

