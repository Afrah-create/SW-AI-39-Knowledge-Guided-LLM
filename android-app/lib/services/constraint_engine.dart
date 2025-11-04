// lib/services/constraint_engine.dart
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:agricultural_app/models/crop_models.dart';

class CropConstraints {
  final Map<String, double> pHRange;
  final double organicMatterMin;
  final Map<String, double> temperatureRange;
  final Map<String, double> rainfallRange;
  final List<String> soilTextures;
  final Map<String, double>? nitrogenRange;
  final Map<String, double>? phosphorusRange;
  final Map<String, double>? potassiumRange;

  CropConstraints({
    required this.pHRange,
    required this.organicMatterMin,
    required this.temperatureRange,
    required this.rainfallRange,
    required this.soilTextures,
    this.nitrogenRange,
    this.phosphorusRange,
    this.potassiumRange,
  });

  factory CropConstraints.fromJson(Map<String, dynamic> json) {
    return CropConstraints(
      pHRange: {
        'min': (json['pH_range']?[0] ?? json['ph_min']).toDouble(),
        'max': (json['pH_range']?[1] ?? json['ph_max']).toDouble(),
      },
      organicMatterMin: (json['organic_matter_min'] ?? 0.0).toDouble(),
      temperatureRange: {
        'min': (json['temperature_range']?[0] ?? json['temp_min']).toDouble(),
        'max': (json['temperature_range']?[1] ?? json['temp_max']).toDouble(),
      },
      rainfallRange: {
        'min': (json['rainfall_range']?[0] ?? json['rain_min']).toDouble(),
        'max': (json['rainfall_range']?[1] ?? json['rain_max']).toDouble(),
      },
      soilTextures: List<String>.from(json['soil_textures'] ?? []),
      nitrogenRange: json['nitrogen_range'] != null
          ? {'min': json['nitrogen_range'][0].toDouble(), 'max': json['nitrogen_range'][1].toDouble()}
          : null,
      phosphorusRange: json['phosphorus_range'] != null
          ? {'min': json['phosphorus_range'][0].toDouble(), 'max': json['phosphorus_range'][1].toDouble()}
          : null,
      potassiumRange: json['potassium_range'] != null
          ? {'min': json['potassium_range'][0].toDouble(), 'max': json['potassium_range'][1].toDouble()}
          : null,
    );
  }
}

class CropEvaluation {
  final bool suitable;
  final List<String> violations;
  final List<String> recommendations;
  final double suitabilityScore;

  CropEvaluation({
    required this.suitable,
    required this.violations,
    required this.recommendations,
    required this.suitabilityScore,
  });
}

class ConstraintEngine {
  Map<String, CropConstraints> _cropConstraints = {};
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) return;

    // Load from assets or use hardcoded data
    try {
      final data = await rootBundle.loadString('assets/data/crop_constraints.json');
      final json = jsonDecode(data) as Map<String, dynamic>;
      
      _cropConstraints = json.map((key, value) =>
          MapEntry(key, CropConstraints.fromJson(value)));
    } catch (e) {
      // Fallback to hardcoded constraints
      _loadDefaultConstraints();
    }
    
    _initialized = true;
  }

  void _loadDefaultConstraints() {
    _cropConstraints = {
      'maize': CropConstraints(
        pHRange: {'min': 5.5, 'max': 7.5},
        organicMatterMin: 1.0,
        temperatureRange: {'min': 18, 'max': 30},
        rainfallRange: {'min': 500, 'max': 1500},
        soilTextures: ['loam', 'clay_loam', 'sandy_loam'],
        nitrogenRange: {'min': 50, 'max': 200},
        phosphorusRange: {'min': 10, 'max': 50},
        potassiumRange: {'min': 80, 'max': 300},
      ),
      'rice': CropConstraints(
        pHRange: {'min': 5.0, 'max': 7.0},
        organicMatterMin: 2.0,
        temperatureRange: {'min': 20, 'max': 35},
        rainfallRange: {'min': 1000, 'max': 2500},
        soilTextures: ['clay', 'clay_loam'],
        nitrogenRange: {'min': 60, 'max': 250},
        phosphorusRange: {'min': 15, 'max': 60},
        potassiumRange: {'min': 100, 'max': 400},
      ),
      'beans': CropConstraints(
        pHRange: {'min': 6.0, 'max': 7.5},
        organicMatterMin: 1.5,
        temperatureRange: {'min': 15, 'max': 25},
        rainfallRange: {'min': 600, 'max': 1200},
        soilTextures: ['loam', 'sandy_loam', 'clay_loam'],
        nitrogenRange: {'min': 40, 'max': 150},
        phosphorusRange: {'min': 20, 'max': 80},
        potassiumRange: {'min': 60, 'max': 200},
      ),
      'cassava': CropConstraints(
        pHRange: {'min': 4.5, 'max': 8.0},
        organicMatterMin: 0.5,
        temperatureRange: {'min': 20, 'max': 30},
        rainfallRange: {'min': 800, 'max': 2000},
        soilTextures: ['sandy', 'sandy_loam', 'loam'],
        nitrogenRange: {'min': 30, 'max': 120},
        phosphorusRange: {'min': 5, 'max': 30},
        potassiumRange: {'min': 40, 'max': 150},
      ),
      'sweet_potato': CropConstraints(
        pHRange: {'min': 5.0, 'max': 7.5},
        organicMatterMin: 1.0,
        temperatureRange: {'min': 18, 'max': 28},
        rainfallRange: {'min': 600, 'max': 1500},
        soilTextures: ['sandy_loam', 'loam'],
        nitrogenRange: {'min': 40, 'max': 180},
        phosphorusRange: {'min': 10, 'max': 40},
        potassiumRange: {'min': 80, 'max': 250},
      ),
      'coffee': CropConstraints(
        pHRange: {'min': 5.5, 'max': 6.5},
        organicMatterMin: 2.0,
        temperatureRange: {'min': 18, 'max': 24},
        rainfallRange: {'min': 1200, 'max': 2000},
        soilTextures: ['loam', 'clay_loam'],
        nitrogenRange: {'min': 80, 'max': 200},
        phosphorusRange: {'min': 15, 'max': 50},
        potassiumRange: {'min': 120, 'max': 300},
      ),
      'cotton': CropConstraints(
        pHRange: {'min': 5.5, 'max': 8.0},
        organicMatterMin: 1.0,
        temperatureRange: {'min': 20, 'max': 35},
        rainfallRange: {'min': 500, 'max': 1200},
        soilTextures: ['loam', 'sandy_loam', 'clay_loam'],
        nitrogenRange: {'min': 60, 'max': 180},
        phosphorusRange: {'min': 10, 'max': 40},
        potassiumRange: {'min': 80, 'max': 200},
      ),
      'sugarcane': CropConstraints(
        pHRange: {'min': 5.5, 'max': 8.0},
        organicMatterMin: 1.5,
        temperatureRange: {'min': 20, 'max': 30},
        rainfallRange: {'min': 1000, 'max': 2000},
        soilTextures: ['loam', 'clay_loam'],
        nitrogenRange: {'min': 100, 'max': 300},
        phosphorusRange: {'min': 20, 'max': 60},
        potassiumRange: {'min': 150, 'max': 400},
      ),
    };
  }

  CropEvaluation evaluateCropSuitability(
    String cropName,
    SoilProperties soil,
    ClimateConditions climate,
  ) {
    if (!_cropConstraints.containsKey(cropName)) {
      return CropEvaluation(
        suitable: false,
        violations: ['Unknown crop'],
        recommendations: [],
        suitabilityScore: 0.0,
      );
    }

    final constraints = _cropConstraints[cropName]!;
    final violations = <String>[];
    final recommendations = <String>[];

    // Check pH
    if (soil.pH < constraints.pHRange['min']! || soil.pH > constraints.pHRange['max']!) {
      violations.add(
        'pH ${soil.pH.toStringAsFixed(1)} outside optimal range '
        '(${constraints.pHRange['min']}-${constraints.pHRange['max']})',
      );
      if (soil.pH < constraints.pHRange['min']!) {
        recommendations.add('Add lime to increase soil pH');
      } else {
        recommendations.add('Add sulfur to decrease soil pH');
      }
    }

    // Check organic matter
    if (soil.organicMatter < constraints.organicMatterMin) {
      violations.add(
        'Organic matter ${soil.organicMatter.toStringAsFixed(1)}% below minimum '
        '${constraints.organicMatterMin}%',
      );
      recommendations.add('Add compost or organic fertilizers');
    }

    // Check soil texture
    if (!constraints.soilTextures.contains(soil.textureClass.toLowerCase())) {
      violations.add('Soil texture "${soil.textureClass}" not optimal for $cropName');
      recommendations.add('Consider soil amendments for better texture');
    }

    // Check temperature
    if (climate.temperatureMean < constraints.temperatureRange['min']! ||
        climate.temperatureMean > constraints.temperatureRange['max']!) {
      violations.add(
        'Temperature ${climate.temperatureMean.toStringAsFixed(1)}°C outside optimal range '
        '(${constraints.temperatureRange['min']}-${constraints.temperatureRange['max']}°C)',
      );
    }

    // Check rainfall
    if (climate.rainfallMean < constraints.rainfallRange['min']! ||
        climate.rainfallMean > constraints.rainfallRange['max']!) {
      violations.add(
        'Rainfall ${climate.rainfallMean.toStringAsFixed(0)}mm outside optimal range '
        '(${constraints.rainfallRange['min']}-${constraints.rainfallRange['max']}mm)',
      );
    }

    // Check nutrients if provided
    if (soil.nitrogen != null && constraints.nitrogenRange != null) {
      if (soil.nitrogen! < constraints.nitrogenRange!['min']! ||
          soil.nitrogen! > constraints.nitrogenRange!['max']!) {
        violations.add(
          'Nitrogen ${soil.nitrogen!.toStringAsFixed(0)}ppm outside optimal range '
          '(${constraints.nitrogenRange!['min']}-${constraints.nitrogenRange!['max']}ppm)',
        );
        if (soil.nitrogen! < constraints.nitrogenRange!['min']!) {
          recommendations.add('Add nitrogen fertilizer');
        } else {
          recommendations.add('Reduce nitrogen application');
        }
      }
    }

    if (soil.phosphorus != null && constraints.phosphorusRange != null) {
      if (soil.phosphorus! < constraints.phosphorusRange!['min']! ||
          soil.phosphorus! > constraints.phosphorusRange!['max']!) {
        violations.add(
          'Phosphorus ${soil.phosphorus!.toStringAsFixed(0)}ppm outside optimal range '
          '(${constraints.phosphorusRange!['min']}-${constraints.phosphorusRange!['max']}ppm)',
        );
        if (soil.phosphorus! < constraints.phosphorusRange!['min']!) {
          recommendations.add('Add phosphorus fertilizer');
        } else {
          recommendations.add('Reduce phosphorus application');
        }
      }
    }

    if (soil.potassium != null && constraints.potassiumRange != null) {
      if (soil.potassium! < constraints.potassiumRange!['min']! ||
          soil.potassium! > constraints.potassiumRange!['max']!) {
        violations.add(
          'Potassium ${soil.potassium!.toStringAsFixed(0)}ppm outside optimal range '
          '(${constraints.potassiumRange!['min']}-${constraints.potassiumRange!['max']}ppm)',
        );
        if (soil.potassium! < constraints.potassiumRange!['min']!) {
          recommendations.add('Add potassium fertilizer');
        } else {
          recommendations.add('Reduce potassium application');
        }
      }
    }

    // Calculate suitability score
    const totalChecks = 8;
    final violationsCount = violations.length;
    final suitabilityScore = ((totalChecks - violationsCount) / totalChecks * 100).clamp(0, 100);

    return CropEvaluation(
      suitable: violationsCount <= 2,
      violations: violations,
      recommendations: recommendations,
      suitabilityScore: suitabilityScore,
    );
  }

  List<String> get availableCrops => _cropConstraints.keys.toList();
}

