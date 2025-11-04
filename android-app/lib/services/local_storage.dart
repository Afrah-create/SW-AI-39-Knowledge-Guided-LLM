// lib/services/local_storage.dart
import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import '../models/crop_models.dart';

class LocalStorage {
  Database? _database;
  static const String _dbName = 'agricultural_app.db';
  static const int _dbVersion = 1;

  Future<void> initialize() async {
    if (_database != null) return;

    final documentsDirectory = await getApplicationDocumentsDirectory();
    final dbPath = path.join(documentsDirectory.path, _dbName);

    _database = await openDatabase(
      dbPath,
      version: _dbVersion,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE recommendations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_data TEXT NOT NULL,
            climate_data TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            created_at INTEGER NOT NULL
          )
        ''');
        
        await db.execute('''
          CREATE INDEX idx_created_at ON recommendations(created_at DESC)
        ''');
      },
    );
  }

  Future<void> saveRecommendation({
    required SoilProperties soil,
    required ClimateConditions climate,
    required List<CropRecommendation> recommendations,
  }) async {
    if (_database == null) await initialize();

    await _database!.insert('recommendations', {
      'soil_data': jsonEncode(soil.toJson()),
      'climate_data': jsonEncode(climate.toJson()),
      'recommendations': jsonEncode(
        recommendations.map((r) => {
          'crop': r.crop,
          'score': r.score,
          'violations': r.violations,
          'recommendations': r.recommendations,
          'suitable': r.suitable,
        }).toList(),
      ),
      'created_at': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<List<Map<String, dynamic>>> getHistory({int limit = 50}) async {
    if (_database == null) await initialize();

    final results = await _database!.query(
      'recommendations',
      orderBy: 'created_at DESC',
      limit: limit,
    );

    return results.map((row) {
      return {
        'id': row['id'],
        'soil': SoilProperties.fromJson(jsonDecode(row['soil_data'] as String)),
        'climate': ClimateConditions.fromJson(jsonDecode(row['climate_data'] as String)),
        'recommendations': (jsonDecode(row['recommendations'] as String) as List)
            .map((r) => CropRecommendation(
                  crop: r['crop'],
                  score: (r['score'] as num).toDouble(),
                  violations: List<String>.from(r['violations']),
                  recommendations: List<String>.from(r['recommendations']),
                  suitable: r['suitable'] as bool,
                ))
            .toList(),
        'createdAt': DateTime.fromMillisecondsSinceEpoch(row['created_at'] as int),
      };
    }).toList();
  }

  Future<void> deleteRecommendation(int id) async {
    if (_database == null) await initialize();
    await _database!.delete('recommendations', where: 'id = ?', whereArgs: [id]);
  }

  Future<void> clearHistory() async {
    if (_database == null) await initialize();
    await _database!.delete('recommendations');
  }

  Future<void> close() async {
    await _database?.close();
    _database = null;
  }
}

