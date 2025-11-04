// lib/screens/results_screen.dart
import 'package:flutter/material.dart';
import 'package:agricultural_app/models/crop_models.dart';
import 'package:agricultural_app/widgets/crop_card.dart';

class ResultsScreen extends StatelessWidget {
  final SoilProperties soil;
  final ClimateConditions climate;
  final List<CropRecommendation> recommendations;

  const ResultsScreen({
    super.key,
    required this.soil,
    required this.climate,
    required this.recommendations,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recommendations'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {
              // Share functionality
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Summary Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: const Color(0xFF2E7D32).withOpacity(0.1),
            child: Column(
              children: [
                const Text(
                  'Input Summary',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildSummaryItem('pH', '${soil.pH.toStringAsFixed(1)}'),
                    _buildSummaryItem('Temp', '${climate.temperatureMean.toStringAsFixed(0)}°C'),
                    _buildSummaryItem('Rain', '${climate.rainfallMean.toStringAsFixed(0)}mm'),
                  ],
                ),
              ],
            ),
          ),
          
          // Recommendations List
          Expanded(
            child: recommendations.isEmpty
                ? const Center(
                    child: Text('No suitable crops found'),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: recommendations.length,
                    itemBuilder: (context, index) {
                      final recommendation = recommendations[index];
                      return CropCard(
                        recommendation: recommendation,
                        rank: index + 1,
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryItem(String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Color(0xFF2E7D32),
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }
}

