// lib/widgets/crop_card.dart
import 'package:flutter/material.dart';
import 'package:agricultural_app/models/crop_models.dart';

class CropCard extends StatelessWidget {
  final CropRecommendation recommendation;
  final int rank;

  const CropCard({
    super.key,
    required this.recommendation,
    required this.rank,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: ExpansionTile(
        leading: CircleAvatar(
          backgroundColor: _getScoreColor(recommendation.score),
          child: Text(
            recommendation.cropIcon,
            style: const TextStyle(fontSize: 24),
          ),
        ),
        title: Row(
          children: [
            Text(
              '#$rank',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                recommendation.displayName,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(
                  recommendation.suitable ? Icons.check_circle : Icons.warning,
                  size: 16,
                  color: recommendation.suitable ? Colors.green : Colors.orange,
                ),
                const SizedBox(width: 4),
                Text(
                  '${recommendation.score.toStringAsFixed(0)}% Match',
                  style: TextStyle(
                    fontWeight: FontWeight.w500,
                    color: _getScoreColor(recommendation.score),
                  ),
                ),
              ],
            ),
          ],
        ),
        trailing: _buildScoreIndicator(recommendation.score),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (recommendation.violations.isNotEmpty) ...[
                  const Text(
                    'Issues:',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.orange,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...recommendation.violations.map((violation) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Icon(Icons.info_outline, size: 16, color: Colors.orange),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                violation,
                                style: const TextStyle(fontSize: 14),
                              ),
                            ),
                          ],
                        ),
                      )),
                  const SizedBox(height: 16),
                ],
                if (recommendation.recommendations.isNotEmpty) ...[
                  const Text(
                    'Recommendations:',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF2E7D32),
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...recommendation.recommendations.map((rec) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Icon(Icons.check_circle_outline,
                                size: 16, color: Color(0xFF2E7D32)),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                rec,
                                style: const TextStyle(fontSize: 14),
                              ),
                            ),
                          ],
                        ),
                      )),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScoreIndicator(double score) {
    return SizedBox(
      width: 60,
      height: 60,
      child: CircularProgressIndicator(
        value: score / 100,
        strokeWidth: 6,
        backgroundColor: Colors.grey[300],
        valueColor: AlwaysStoppedAnimation<Color>(_getScoreColor(score)),
      ),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 80) return Colors.green;
    if (score >= 60) return Colors.orange;
    return Colors.red;
  }
}

