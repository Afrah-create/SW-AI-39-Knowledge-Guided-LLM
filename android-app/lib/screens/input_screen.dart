// lib/screens/input_screen.dart
import 'package:flutter/material.dart';
import 'package:agricultural_app/models/crop_models.dart';
import 'package:agricultural_app/screens/results_screen.dart';
import 'package:agricultural_app/services/recommendation_service.dart';
import 'package:agricultural_app/services/local_storage.dart';

class InputScreen extends StatefulWidget {
  const InputScreen({super.key});

  @override
  State<InputScreen> createState() => _InputScreenState();
}

class _InputScreenState extends State<InputScreen> {
  final _formKey = GlobalKey<FormState>();
  final _recommendationService = RecommendationService();
  final _storage = LocalStorage();

  // Soil properties
  double _ph = 7.0;
  double _organicMatter = 2.0;
  String _textureClass = 'Loam';
  double? _nitrogen;
  double? _phosphorus;
  double? _potassium;

  // Climate conditions
  double _temperature = 25.0;
  double _rainfall = 1200.0;

  bool _isLoading = false;
  final List<String> _textureOptions = [
    'Sandy',
    'Sandy Loam',
    'Loam',
    'Clay Loam',
    'Clay',
  ];

  @override
  void initState() {
    super.initState();
    _recommendationService.initialize();
    _storage.initialize();
  }

  Future<void> _getRecommendations() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final soil = SoilProperties(
        pH: _ph,
        organicMatter: _organicMatter,
        textureClass: _textureClass.toLowerCase().replaceAll(' ', '_'),
        nitrogen: _nitrogen,
        phosphorus: _phosphorus,
        potassium: _potassium,
      );

      final climate = ClimateConditions(
        temperatureMean: _temperature,
        rainfallMean: _rainfall,
      );

      final recommendations = await _recommendationService.getRecommendations(
        soil: soil,
        climate: climate,
      );

      // Save to history
      await _storage.saveRecommendation(
        soil: soil,
        climate: climate,
        recommendations: recommendations,
      );

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(
              soil: soil,
              climate: climate,
              recommendations: recommendations,
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Enter Conditions'),
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Soil Properties Section
            _buildSectionHeader('Soil Properties'),
            const SizedBox(height: 16),
            
            _buildSlider(
              label: 'pH Level',
              value: _ph,
              min: 4.0,
              max: 9.0,
              divisions: 50,
              unit: '',
              onChanged: (value) => setState(() => _ph = value),
            ),
            const SizedBox(height: 16),
            
            _buildSlider(
              label: 'Organic Matter',
              value: _organicMatter,
              min: 0.0,
              max: 10.0,
              divisions: 100,
              unit: '%',
              onChanged: (value) => setState(() => _organicMatter = value),
            ),
            const SizedBox(height: 16),
            
            _buildDropdown(
              label: 'Soil Texture',
              value: _textureClass,
              items: _textureOptions,
              onChanged: (value) => setState(() => _textureClass = value!),
            ),
            const SizedBox(height: 16),
            
            // Optional Nutrients
            ExpansionTile(
              title: const Text('Nutrients (Optional)'),
              children: [
                _buildSlider(
                  label: 'Nitrogen',
                  value: _nitrogen ?? 100.0,
                  min: 0.0,
                  max: 300.0,
                  divisions: 300,
                  unit: 'ppm',
                  onChanged: (value) => setState(() => _nitrogen = value),
                ),
                _buildSlider(
                  label: 'Phosphorus',
                  value: _phosphorus ?? 30.0,
                  min: 0.0,
                  max: 100.0,
                  divisions: 100,
                  unit: 'ppm',
                  onChanged: (value) => setState(() => _phosphorus = value),
                ),
                _buildSlider(
                  label: 'Potassium',
                  value: _potassium ?? 150.0,
                  min: 0.0,
                  max: 500.0,
                  divisions: 500,
                  unit: 'ppm',
                  onChanged: (value) => setState(() => _potassium = value),
                ),
              ],
            ),
            const SizedBox(height: 24),
            
            // Climate Conditions Section
            _buildSectionHeader('Climate Conditions'),
            const SizedBox(height: 16),
            
            _buildSlider(
              label: 'Average Temperature',
              value: _temperature,
              min: 10.0,
              max: 40.0,
              divisions: 300,
              unit: '°C',
              onChanged: (value) => setState(() => _temperature = value),
            ),
            const SizedBox(height: 16),
            
            _buildSlider(
              label: 'Annual Rainfall',
              value: _rainfall,
              min: 0.0,
              max: 3000.0,
              divisions: 300,
              unit: 'mm',
              onChanged: (value) => setState(() => _rainfall = value),
            ),
            const SizedBox(height: 32),
            
            // Submit Button
            ElevatedButton(
              onPressed: _isLoading ? null : _getRecommendations,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: const Color(0xFF2E7D32),
                foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.grey,
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    )
                  : const Text(
                      'Get Recommendations',
                      style: TextStyle(fontSize: 18),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
        color: Color(0xFF2E7D32),
      ),
    );
  }

  Widget _buildSlider({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String unit,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
            Text(
              '${value.toStringAsFixed(1)}$unit',
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFF2E7D32),
              ),
            ),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          label: '${value.toStringAsFixed(1)}$unit',
          onChanged: onChanged,
          activeColor: const Color(0xFF2E7D32),
        ),
      ],
    );
  }

  Widget _buildDropdown({
    required String label,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: value,
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          ),
          items: items.map((item) {
            return DropdownMenuItem(
              value: item,
              child: Text(item),
            );
          }).toList(),
          onChanged: onChanged,
        ),
      ],
    );
  }
}

