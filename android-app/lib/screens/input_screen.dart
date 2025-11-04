// lib/screens/input_screen.dart
import 'package:flutter/material.dart';
import '../models/crop_models.dart';
import 'results_screen.dart';
import '../services/recommendation_service.dart';
import '../services/local_storage.dart';

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
    final screenWidth = MediaQuery.of(context).size.width;
    final isSmallScreen = screenWidth < 600;
    final padding = isSmallScreen ? 16.0 : 24.0;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Enter Conditions'),
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: EdgeInsets.all(padding),
            children: [
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 700),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                  // Soil Properties Section
                  _buildSectionHeader(
                    'Soil Properties',
                    Icons.landscape,
                    isSmallScreen,
                  ),
                  SizedBox(height: isSmallScreen ? 16 : 20),
                  
                  Card(
                    elevation: 1,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(isSmallScreen ? 16 : 20),
                      child: Column(
                        children: [
                          _buildSlider(
                            label: 'pH Level',
                            icon: Icons.water_drop_outlined,
                            value: _ph,
                            min: 4.0,
                            max: 9.0,
                            divisions: 50,
                            unit: '',
                            isSmallScreen: isSmallScreen,
                            onChanged: (value) => setState(() => _ph = value),
                          ),
                          SizedBox(height: isSmallScreen ? 20 : 24),
                          _buildSlider(
                            label: 'Organic Matter',
                            icon: Icons.eco_outlined,
                            value: _organicMatter,
                            min: 0.0,
                            max: 10.0,
                            divisions: 100,
                            unit: '%',
                            isSmallScreen: isSmallScreen,
                            onChanged: (value) => setState(() => _organicMatter = value),
                          ),
                          SizedBox(height: isSmallScreen ? 20 : 24),
                          _buildDropdown(
                            label: 'Soil Texture',
                            icon: Icons.layers_outlined,
                            value: _textureClass,
                            items: _textureOptions,
                            isSmallScreen: isSmallScreen,
                            onChanged: (value) => setState(() => _textureClass = value!),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: isSmallScreen ? 16 : 20),
                  
                  // Optional Nutrients
                  Card(
                    elevation: 1,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ExpansionTile(
                      leading: const Icon(Icons.science_outlined, color: Color(0xFF2E7D32)),
                      title: Text(
                        'Nutrients (Optional)',
                        style: TextStyle(
                          fontSize: isSmallScreen ? 15 : 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      children: [
                        Padding(
                          padding: EdgeInsets.all(isSmallScreen ? 12 : 16),
                          child: Column(
                            children: [
                              _buildSlider(
                                label: 'Nitrogen',
                                icon: Icons.air_outlined,
                                value: _nitrogen ?? 100.0,
                                min: 0.0,
                                max: 300.0,
                                divisions: 300,
                                unit: 'ppm',
                                isSmallScreen: isSmallScreen,
                                onChanged: (value) => setState(() => _nitrogen = value),
                              ),
                              SizedBox(height: isSmallScreen ? 16 : 20),
                              _buildSlider(
                                label: 'Phosphorus',
                                icon: Icons.whatshot_outlined,
                                value: _phosphorus ?? 30.0,
                                min: 0.0,
                                max: 100.0,
                                divisions: 100,
                                unit: 'ppm',
                                isSmallScreen: isSmallScreen,
                                onChanged: (value) => setState(() => _phosphorus = value),
                              ),
                              SizedBox(height: isSmallScreen ? 16 : 20),
                              _buildSlider(
                                label: 'Potassium',
                                icon: Icons.bolt_outlined,
                                value: _potassium ?? 150.0,
                                min: 0.0,
                                max: 500.0,
                                divisions: 500,
                                unit: 'ppm',
                                isSmallScreen: isSmallScreen,
                                onChanged: (value) => setState(() => _potassium = value),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(height: isSmallScreen ? 24 : 28),
                  
                  // Climate Conditions Section
                  _buildSectionHeader(
                    'Climate Conditions',
                    Icons.wb_sunny_outlined,
                    isSmallScreen,
                  ),
                  SizedBox(height: isSmallScreen ? 16 : 20),
                  
                  Card(
                    elevation: 1,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(isSmallScreen ? 16 : 20),
                      child: Column(
                        children: [
                          _buildSlider(
                            label: 'Average Temperature',
                            icon: Icons.thermostat_outlined,
                            value: _temperature,
                            min: 10.0,
                            max: 40.0,
                            divisions: 300,
                            unit: '°C',
                            isSmallScreen: isSmallScreen,
                            onChanged: (value) => setState(() => _temperature = value),
                          ),
                          SizedBox(height: isSmallScreen ? 20 : 24),
                          _buildSlider(
                            label: 'Annual Rainfall',
                            icon: Icons.cloud_outlined,
                            value: _rainfall,
                            min: 0.0,
                            max: 3000.0,
                            divisions: 300,
                            unit: 'mm',
                            isSmallScreen: isSmallScreen,
                            onChanged: (value) => setState(() => _rainfall = value),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: isSmallScreen ? 32 : 40),
                  
                  // Submit Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _isLoading ? null : _getRecommendations,
                      icon: _isLoading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                            )
                          : const Icon(Icons.search, size: 24),
                      label: Text(
                        _isLoading ? 'Analyzing...' : 'Get Recommendations',
                        style: TextStyle(
                          fontSize: isSmallScreen ? 16 : 18,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                          vertical: isSmallScreen ? 16 : 18,
                          horizontal: 24,
                        ),
                        backgroundColor: const Color(0xFF2E7D32),
                        foregroundColor: Colors.white,
                        disabledBackgroundColor: Colors.grey[400],
                        elevation: 2,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: isSmallScreen ? 16 : 24),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon, bool isSmallScreen) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: const Color(0xFF2E7D32).withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: const Color(0xFF2E7D32), size: 20),
        ),
        const SizedBox(width: 12),
        Text(
          title,
          style: TextStyle(
            fontSize: isSmallScreen ? 20 : 22,
            fontWeight: FontWeight.bold,
            color: const Color(0xFF2E7D32),
          ),
        ),
      ],
    );
  }

  Widget _buildSlider({
    required String label,
    required IconData icon,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String unit,
    required bool isSmallScreen,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 18, color: Colors.grey[700]),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: isSmallScreen ? 15 : 16,
                  fontWeight: FontWeight.w500,
                  color: Colors.grey[900],
                ),
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFF2E7D32).withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '${value.toStringAsFixed(1)}$unit',
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF2E7D32),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          label: '${value.toStringAsFixed(1)}$unit',
          onChanged: onChanged,
          activeColor: const Color(0xFF2E7D32),
          inactiveColor: Colors.grey[300],
        ),
      ],
    );
  }

  Widget _buildDropdown({
    required String label,
    required IconData icon,
    required String value,
    required List<String> items,
    required bool isSmallScreen,
    required ValueChanged<String?> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 18, color: Colors.grey[700]),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                fontSize: isSmallScreen ? 15 : 16,
                fontWeight: FontWeight.w500,
                color: Colors.grey[900],
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          value: value,
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(color: Colors.grey[300]!),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(color: Colors.grey[300]!),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: Color(0xFF2E7D32), width: 2),
            ),
            contentPadding: EdgeInsets.symmetric(
              horizontal: isSmallScreen ? 16 : 20,
              vertical: isSmallScreen ? 14 : 16,
            ),
            filled: true,
            fillColor: Colors.grey[50],
          ),
          items: items.map((item) {
            return DropdownMenuItem(
              value: item,
              child: Text(item),
            );
          }).toList(),
          onChanged: onChanged,
          style: TextStyle(
            fontSize: isSmallScreen ? 15 : 16,
            color: Colors.grey[900],
          ),
        ),
      ],
    );
  }
}

