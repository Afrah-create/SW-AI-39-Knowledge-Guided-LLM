// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'input_screen.dart';
import '../services/local_storage.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final LocalStorage _storage = LocalStorage();
  List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    await _storage.initialize();
    final history = await _storage.getHistory(limit: 5);
    setState(() {
      _history = history;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Recommendations'),
        actions: [
          if (_history.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.history),
              onPressed: () {
                // Navigate to history screen
              },
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Welcome Card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    const Icon(
                      Icons.agriculture,
                      size: 64,
                      color: Color(0xFF2E7D32),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Agricultural Crop Recommendations',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Get personalized crop recommendations based on your soil and climate conditions',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            
            // Get Started Button
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const InputScreen(),
                  ),
                ).then((_) => _loadHistory());
              },
              icon: const Icon(Icons.science),
              label: const Text(
                'Get Recommendations',
                style: TextStyle(fontSize: 18),
              ),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: const Color(0xFF2E7D32),
                foregroundColor: Colors.white,
              ),
            ),
            const SizedBox(height: 32),
            
            // Recent History
            if (_history.isNotEmpty) ...[
              const Text(
                'Recent Recommendations',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              ..._history.map((item) => Card(
                    margin: const EdgeInsets.only(bottom: 12),
                    child: ListTile(
                      leading: const Icon(Icons.history, color: Color(0xFF2E7D32)),
                      title: Text(
                        '${item['soil'].pH.toStringAsFixed(1)} pH • ${item['climate'].temperatureMean.toStringAsFixed(0)}°C',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      subtitle: Text(
                        '${item['recommendations'].length} crops recommended',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                      trailing: Text(
                        _formatDate(item['createdAt'] as DateTime),
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[600],
                        ),
                      ),
                      onTap: () {
                        // Show details
                      },
                    ),
                  )),
            ],
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final difference = now.difference(date);

    if (difference.inDays == 0) {
      return 'Today';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else if (difference.inDays < 7) {
      return '${difference.inDays} days ago';
    } else {
      return '${date.day}/${date.month}/${date.year}';
    }
  }
}

