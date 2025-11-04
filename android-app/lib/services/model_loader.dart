// lib/services/model_loader.dart
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelLoader {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  Future<bool> loadModel() async {
    if (_isLoaded) return true;

    try {
      _interpreter = await Interpreter.fromAsset('models/crop_model.tflite');
      
      // Get input/output shapes for verification
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      
      print('Model loaded: Input shape: $inputShape, Output shape: $outputShape');
      _isLoaded = true;
      return true;
    } catch (e) {
      print('Error loading model: $e');
      print('Model loading failed - app will use constraint-based recommendations only');
      _isLoaded = false;
      return false;
    }
  }

  Future<double> predict(List<double> input) async {
    if (!_isLoaded || _interpreter == null) {
      throw Exception('Model not loaded');
    }

    try {
      // Prepare input
      final inputBuffer = Float32List.fromList(input);
      final inputTensor = inputBuffer.reshape([1, input.length]);
      
      // Prepare output
      final outputBuffer = Float32List(1);
      final outputTensor = outputBuffer.reshape([1, 1]);
      
      // Run inference
      _interpreter!.run(inputTensor, outputTensor);
      
      return outputBuffer[0];
    } catch (e) {
      print('Prediction error: $e');
      rethrow;
    }
  }

  void dispose() {
    _interpreter?.close();
    _isLoaded = false;
  }
}

