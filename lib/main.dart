import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tflite;
import 'package:image/image.dart' as img;
import 'dart:developer';
import 'dart:io' as dio;

void main() {
  runApp(const MyApp(title: 'App'));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key, required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      home: MyHomePage(title: "My homepage"),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => DenseExample();
}

class DenseExample extends State<MyHomePage> {
  final String modelPath_predict = "assets/models/style_predict.tflite";
  final String modelPath_style = "assets/models/style_transform.tflite";
  late tflite.Interpreter _interpreterPredict;
  late tflite.Interpreter _interpreterStyle;
  late tflite.Tensor _inputTensorPredict;
  late tflite.Tensor _outputTensorPredict;
  late List<List<List<List<double>>>> inputsStyle;
  late List<List<List<List<double>>>> inputsContent;
  late List<tflite.Tensor> _inputTensorStyleMap;
  late tflite.Tensor _outputTensorStyle;
  late List<List<List<List<double>>>> outputsPredict;
  late List<List<List<List<double>>>> outputsStylized;
  late Uint8List image;
  String modelLoadedPredict = 'Predict NotLoaded';
  String modelLoadedStyle = 'Predict NotLoaded';
  String inputText = '0';
  String outputText = '0';

  Future<void> _loadModelPredict() async {
    final options = tflite.InterpreterOptions();
    options.threads = 5;
    //options.useNnApiForAndroid = true;

    if (dio.Platform.isAndroid) {
      final delegate = tflite.XNNPackDelegate();
      //final delegate = tflite.GpuDelegateV2();
      //final delegate_options = tflite.GpuDelegateOptionsV2();
      
      options.addDelegate(delegate);
    }
    // final delegate = tflite.GpuDelegate();
    // options.addDelegate(delegate);
    _interpreterPredict =
        await tflite.Interpreter.fromAsset(modelPath_predict, options: options);
    _inputTensorPredict = _interpreterPredict.getInputTensors().first;
    log('predict inputs ${_inputTensorPredict.toString()}');
    _outputTensorPredict = _interpreterPredict.getOutputTensors().first;
    log('predict output ${_outputTensorPredict.toString()}');
    setState(() {
      modelLoadedPredict = 'Predict Loaded';
    });
  }

  Future<void> _loadModelStyle() async {
    final options = tflite.InterpreterOptions();
    if (dio.Platform.isAndroid) {
      //final delegate = tflite.XNNPackDelegate();
      final delegate = tflite.GpuDelegateV2();
      //final delegate_options = tflite.GpuDelegateOptionsV2();
      options.addDelegate(delegate);
    }
    _interpreterStyle =
        await tflite.Interpreter.fromAsset(modelPath_style, options: options);
    _inputTensorStyleMap = _interpreterStyle.getInputTensors();
    log('input style 0 ${_inputTensorStyleMap[0].toString()}');
    log('input style 1 ${_inputTensorStyleMap[1].toString()}');
    _outputTensorStyle = _interpreterStyle.getOutputTensors().first;
    log('output style ${_outputTensorStyle.toString()}');
    setState(() {
      modelLoadedStyle = 'Style Loaded';
    });
  }

  
  Future<void> _inferencePredict() async {
    inputsStyle = List.generate(1, (i) => List.generate(256, (j) => List.generate(256, (k)=> List.generate(3, (z) => 1.0))));
    outputsPredict = List.generate(1, (i) => List.generate(1, (j) => List.generate(1, (k)=> List.generate(100, (z) => 1.0))));
    _interpreterPredict.run(inputsStyle, outputsPredict);
  }
  
  Future<void> _inferenceStylized() async {
    inputsContent = List.generate(1, (i) => List.generate(384, (j) => List.generate(384, (k)=> List.generate(3, (z) => 1.0))));
    outputsStylized = List.generate(1, (i) => List.generate(384, (j) => List.generate(384, (k)=> List.generate(3, (z) => 1.0))));
    _interpreterStyle.runForMultipleInputs([inputsContent, outputsPredict], {0:outputsStylized});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(modelLoadedPredict),
            ElevatedButton(
              onPressed: () {
                _loadModelPredict();
              },
              child: const Text("Load Model Predict"),
            ),
            Text(modelLoadedStyle),
            ElevatedButton(
              onPressed: () {
                _loadModelStyle();
              },
              child: const Text("Load Model Style"),
            ),
            ElevatedButton(
              onPressed: () {
                _inferencePredict();
              },
              child: const Text("Infer1"),
            ),
            ElevatedButton(
              onPressed: () {
                _inferenceStylized();
              },
              child: const Text("Infer2"),
            ),
          ],
        ),
      ),
    );
  }
}
