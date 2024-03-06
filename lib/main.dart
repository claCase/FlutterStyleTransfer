import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tflite;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'dart:developer';
import 'dart:io' as dio;
import 'package:flutter/services.dart' show rootBundle;

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
  final styleBasePath = "assets/styles/";
  String? stylePath = '';
  String? contentPath = '';
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
  final _picker = ImagePicker();
  late img.Image? imageContent;
  Uint8List? imageContentFile = null;
  Uint8List? imageContentStylized = null;
  late img.Image? imageStyle;
  String modelLoadedPredict = 'Predict NotLoaded';
  String modelLoadedStyle = 'Predict NotLoaded';
  String inputText = '0';
  String outputText = '0';
  Map contentSize = {"height": 384, "width": 384};
  Map styleSize = {"height": 256, "width": 256};

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
      final delegate = tflite.XNNPackDelegate();
      //final delegate = tflite.GpuDelegateV2();
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

  Future<void> _processContent() async {
    log("Processing content... ${contentPath!}");
    log("reading content image----");
    imageContentFile = dio.File(contentPath!).readAsBytesSync();
    log("decoding image...");
    final image = img.decodeImage(imageContentFile!);
    //final image = await img.decodeImageFile(contentPath!);
    setState(() {
      imageContent = img.copyResize(image!,
          width: contentSize["height"], height: contentSize["width"]);
    });
  }

  Future<void> _processStyle() async {
    log("loading style...");
    ByteData styleImage = await rootBundle.load(stylePath!);
    log("decoding style...");
    final image = img.decodeImage(styleImage.buffer.asUint8List());
    setState(() {
      log("resizing style...");
      imageStyle = img.copyResize(image!,
          height: styleSize["height"], width: styleSize["width"]);
    });
  }

  Future<void> _initializePredict() async {
    inputsStyle = List.generate(
        1,
        (i) => List.generate(256,
            (j) => List.generate(256, (k) => List.generate(3, (z) => 1.0))));
    outputsPredict = List.generate(
        1,
        (i) => List.generate(
            1, (j) => List.generate(1, (k) => List.generate(100, (z) => 1.0))));
  }

  Future<void> _initilizeStylized() async {
    inputsContent = List.generate(
        1,
        (i) => List.generate(384,
            (j) => List.generate(384, (k) => List.generate(3, (z) => 1.0))));
    outputsStylized = List.generate(
        1,
        (i) => List.generate(384,
            (j) => List.generate(384, (k) => List.generate(3, (z) => 1.0))));
  }

  Future<void> _runInference() async {
    log("processing style...");
    await _processStyle();
    log("processing inputs style...");
    inputsStyle = [
      List.generate(
          styleSize["width"],
          (i) => List.generate(styleSize["height"], (j) {
                final pixel = imageStyle!.getPixel(i, j);
                return [pixel.r / 255, pixel.g / 255, pixel.b / 255];
              }))
    ];
    log("processing inputs content...");
    inputsContent = [
      List.generate(
          contentSize["width"],
          (i) => List.generate(contentSize["height"], (j) {
                final pixel = imageContent!.getPixel(j,i);
                return [pixel.r / 255, pixel.g / 255, pixel.b / 255];
              }))
    ];
    log("running predict...");
    _interpreterPredict.run(inputsStyle, outputsPredict);
    log("running style transfer...");
    _interpreterStyle.runForMultipleInputs(
        [inputsContent, outputsPredict], {0: outputsStylized});
    log("decoding style output...");
    final buffer = Uint8List.fromList(outputsStylized.first
        .expand(
            (col) => col.expand((pixel) => pixel.map((e) => (e * 255).toInt())))
        .toList());
    final stylized = img.Image.fromBytes(
        width: contentSize["width"],
        height: contentSize["height"],
        bytes: buffer.buffer);
    log("setting stylized image...");
    imageContentStylized = img.encodeJpg(stylized);
    setState(() {});
  }

  @override
  void initState() {
    super.initState();
    _loadModelPredict();
    _initializePredict();
    _loadModelStyle();
    _initilizeStylized();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Align(
        alignment: Alignment.center,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(children: [
              const SizedBox(height: 200, width: 200),
              if (imageContentFile != null)
                Image.memory(
                  imageContentFile!,
                  height: 200,
                  width: 200,
                ),
              ElevatedButton(
                  onPressed: () async {
                    final result =
                        await _picker.pickImage(source: ImageSource.gallery);
                    contentPath = result?.path;
                    log(contentPath.toString());
                    setState(() {});
                    _processContent();
                  },
                  child: const Text("Pick Content Image")
                  ),
              //const SizedBox(height: 200, width: 200),
              if (imageContentStylized != null)
                  Image.memory(
                    imageContentStylized!,
                    height: 200,
                    width: 200,
                  ),
            ]),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: List.generate(10, (index) {
                  final currentImagePath =
                      styleBasePath + index.toString() + ".jpg";
                  return Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: InkWell(
                      onTap: () {
                        log(stylePath.toString());
                        stylePath = currentImagePath;
                        setState(() {});
                        if (stylePath != null && contentPath != null) {
                          _processStyle();
                          if (imageContent != null) _runInference();
                        }
                      },
                      child: Padding(
                        padding: const EdgeInsets.all(8),
                        child: Image.asset(
                          currentImagePath,
                          height: 96,
                        ),
                      ),
                    ),
                  );
                }),
              ),
            )
          ],
        ),
      ),
    );
  }
}
