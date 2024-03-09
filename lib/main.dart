//import 'dart:ui';
//import 'package:flutter/painting.dart';
import 'package:flutter/services.dart';
import 'package:device_info_plus/device_info_plus.dart';
//import 'package:flutter/cupertino.dart';
//import 'package:flutter/rendering.dart';
//import 'package:flutter/widgets.dart';
//import 'package:app_settings/app_settings.dart' as app_settings;
import 'package:permission_handler/permission_handler.dart' as ph;
import 'dart:async';
//import 'dart:typed_data';
import 'package:path/path.dart' as pd;
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tflite;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'dart:developer';
import 'dart:io' as dio;
//import 'package:flutter/services.dart' show rootBundle;
//import 'package:path_provider/path_provider.dart' as pp;

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
  State<MyHomePage> createState() => StyleTransfer();
}

class StyleTransfer extends State<MyHomePage> {
  final String modelPathPredict = "assets/models/style_predict.tflite";
  final String modelPathStyle = "assets/models/style_transform.tflite";
  final styleBasePath = "assets/styles/";
  String? stylePath = '';
  String? contentPath = '';
  late tflite.Interpreter _interpreterPredict;
  late tflite.Interpreter _interpreterStyle;
  //late final tflite.IsolateInterpreter predictionIsolateInterpreter;
  //late final tflite.IsolateInterpreter styleIsolateInterpreter;
  late tflite.Tensor _inputTensorPredict;
  late tflite.Tensor _outputTensorPredict;
  late List<List<List<List<double>>>> inputsStyle;
  late List<List<List<List<double>>>> inputsContent;
  late List<tflite.Tensor> _inputTensorStyleMap;
  late tflite.Tensor _outputTensorStyle;
  late List<List<List<List<double>>>> outputsPredict;
  late List<List<List<List<double>>>> outputsStylized;
  final _picker = ImagePicker();
  img.Image? imageContent;
  Uint8List? imageContentFile;
  Uint8List? imageContentStylized;
  //Uint8List? _currentStyleImage;
  late img.Image? imageStyle;
  late List<Uint8List> stylesListView;
  String modelLoadedPredict = 'Predict NotLoaded';
  String modelLoadedStyle = 'Predict NotLoaded';
  String inputText = '0';
  String outputText = '0';
  Map contentSize = {"height": 384, "width": 384};
  Map styleSize = {"height": 256, "width": 256};
  Map originalSize = {"height": 384, "width": 384};
  bool displayContent = false;
  bool _isLoading = false;

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
        await tflite.Interpreter.fromAsset(modelPathPredict, options: options);
    //predictionIsolateInterpreter = await tflite.IsolateInterpreter.create(address: _interpreterPredict.address);
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
        await tflite.Interpreter.fromAsset(modelPathStyle, options: options);
    //styleIsolateInterpreter = await tflite.IsolateInterpreter.create(address: _interpreterStyle.address);

    _inputTensorStyleMap = _interpreterStyle.getInputTensors();
    log('input style 0 ${_inputTensorStyleMap[0].toString()}');
    log('input style 1 ${_inputTensorStyleMap[1].toString()}');
    _outputTensorStyle = _interpreterStyle.getOutputTensors().first;
    log('output style ${_outputTensorStyle.toString()}');
    setState(() {
      modelLoadedStyle = 'Style Loaded';
    });
  }

  Future<Image> _processCurrentStyleImage(String currentPath) async {
    ByteData currentStyleImage = await rootBundle.load(currentPath);
    log("decoding style...");
    final cmd = img.Command()
      ..decodeImage(currentStyleImage.buffer.asUint8List())
      ..copyCrop(
          x: 100,
          y: 100,
          height: styleSize["height"],
          width: styleSize["width"])
      ..executeThread();
    log("getting style...");
    final currentStyle = await cmd.getImage();

    return Image.memory(currentStyle!.buffer.asUint8List());
  }

  Future<void> _processContent() async {
    log("Processing content... ${contentPath!}");
    imageContentFile = dio.File(contentPath!).readAsBytesSync();
    final cmd = img.Command()
      ..decodeImageFile(contentPath!)
      ..executeThread();
    log("awaiting content image...");
    imageContent = await cmd.getImage();
    originalSize["height"] = imageContent!.height.toInt();
    originalSize["width"] = imageContent!.width.toInt();
    imageContent = img.copyResize(imageContent!,
        width: contentSize["height"], height: contentSize["width"]);
    setState(() {
      displayContent = true;
    });
  }

  Future<void> _processStyle() async {
    ByteData styleImage = await rootBundle.load(stylePath!);
    log("decoding style...");
    final cmd = img.Command()
      ..decodeImage(styleImage.buffer.asUint8List())
      ..copyResize(height: styleSize["height"], width: styleSize["width"])
      ..executeThread();
    log("getting style...");
    imageStyle = await cmd.getImage();
    log("style acquired!");
    setState(() {});
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
    log('is loading $_isLoading');
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
                final pixel = imageContent!.getPixel(j, i);
                return [pixel.r / 255, pixel.g / 255, pixel.b / 255];
              }))
    ];
    log("running predict...");
    //predictionIsolateInterpreter.run(inputsStyle, outputsPredict);
    _interpreterPredict.run(inputsStyle, outputsPredict);
    log("running style transfer...");
    //styleIsolateInterpreter.runForMultipleInputs(
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
    final stylizedReshaped = img.copyResize(stylized,
        height: originalSize["height"], width: originalSize["width"]);
    log("setting stylized image...");
    setState(() {
      imageContentStylized = img.encodeJpg(stylizedReshaped);
      displayContent = false;
      _isLoading = false;
      log('is loading $_isLoading');
    });
  }

  Future<void> _saveContent() async {
    final plugin = DeviceInfoPlugin();
    final android = await plugin.androidInfo;
    const permissionStorage = ph.Permission.storage;
    const permissionPhotos = ph.Permission.photos;
    //const permissionVideos = ph.Permission.videos;
    int androidVersion = android.version.sdkInt;

    if (androidVersion < 33) {
      final permissionDenied = await permissionStorage.isDenied;
      if (permissionDenied) {
        log('requesting storage permission...');
        await permissionStorage.request();
      }
      log('Permission storage: ${await permissionStorage.isGranted}');
    } else {
      final permissionDenied = await permissionPhotos.isDenied;
      if (permissionDenied | await permissionPhotos.isPermanentlyDenied) {
        log("requesting photo permission...");
        await permissionPhotos.request();
        //final videosPermission = await permissionVideos.request();
      }
      log("Permission photos: ${await permissionPhotos.isGranted}");
      //log("Permission videos: ${videosPermission.isGranted}");
    }

    String now = DateTime.now()
        .toString()
        .replaceAll(" ", "_")
        .replaceAll(":", "-")
        .replaceAll(".", "");
    const basePath = '/storage/emulated/0/Download';
    String baseName = pd.basenameWithoutExtension(contentPath!);
    String baseExtension = pd.extension(contentPath!);
    //final basePath = await pp.getDownloadsDirectory();
    if ((androidVersion < 33 && await permissionStorage.isGranted) |
        (androidVersion >= 33 && await permissionPhotos.isGranted)) {
      log("Writing to: $basePath/${baseName}_stylized_$now$baseExtension");
      await dio.File("$basePath/${baseName}_stylized$now$baseExtension")
          .writeAsBytes(imageContentStylized!.buffer.asInt8List());
      log("file saved");
    }
  }

  Image? _returnImage() {
    if (imageContentStylized != null && !displayContent) {
      return Image.memory(
        imageContentStylized!,
        fit: BoxFit.fill,
      );
    } else {
      if (imageContentFile != null && displayContent) {
        return Image.memory(
          imageContentFile!,
          fit: BoxFit.fill,
        );
      } else {
        return null;
      }
    }
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
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.amber,
        title: Title(
          title: "Style Transfer",
          color: Colors.black,
          child: const Center(child: Text("Style Transfer")),
        ),
      ),
      body: Align(
        alignment: Alignment.center,
        child: Column(
          //mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(child: LayoutBuilder(
                builder: (BuildContext context, BoxConstraints constraints) {
              return Container(
                  alignment: Alignment.center,
                  height: constraints.maxHeight,
                  width: constraints.maxWidth,
                  color: Colors.black,
                  child: _returnImage());
            })),
            /*Stack(fit: StackFit.values.first, children: [
                  if (_isLoading)
                    const Opacity(
                        opacity: 0.8,
                        child: ModalBarrier(
                          dismissible: false,
                          color: Colors.black,
                        )),
                  if (_isLoading)
                    const Center(child: CircularProgressIndicator()),
                  AspectRatio(
                      aspectRatio:
                          originalSize["width"] / originalSize["height"],
                      child: _returnImage())),
                ])),*/
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Align(
                  alignment: Alignment.centerLeft,
                  child: ElevatedButton(
                      onPressed: () async {
                        final result = await _picker.pickImage(
                            source: ImageSource.gallery);
                        contentPath = result?.path;
                        log(contentPath.toString());
                        setState(() {});
                        _processContent();
                      },
                      child: const Text("Pick Content Image"))),
              if (imageContentStylized != null)
                Align(
                    alignment: Alignment.centerRight,
                    child: ElevatedButton(
                        onPressed: _saveContent,
                        child: const Text("Save Stylized Image")))
            ]),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: List.generate(17, (index) {
                  final currentImagePath =
                      '$styleBasePath${index.toString()}.jpg';
                  return Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: InkWell(
                      onTap: () {
                        setState(() {
                          stylePath = currentImagePath;
                        });
                        if (stylePath != null && contentPath != null) {
                          setState(() {
                            _isLoading = true;
                          });
                          if (imageContent != null) _runInference();
                        }
                      },
                      child: Padding(
                        padding: const EdgeInsets.all(8),
                        child: Image.asset(
                          currentImagePath,
                          height: 100,
                          width: 100,
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
