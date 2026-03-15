import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:webview_flutter/webview_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RealityCheckApp());
}

class RealityCheckApp extends StatelessWidget {
  const RealityCheckApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Reality Check',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1D4ED8)),
        useMaterial3: true,
      ),
      home: const RealityCheckHome(),
    );
  }
}

class RealityCheckHome extends StatefulWidget {
  const RealityCheckHome({super.key});

  @override
  State<RealityCheckHome> createState() => _RealityCheckHomeState();
}

class _RealityCheckHomeState extends State<RealityCheckHome> {
  static const String analyzerUrl =
      'https://cyleunggg.github.io/bereal/analyzer/index.html';

  final ImagePicker _picker = ImagePicker();
  late final WebViewController _controller;

  bool _pageReady = false;
  bool _analyzing = false;
  Uint8List? _imageBytes;
  Map<String, dynamic>? _result;
  String? _error;

  @override
  void initState() {
    super.initState();

    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..addJavaScriptChannel(
        'AnalyzerChannel',
        onMessageReceived: (JavaScriptMessage message) {
          _handleAnalyzerMessage(message.message);
        },
      )
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageFinished: (_) {
            if (mounted) {
              setState(() {
                _pageReady = true;
              });
            }
          },
          onWebResourceError: (error) {
            if (mounted) {
              setState(() {
                _error = 'Failed to load analyzer page.';
              });
            }
          },
        ),
      )
      ..loadRequest(Uri.parse(analyzerUrl));
  }

  Future<void> _pickImage() async {
    setState(() {
      _error = null;
      _result = null;
      _imageBytes = null;
    });

    final XFile? image = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1600,
      maxHeight: 1600,
      imageQuality: 92,
    );

    if (image == null) {
      return;
    }

    final bytes = await image.readAsBytes();
    if (!mounted) return;

    setState(() {
      _imageBytes = bytes;
    });

    await _analyzeImage(bytes);
  }

  Future<void> _analyzeImage(Uint8List bytes) async {
    if (!_pageReady) {
      setState(() {
        _error = 'Analyzer page is not ready yet. Please wait.';
      });
      return;
    }

    setState(() {
      _analyzing = true;
      _error = null;
      _result = null;
    });

    final payload = jsonEncode(<String, dynamic>{
      'imageBase64': base64Encode(bytes),
      'meta': {
        'filename': 'selected-image',
        'timestamp': DateTime.now().toIso8601String(),
      },
    });

    final script = 'window.analyzeImage($payload);';
    try {
      await _controller.runJavaScript(script);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _analyzing = false;
        _error = 'Failed to start analysis.';
      });
    }
  }

  void _handleAnalyzerMessage(String message) {
    try {
      final decoded = jsonDecode(message);
      if (!mounted) return;
      setState(() {
        _analyzing = false;
        _result = decoded is Map<String, dynamic>
            ? decoded
            : <String, dynamic>{'raw': decoded};
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _analyzing = false;
        _error = 'Invalid response from analyzer.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Reality Check'),
        backgroundColor: colorScheme.primaryContainer,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildStatusCard(),
                const SizedBox(height: 16),
                _buildImagePreview(),
                const SizedBox(height: 16),
                _buildResultPanel(),
              ],
            ),
          ),
          _buildBottomBar(),
          SizedBox(
            height: 1,
            child: Opacity(
              opacity: 0.01,
              child: WebViewWidget(controller: _controller),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard() {
    final statusText = _analyzing
        ? 'Analyzing image...'
        : _pageReady
            ? 'Analyzer ready'
            : 'Loading analyzer...';

    return Card(
      child: ListTile(
        leading: Icon(
          _analyzing
              ? Icons.hourglass_top
              : _pageReady
                  ? Icons.check_circle
                  : Icons.cloud_download,
        ),
        title: Text(statusText),
        subtitle: Text(
          analyzerUrl,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_imageBytes == null) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text('Select an image to start analysis.'),
        ),
      );
    }

    return Card(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.memory(
          _imageBytes!,
          height: 220,
          fit: BoxFit.cover,
        ),
      ),
    );
  }

  Widget _buildResultPanel() {
    if (_error != null) {
      return Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(_error!),
        ),
      );
    }

    if (_analyzing) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: LinearProgressIndicator(),
        ),
      );
    }

    if (_result == null) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('No results yet.'),
        ),
      );
    }

    final summary = _result!['summary'] ?? _result;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(
          const JsonEncoder.withIndent('  ').convert(summary),
        ),
      ),
    );
  }

  Widget _buildBottomBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _analyzing ? null : _pickImage,
            icon: const Icon(Icons.photo_library),
            label: const Text('Pick Image for Analysis'),
          ),
        ),
      ),
    );
  }
}
