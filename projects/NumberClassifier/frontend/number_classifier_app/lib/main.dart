// main.dart
//
// Digit Classifier — dark, glassmorphic, neon-accented front-end for your
// FastAPI /predict endpoint.
//
// Add to pubspec.yaml:
//   http: ^1.2.0
//   http_parser: ^1.0.2
//   image_picker: ^1.1.2
//
// Set kApiBaseUrl below to match how your device reaches the server:
//   Android emulator  -> http://10.0.2.2:8000
//   iOS simulator     -> http://127.0.0.1:8000
//   Physical device   -> http://<your-computer-LAN-IP>:8000

import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';

const String kApiBaseUrl = 'http://127.0.0.1:8000'; // <-- change me

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

class AppColors {
  static const bgTop = Color(0xFF0A0A12);
  static const bgBottom = Color(0xFF06060A);
  static const glow1 = Color(0xFF8B5CF6); // violet
  static const glow2 = Color(0xFF22D3EE); // cyan
  static const glow3 = Color(0xFFEC4899); // pink
  static const glass = Color(0x14FFFFFF);
  static const glassBorder = Color(0x33FFFFFF);
  static const textDim = Color(0xFF9CA3AF);
}

void main() {
  runApp(const DigitClassifierApp());
}

class DigitClassifierApp extends StatelessWidget {
  const DigitClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Digit Classifier',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark, // dark mode only, always
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AppColors.bgBottom,
        colorScheme: const ColorScheme.dark(
          primary: AppColors.glow1,
          secondary: AppColors.glow2,
        ),
        fontFamily: 'Roboto',
      ),
      home: const HomePage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Reusable premium building blocks
// ---------------------------------------------------------------------------

/// Frosted glass card with a soft neon-tinted border.
class GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  const GlassCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.radius = 24,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 18, sigmaY: 18),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: AppColors.glass,
            borderRadius: BorderRadius.circular(radius),
            border: Border.all(color: AppColors.glassBorder, width: 1),
          ),
          child: child,
        ),
      ),
    );
  }
}

/// Big soft blurred color blob used to create ambient neon glow in the bg.
class GlowBlob extends StatelessWidget {
  final Color color;
  final double size;
  const GlowBlob({super.key, required this.color, this.size = 260});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: RadialGradient(
            colors: [color.withOpacity(0.55), color.withOpacity(0.0)],
          ),
        ),
      ),
    );
  }
}

/// Gradient-filled button with a neon glow shadow.
class NeonButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;
  final List<Color> colors;
  const NeonButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
    this.colors = const [AppColors.glow1, AppColors.glow2],
  });

  @override
  Widget build(BuildContext context) {
    final disabled = onPressed == null;
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 200),
      opacity: disabled ? 0.45 : 1.0,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(colors: colors),
          boxShadow: disabled
              ? []
              : [
                  BoxShadow(
                    color: colors.first.withOpacity(0.55),
                    blurRadius: 20,
                    spreadRadius: 0.5,
                    offset: const Offset(0, 6),
                  ),
                ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(16),
            onTap: onPressed,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, color: Colors.white, size: 20),
                  const SizedBox(width: 8),
                  Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Ghost / outlined glass button.
class GhostButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;
  const GhostButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final disabled = onPressed == null;
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 200),
      opacity: disabled ? 0.4 : 1.0,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          color: AppColors.glass,
          border: Border.all(color: AppColors.glassBorder, width: 1),
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(16),
            onTap: onPressed,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 13),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, color: Colors.white70, size: 19),
                  const SizedBox(width: 8),
                  Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Segmented glass pill switch used instead of a default TabBar.
class GlassSegmentedTabs extends StatelessWidget {
  final int index;
  final ValueChanged<int> onChanged;
  const GlassSegmentedTabs({
    super.key,
    required this.index,
    required this.onChanged,
  });

  static const _tabs = [
    (icon: Icons.brush_rounded, label: 'Draw'),
    (icon: Icons.image_rounded, label: 'Upload'),
  ];

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      radius: 20,
      padding: const EdgeInsets.all(6),
      child: Row(
        children: List.generate(_tabs.length, (i) {
          final selected = i == index;
          final tab = _tabs[i];
          return Expanded(
            child: GestureDetector(
              onTap: () => onChanged(i),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 250),
                curve: Curves.easeOut,
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(14),
                  gradient: selected
                      ? const LinearGradient(
                          colors: [AppColors.glow1, AppColors.glow2],
                        )
                      : null,
                  boxShadow: selected
                      ? [
                          BoxShadow(
                            color: AppColors.glow1.withOpacity(0.45),
                            blurRadius: 16,
                            offset: const Offset(0, 4),
                          ),
                        ]
                      : [],
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      tab.icon,
                      size: 18,
                      color: selected ? Colors.white : AppColors.textDim,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      tab.label,
                      style: TextStyle(
                        color: selected ? Colors.white : AppColors.textDim,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Shared API call
// ---------------------------------------------------------------------------

class PredictionResult {
  final String? prediction;
  final String? error;
  PredictionResult({this.prediction, this.error});
}

Future<PredictionResult> sendImageForPrediction(
  Uint8List pngBytes, {
  String filename = 'digit.png',
}) async {
  final uri = Uri.parse('$kApiBaseUrl/predict');

  try {
    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes(
          'image', // must match `image: UploadFile` param name in FastAPI
          pngBytes,
          filename: filename,
          contentType: MediaType('image', 'png'),
        ),
      );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      return PredictionResult(
        error: 'Server error (${response.statusCode}): ${response.body}',
      );
    }

    final Map<String, dynamic> body = jsonDecode(response.body);

    if (body.containsKey('message')) {
      return PredictionResult(error: body['message'].toString());
    }

    if (body.isNotEmpty) {
      final value = body.values.first;
      return PredictionResult(prediction: value.toString());
    }

    return PredictionResult(error: 'Unexpected empty response from server.');
  } catch (e) {
    return PredictionResult(
      error: 'Could not reach the server. Is it running at $kApiBaseUrl?\n$e',
    );
  }
}

// ---------------------------------------------------------------------------
// Home page
// ---------------------------------------------------------------------------

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _tabIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Ambient gradient background
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [AppColors.bgTop, AppColors.bgBottom],
              ),
            ),
          ),
          // Floating neon glow blobs
          Positioned(
            top: -60,
            left: -60,
            child: GlowBlob(color: AppColors.glow1, size: 260),
          ),
          Positioned(
            top: 120,
            right: -80,
            child: GlowBlob(color: AppColors.glow2, size: 220),
          ),
          Positioned(
            bottom: -80,
            left: 40,
            child: GlowBlob(color: AppColors.glow3, size: 240),
          ),

          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              child: Column(
                children: [
                  const _HeaderTitle(),
                  const SizedBox(height: 18),
                  GlassSegmentedTabs(
                    index: _tabIndex,
                    onChanged: (i) => setState(() => _tabIndex = i),
                  ),
                  const SizedBox(height: 18),
                  Expanded(
                    child: IndexedStack(
                      index: _tabIndex,
                      children: const [DrawTab(), UploadTab()],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeaderTitle extends StatelessWidget {
  const _HeaderTitle();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            gradient: const LinearGradient(
              colors: [AppColors.glow1, AppColors.glow2],
            ),
            boxShadow: [
              BoxShadow(
                color: AppColors.glow1.withOpacity(0.5),
                blurRadius: 18,
              ),
            ],
          ),
          child: const Icon(Icons.filter_9_plus_rounded, color: Colors.white),
        ),
        const SizedBox(width: 12),
        ShaderMask(
          shaderCallback: (bounds) =>
              const LinearGradient(colors: [Colors.white, AppColors.glow2])
                  .createShader(bounds),
          child: const Text(
            'Digit Classifier',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              color: Colors.white,
              letterSpacing: 0.2,
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Result display
// ---------------------------------------------------------------------------

class ResultBanner extends StatelessWidget {
  final bool loading;
  final PredictionResult? result;
  const ResultBanner({super.key, required this.loading, this.result});

  @override
  Widget build(BuildContext context) {
    if (loading) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 18),
        child: SizedBox(
          height: 28,
          width: 28,
          child: CircularProgressIndicator(
            strokeWidth: 3,
            valueColor: AlwaysStoppedAnimation(AppColors.glow2),
          ),
        ),
      );
    }
    if (result == null) return const SizedBox(height: 4);

    if (result!.error != null) {
      return Padding(
        padding: const EdgeInsets.only(top: 14),
        child: GlassCard(
          radius: 16,
          child: Row(
            children: [
              const Icon(Icons.error_outline_rounded, color: AppColors.glow3),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  result!.error!,
                  style: const TextStyle(color: Colors.white70, height: 1.3),
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.only(top: 14),
      child: GlassCard(
        radius: 20,
        child: Column(
          children: [
            const Text(
              'PREDICTION',
              style: TextStyle(
                color: AppColors.textDim,
                fontSize: 12,
                letterSpacing: 3,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 6),
            ShaderMask(
              shaderCallback: (bounds) => const LinearGradient(
                colors: [AppColors.glow2, AppColors.glow1],
              ).createShader(bounds),
              child: Text(
                result!.prediction ?? '-',
                style: const TextStyle(
                  fontSize: 64,
                  fontWeight: FontWeight.w900,
                  color: Colors.white,
                  height: 1.1,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// DRAW TAB
// ---------------------------------------------------------------------------

class DrawTab extends StatefulWidget {
  const DrawTab({super.key});

  @override
  State<DrawTab> createState() => _DrawTabState();
}

class _DrawTabState extends State<DrawTab> {
  final GlobalKey _canvasKey = GlobalKey();
  final List<List<Offset>> _strokes = [];
  bool _loading = false;
  PredictionResult? _result;

  void _onPanStart(DragStartDetails details) {
    setState(() => _strokes.add([details.localPosition]));
  }

  void _onPanUpdate(DragUpdateDetails details) {
    setState(() => _strokes.last.add(details.localPosition));
  }

  void _clear() {
    setState(() {
      _strokes.clear();
      _result = null;
    });
  }

  Future<void> _predict() async {
    if (_strokes.isEmpty) {
      setState(() => _result = PredictionResult(error: 'Draw a digit first.'));
      return;
    }
    setState(() {
      _loading = true;
      _result = null;
    });
    try {
      final boundary =
          _canvasKey.currentContext!.findRenderObject()
              as RenderRepaintBoundary;
      final ui.Image image = await boundary.toImage(pixelRatio: 3.0);
      final ByteData? byteData = await image.toByteData(
        format: ui.ImageByteFormat.png,
      );
      final Uint8List pngBytes = byteData!.buffer.asUint8List();
      final result = await sendImageForPrediction(pngBytes);
      setState(() {
        _result = result;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _result = PredictionResult(error: 'Failed to capture drawing: $e');
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      physics: const NeverScrollableScrollPhysics(),
      child: Column(
        children: [
          AspectRatio(
            aspectRatio: 1,
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                boxShadow: [
                  BoxShadow(
                    color: AppColors.glow2.withOpacity(0.28),
                    blurRadius: 30,
                    spreadRadius: 1,
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: RepaintBoundary(
                  key: _canvasKey,
                  child: GestureDetector(
                    onPanStart: _onPanStart,
                    onPanUpdate: _onPanUpdate,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.black,
                        border: Border.all(
                          color: AppColors.glassBorder,
                          width: 1.5,
                        ),
                      ),
                      child: CustomPaint(
                        painter: _NeonStrokePainter(_strokes),
                        size: Size.infinite,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 18),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GhostButton(
                onPressed: _loading ? null : _clear,
                icon: Icons.clear_rounded,
                label: 'Clear',
              ),
              const SizedBox(width: 14),
              NeonButton(
                onPressed: _loading ? null : _predict,
                icon: Icons.auto_awesome_rounded,
                label: 'Predict',
              ),
            ],
          ),
          ResultBanner(loading: _loading, result: _result),
        ],
      ),
    );
  }
}

/// Paints strokes with a soft neon glow (blurred underlay + crisp core).
class _NeonStrokePainter extends CustomPainter {
  final List<List<Offset>> strokes;
  _NeonStrokePainter(this.strokes);

  @override
  void paint(Canvas canvas, Size size) {
    final glowPaint = Paint()
      ..color = AppColors.glow2.withOpacity(0.55)
      ..strokeWidth = 30
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);

    final corePaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 16
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;

    for (final stroke in strokes) {
      for (int i = 0; i < stroke.length - 1; i++) {
        canvas.drawLine(stroke[i], stroke[i + 1], glowPaint);
      }
    }
    for (final stroke in strokes) {
      for (int i = 0; i < stroke.length - 1; i++) {
        canvas.drawLine(stroke[i], stroke[i + 1], corePaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _NeonStrokePainter oldDelegate) => true;
}

// ---------------------------------------------------------------------------
// UPLOAD TAB
// ---------------------------------------------------------------------------

class UploadTab extends StatefulWidget {
  const UploadTab({super.key});

  @override
  State<UploadTab> createState() => _UploadTabState();
}

class _UploadTabState extends State<UploadTab> {
  final ImagePicker _picker = ImagePicker();
  Uint8List? _pickedBytes;
  String? _pickedName;
  bool _loading = false;
  PredictionResult? _result;

  Future<void> _pickImage(ImageSource source) async {
    final XFile? file = await _picker.pickImage(source: source);
    if (file == null) return;
    final bytes = await file.readAsBytes();

    if (bytes.lengthInBytes > 5 * 1024 * 1024) {
      setState(() {
        _result = PredictionResult(error: 'Image size exceeds limit (max 5MB)');
        _pickedBytes = null;
      });
      return;
    }

    setState(() {
      _pickedBytes = bytes;
      _pickedName = file.name;
      _result = null;
    });
  }

  Future<void> _predict() async {
    if (_pickedBytes == null) {
      setState(() => _result = PredictionResult(error: 'Choose a PNG first.'));
      return;
    }
    setState(() {
      _loading = true;
      _result = null;
    });
    final result = await sendImageForPrediction(
      _pickedBytes!,
      filename: _pickedName ?? 'upload.png',
    );
    setState(() {
      _result = result;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      physics: const NeverScrollableScrollPhysics(),
      child: Column(
        children: [
          AspectRatio(
            aspectRatio: 1,
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                boxShadow: _pickedBytes == null
                    ? []
                    : [
                        BoxShadow(
                          color: AppColors.glow1.withOpacity(0.28),
                          blurRadius: 30,
                          spreadRadius: 1,
                        ),
                      ],
              ),
              child: GlassCard(
                radius: 24,
                padding: EdgeInsets.zero,
                child: _pickedBytes == null
                    ? const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.cloud_upload_rounded,
                              size: 42,
                              color: AppColors.textDim,
                            ),
                            SizedBox(height: 10),
                            Text(
                              'No image selected',
                              style: TextStyle(color: AppColors.textDim),
                            ),
                          ],
                        ),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(24),
                        child: Image.memory(_pickedBytes!, fit: BoxFit.contain),
                      ),
              ),
            ),
          ),
          const SizedBox(height: 18),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GhostButton(
                onPressed: _loading
                    ? null
                    : () => _pickImage(ImageSource.gallery),
                icon: Icons.photo_library_rounded,
                label: 'Gallery',
              ),
              const SizedBox(width: 12),
              GhostButton(
                onPressed: _loading
                    ? null
                    : () => _pickImage(ImageSource.camera),
                icon: Icons.camera_alt_rounded,
                label: 'Camera',
              ),
            ],
          ),
          const SizedBox(height: 14),
          NeonButton(
            onPressed: _loading ? null : _predict,
            icon: Icons.auto_awesome_rounded,
            label: 'Predict',
            colors: const [AppColors.glow3, AppColors.glow1],
          ),
          ResultBanner(loading: _loading, result: _result),
        ],
      ),
    );
  }
}
