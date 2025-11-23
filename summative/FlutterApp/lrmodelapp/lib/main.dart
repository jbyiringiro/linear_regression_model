import 'dart:convert';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Battery Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        colorScheme: ColorScheme.fromSwatch(
          primarySwatch: Colors.blue,
        ).copyWith(secondary: Colors.blueAccent),
        scaffoldBackgroundColor: const Color(
          0xFFF3F8FF,
        ), // soft blue background
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          filled: true,
          fillColor: Colors.white,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue, // main button color
            foregroundColor: Colors.white, // text color
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
            padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 28),
          ),
        ),
      ),
      home: const PredictPage(),
    );
  }
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});
  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  final _formKey = GlobalKey<FormState>();

  final ageCtl = TextEditingController();
  final heightCtl = TextEditingController(); // user enters cm
  final weightCtl = TextEditingController(); // kg
  final bmiCtl = TextEditingController(); // auto-calculated but still shown
  final distCtl = TextEditingController();
  double terrainValue = 1.0;

  String result = 'Enter values and press Predict';
  bool loading = false;

  // ---- SET THE API URL HERE BEFORE RUNNING ----
  // For Android emulator use 10.0.2.2
  // For iOS simulator use http://127.0.0.1:8000/predict
  // For a physical device use http://<YOUR_PC_IP>:8000/predict (and allow Windows firewall)
  // If you later deploy to Render use https://your-app.onrender.com/predict
  String get apiUrl {
    // If you're running web, you can't use 10.0.2.2; use a host reachable by the browser or a deployed URL
    if (kIsWeb) {
      return "http://127.0.0.1:8000/predict"; // usually works for web localhost testing only
    }
    // On mobile emulator:
    if (Platform.isAndroid) {
      // Android emulator -> 10.0.2.2 maps to host machine
      return "http://10.0.2.2:8000/predict";
    } else if (Platform.isIOS) {
      return "http://127.0.0.1:8000/predict";
    } else {
      // fallback for desktop
      return "http://127.0.0.1:8000/predict";
    }
  }
  // ---------------------------------------------

  final List<Map<String, dynamic>> terrainOptions = [
    {'v': 1.0, 'label': '1.0 — Smooth (indoor: tiles, halls)'},
    {'v': 1.3, 'label': '1.3 — Pavement/Sidewalk (normal outdoor)'},
    {'v': 1.6, 'label': '1.6 — Rough (grass, gravel, dirt)'},
  ];

  @override
  void initState() {
    super.initState();
    // recalc BMI whenever height or weight changes
    heightCtl.addListener(_recalculateBmi);
    weightCtl.addListener(_recalculateBmi);
  }

  @override
  void dispose() {
    ageCtl.dispose();
    heightCtl.dispose();
    weightCtl.dispose();
    bmiCtl.dispose();
    distCtl.dispose();
    super.dispose();
  }

  void _recalculateBmi() {
    final hText = heightCtl.text;
    final wText = weightCtl.text;
    final h = double.tryParse(hText);
    final w = double.tryParse(wText);
    if (h != null && h > 0 && w != null && w > 0) {
      // height in cm -> convert to meters
      final hm = h / 100.0;
      final bmi = w / (hm * hm);
      bmiCtl.text = bmi.toStringAsFixed(1);
    } else {
      bmiCtl.text = '';
    }
  }

  String? _validateInt(String? v, int min, int max) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final n = int.tryParse(v);
    if (n == null) return 'Enter integer';
    if (n < min || n > max) return 'Between $min and $max';
    return null;
  }

  String? _validateDouble(String? v, double min, double max) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final d = double.tryParse(v);
    if (d == null) return 'Enter number';
    if (d < min || d > max) return 'Between $min and $max';
    return null;
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      loading = true;
      result = '';
    });

    final body = {
      "Age": int.parse(ageCtl.text),
      "Height":
          double.parse(heightCtl.text) / 100.0, // convert to meters for model
      "Weight": double.parse(weightCtl.text),
      "Bmi": double.parse(bmiCtl.text),
      "daily_distance_km": double.parse(distCtl.text),
      "terrain_factor": terrainValue,
    };

    try {
      final uri = Uri.parse(
        apiUrl,
      ); // may throw FormatException if invalid string
      final resp = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 10));

      if (resp.statusCode == 200) {
        final jsonResp = jsonDecode(resp.body) as Map<String, dynamic>;
        final val = jsonResp['predicted_battery_Wh'];
        setState(() {
          if (val is num) {
            result = "Predicted battery: ${val.toStringAsFixed(2)} Wh";
          } else {
            result = "Prediction: $val";
          }
        });
      } else if (resp.statusCode == 422) {
        setState(() => result = 'Validation error: ${resp.body}');
      } else {
        setState(
          () => result = 'Server error ${resp.statusCode}: ${resp.body}',
        );
      }
    } on FormatException {
      setState(() => result = 'Invalid API URL. Check apiUrl value in code.');
    } on Exception catch (e) {
      setState(
        () => result = 'Request failed: $e\nCheck API running & network.',
      );
    } finally {
      setState(() => loading = false);
    }
  }

  Widget _numField({
    required String label,
    required TextEditingController ctl,
    required String? Function(String?) validator,
    required String hint,
    TextInputType keyboard = TextInputType.number,
    bool readOnly = false,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        controller: ctl,
        readOnly: readOnly,
        keyboardType: keyboard,
        validator: validator,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          filled: true,
          fillColor: Colors.white,
        ),
        style: const TextStyle(fontSize: 18),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final double buttonWidth = 180;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Battery Predictor'),
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.blue,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 12),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // App description card
              Card(
                elevation: 2,
                color: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        'About this app',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 6),
                      Text(
                        'Predicts estimated battery capacity (Wh) needed for an assistive mobility device for a child. '
                        'Enter Age, Height (cm), Weight (kg), Daily distance, and select Terrain using the dropdown. BMI is calculated automatically.',
                        style: TextStyle(fontSize: 14),
                      ),
                      SizedBox(height: 6),
                      Text(
                        'How to use: fill the fields → press Predict → see estimated battery in the box below.',
                        style: TextStyle(fontSize: 13),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
              _numField(
                label: 'Age (years)',
                ctl: ageCtl,
                hint: 'e.g. 8',
                validator: (v) => _validateInt(v, 1, 120),
              ),
              _numField(
                label: 'Height (cm)',
                ctl: heightCtl,
                hint: 'e.g. 120',
                validator: (v) => _validateDouble(v, 50.0, 250.0),
              ),
              _numField(
                label: 'Weight (kg)',
                ctl: weightCtl,
                hint: 'e.g. 30',
                validator: (v) => _validateDouble(v, 2.0, 300.0),
              ),
              _numField(
                label: 'BMI (auto)',
                ctl: bmiCtl,
                hint: '',
                validator: (v) => _validateDouble(v, 2.0, 100.0),
                readOnly: true,
              ),
              _numField(
                label: 'Daily distance (km)',
                ctl: distCtl,
                hint: 'e.g. 3.0',
                validator: (v) => _validateDouble(v, 0.0, 200.0),
              ),
              const SizedBox(height: 6),
              const Text(
                'Terrain factor',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 6),
              DropdownButtonFormField<double>(
                value: terrainValue,
                decoration: const InputDecoration(),
                items: terrainOptions.map((opt) {
                  return DropdownMenuItem<double>(
                    value: opt['v'],
                    child: Text(
                      opt['label'],
                      style: const TextStyle(fontSize: 15),
                    ),
                  );
                }).toList(),
                onChanged: (v) {
                  if (v != null) setState(() => terrainValue = v);
                },
              ),
              Padding(
                padding: const EdgeInsets.only(top: 8.0, bottom: 12.0),
                child: Text(
                  'Choose the terrain that best matches where the child usually travels. Smooth = indoor tiles/halls; Pavement = sidewalks; Rough = grass/gravel.',
                  style: TextStyle(fontSize: 13, color: Colors.grey.shade700),
                ),
              ),
              Center(
                child: SizedBox(
                  width: buttonWidth,
                  height: 52,
                  child: ElevatedButton(
                    onPressed: loading ? null : _predict,
                    child: loading
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : const Text(
                            'Predict',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Text(result, style: const TextStyle(fontSize: 18)),
              ),
              const SizedBox(height: 18),
              Center(
                child: Text(
                  'API URL: ${apiUrl}',
                  style: const TextStyle(fontSize: 12, color: Colors.black54),
                ),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
