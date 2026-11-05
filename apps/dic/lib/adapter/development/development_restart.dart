import 'dart:convert';

import 'package:http/http.dart' as http;

class DevelopmentRestart {
  const DevelopmentRestart({required this.url, required this.token});
  final String url;
  final String token;

  Future<void> call() async {
    final client = http.Client();
    try {
      final response = await client
          .post(
            Uri.parse('$url/reload'),
            headers: {'Authorization': 'Bearer $token'},
          )
          .timeout(const Duration(seconds: 200));
      if (response.statusCode != 200 ||
          (jsonDecode(response.body) as Map<String, dynamic>)['ok'] != true) {
        throw StateError('Development restart failed');
      }
    } finally {
      client.close();
    }
  }
}
