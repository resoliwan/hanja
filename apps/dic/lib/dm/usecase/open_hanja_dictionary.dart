import '../../out/external_dictionary.dart';

class OpenHanjaDictionary {
  const OpenHanjaDictionary(this._dictionary);
  final ExternalDictionary _dictionary;
  Future<void> call(String query) => _dictionary.open(query);
}
