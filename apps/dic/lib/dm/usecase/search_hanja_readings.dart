/// Searches the bundled meanings and readings without opening an external page.
class SearchHanjaReadings {
  const SearchHanjaReadings(this.readings);
  final Map<String, String> readings;

  List<String> call(String query) {
    final whitespace = RegExp(r'\s+');
    final terms = query.trim().split(whitespace);
    if (terms.first.isEmpty) return [];
    return readings.entries
        .where((entry) {
          final reading = entry.value.replaceAll(whitespace, '');
          return terms.every(
            (term) => entry.key.contains(term) || reading.contains(term),
          );
        })
        .map((entry) => entry.key)
        .toList(growable: false);
  }
}
