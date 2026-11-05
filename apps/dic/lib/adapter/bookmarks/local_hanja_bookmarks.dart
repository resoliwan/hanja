import 'package:shared_preferences/shared_preferences.dart';

import '../../out/hanja_bookmarks.dart';

class LocalHanjaBookmarks implements HanjaBookmarks {
  final _preferences = SharedPreferencesAsync();
  static const _key = 'hanja.bookmarks.v1';
  @override
  Future<Set<String>> read() async =>
      (await _preferences.getStringList(_key) ?? []).toSet();
  @override
  Future<void> write(Set<String> characters) =>
      _preferences.setStringList(_key, characters.toList());
}
