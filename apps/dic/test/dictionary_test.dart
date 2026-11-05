import 'package:flutter_test/flutter_test.dart';
import 'package:hanja_app/adapter/dictionary/naver_dictionary.dart';

void main() {
  test('Naver link encodes the selected character', () {
    final uri = dictionaryUri('大');
    expect(uri.host, 'hanja.dict.naver.com');
    expect(Uri.decodeComponent(uri.fragment), '/search?range=all&query=大');
  });
}
