import 'package:url_launcher/url_launcher.dart';

import '../../out/external_dictionary.dart';

Uri dictionaryUri(String query) => Uri.parse(
  'https://hanja.dict.naver.com/#/search?range=all&query=${Uri.encodeComponent(query)}',
);

class NaverDictionary implements ExternalDictionary {
  @override
  Future<void> open(String query) async {
    if (!await launchUrl(
      dictionaryUri(query),
      mode: LaunchMode.externalApplication,
    )) {
      throw StateError('Could not open the dictionary');
    }
  }
}
