import 'dart:convert';

import 'package:flutter/services.dart';

import '../../dm/model/hanja_definition.dart';

Future<Map<String, HanjaDefinition>> loadSimpleDicDictionary({
  AssetBundle? bundle,
}) async {
  final json = jsonDecode(
    await (bundle ?? rootBundle).loadString('assets/hanja_dictionary.json'),
  ) as Map<String, dynamic>;
  return Map.unmodifiable(
    json.map(
      (character, value) =>
          MapEntry(character, _parseEntry(value as Map<String, dynamic>)),
    ),
  );
}

HanjaDefinition _parseEntry(Map<String, dynamic> row) {
  final readings = {
    for (final r in row['readings'] as List)
      r['id'] as String: '${(r['huns'] as List).join('/')} ${r['sound']}',
  };
  final formation = row['formation'] as Map<String, dynamic>? ?? {};
  final components = row['modernComponents'] as Map<String, dynamic>? ?? {};
  final radical = row['radical'] as Map<String, dynamic>;
  final strokes = row['strokes'] as Map<String, dynamic>;
  final explanations = <HanjaExplanation>[];
  void explain(String title, String? text) {
    if (text != null && text.isNotEmpty) {
      explanations.add(HanjaExplanation(title, text));
    }
  }

  explain('제자 원리', _formationLabels[formation['classification']]);
  final parts = components['components'] as List? ?? [];
  if (parts.isNotEmpty) {
    explain(
      '글자 구성',
      parts.map((p) => '${p['character']} ${p['label']}').join(' · '),
    );
  }
  explain('자형 풀이', components['explanation'] as String?);
  explain('자형 풀이 참고', components['note'] as String?);
  for (final hypothesis in formation['hypotheses'] as List? ?? []) {
    explain('자원 해석 · ${hypothesis['name']}', hypothesis['summary'] as String);
  }
  for (final evidence in formation['classicalEvidence'] as List? ?? []) {
    explain('문헌 근거 · ${evidence['title']}', evidence['summary'] as String);
  }
  explain('자원 참고', formation['note'] as String?);
  return HanjaDefinition(
    character: row['headword'] as String,
    referenceNotes: List.unmodifiable([
      for (final note in (row['editorial']['notes'] as List)) note as String,
      for (final conflict in (row['editorial']['conflicts'] as List))
        conflict as String,
    ]),
    readings: List.unmodifiable(readings.values),
    readingNotes: List.unmodifiable([
      for (final r in row['readings'] as List)
        if ((r['note'] as String? ?? '').isNotEmpty)
          '${readings[r['id']]}: ${r['note']}',
    ]),
    meanings: List.unmodifiable([
      for (final m in row['meanings'] as List)
        HanjaMeaning(
          m['short'] as String,
          m['definition'] as String,
          readings[m['readingRef']]!,
        ),
    ]),
    formation: formation['summary'] as String? ?? '',
    mnemonic: formation['mnemonic'] as String? ?? '',
    explanations: List.unmodifiable(explanations),
    radical:
        '${radical['character']} ${(radical['koreanNames'] as List).join('/')} (${radical['number']}번)',
    residualStrokes: radical['residualStrokes'] as int?,
    strokes: strokes['total'] as int,
    strokeOrderUrl:
        (strokes['strokeOrder'] as Map?)?['sourceUrl'] as String? ?? '',
    gradeLabels: List.unmodifiable([
      for (final grade in row['gradeMemberships'] as List)
        '${grade['systemId'] == 'daehan-geomjeonghoe-hanja' ? '대한검정회' : grade['systemId']} ${grade['introducedLevel']}${grade['introducedLevel'] == '사범' ? '' : '급'}',
    ]),
    historicalForms: List.unmodifiable([
      for (final form in row['historicalForms'] as List)
        if ((form['images'] as List).isNotEmpty)
          HanjaHistoricalForm(
            _historicalLabels[form['type']] ?? form['type'] as String,
            form['sourceUrl'] as String,
            List.unmodifiable([
              for (final image in form['images'] as List)
                HanjaHistoricalImage(
                  image['sourceUrl'] as String,
                  image['caption'] as String,
                ),
            ]),
          ),
    ]),
    compounds: List.unmodifiable([
      for (final c in row['compounds'] as List)
        HanjaCompound(
          c['word'] as String,
          c['reading'] as String,
          c['definition'] as String,
          examples: List.unmodifiable([
            for (final example in c['examples'] as List)
              example['text'] as String,
          ]),
        ),
    ]),
    relatedCharacters: List.unmodifiable([
      for (final related in row['relatedCharacters'] as List)
        HanjaRelatedCharacter(
          related['character'] as String,
          _relationLabels[related['relation']] ?? related['relation'] as String,
          related['note'] as String,
        ),
    ]),
  );
}

const _formationLabels = {
  'pictograph': '상형',
  'ideograph': '지사',
  'compound_ideograph': '회의',
  'phonosemantic': '형성',
  'phonetic_loan': '가차',
  'derivative_cognate': '전주',
  'uncertain': '미상',
};
const _historicalLabels = {
  'oracle_bone': '갑골문',
  'bronze': '금문',
  'bamboo_silk': '죽백문',
  'small_seal': '소전',
  'other': '기타 고대 자형',
};
const _relationLabels = {
  'synonym': '동의자',
  'near_synonym': '유의자',
  'antonym': '반의자',
  'paired_concept': '짝을 이루는 개념',
  'reciprocal_concept': '서로 대응하는 개념',
  'formation_related': '자원 관련',
  'historical_shape_related': '고대 자형 관련',
  'confusable': '혼동하기 쉬운 글자',
};
