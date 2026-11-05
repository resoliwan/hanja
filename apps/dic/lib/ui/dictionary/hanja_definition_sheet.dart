import 'hanja_historical_images.dart';

import 'package:flutter/material.dart';

import '../../dm/model/hanja_definition.dart';

class HanjaDefinitionSheet extends StatelessWidget {
  const HanjaDefinitionSheet({
    super.key,
    required this.character,
    required this.definition,
    required this.onOpenExternal,
  });

  final String character;
  final HanjaDefinition? definition;
  final VoidCallback onOpenExternal;

  @override
  Widget build(BuildContext context) {
    final entry = definition;
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: .72,
      minChildSize: .35,
      maxChildSize: .94,
      builder: (context, controller) => SafeArea(
        top: false,
        child: ListView(
          controller: controller,
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(character, style: const TextStyle(fontSize: 56)),
                ),
                IconButton(
                  tooltip: '닫기',
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close_rounded),
                ),
              ],
            ),
            if (entry == null)
              const Text('등록된 사전 정의가 없습니다.')
            else ...[
              Text(
                entry.readings.join(' · '),
                style: const TextStyle(fontSize: 20),
              ),
              if (entry.gradeLabels.isNotEmpty)
                Text(entry.gradeLabels.join(' · ')),
              if (entry.mnemonic.isNotEmpty) ...[
                _heading('한 줄 외우기'),
                Text(
                  entry.mnemonic,
                  style: const TextStyle(fontSize: 18, height: 1.6),
                ),
              ],
              if (entry.formation.isNotEmpty) ...[
                _heading('자원'),
                Text(entry.formation),
              ],
              for (final explanation in entry.explanations) ...[
                _heading(explanation.title),
                Text(explanation.text, style: const TextStyle(height: 1.6)),
              ],
              if (entry.historicalForms.isNotEmpty) ...[
                _heading('고대 자형'),
                HanjaHistoricalImages(forms: entry.historicalForms),
              ],
              _heading('부수와 획수'),
              Text(
                '${entry.radical} · 총 ${entry.strokes}획'
                '${entry.residualStrokes == null ? '' : ' · 부수 밖 ${entry.residualStrokes}획'}',
              ),
              if (entry.strokeOrderUrl.isNotEmpty)
                TextButton(
                  onPressed: () =>
                      openDictionarySource(context, entry.strokeOrderUrl),
                  child: const Text('필순 보기'),
                ),
              _heading('뜻과 음'),
              for (final note in entry.readingNotes) Text(note),
              for (final (index, meaning) in entry.meanings.indexed)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Text(
                    '${index + 1}. ${meaning.short} (${meaning.reading})\n${meaning.definition}',
                    style: const TextStyle(fontSize: 16, height: 1.6),
                  ),
                ),
              if (entry.compounds.isNotEmpty) ...[
                _heading('관련 단어'),
                for (final compound in entry.compounds)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: Text(
                      '${compound.word}  ${compound.reading}\n${compound.definition}'
                      '${compound.examples.isEmpty ? '' : '\n${compound.examples.join('\n')}'}',
                      style: const TextStyle(fontSize: 16, height: 1.6),
                    ),
                  ),
              ],
              if (entry.relatedCharacters.isNotEmpty) ...[
                _heading('관련 한자'),
                for (final related in entry.relatedCharacters)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: Text(
                      '${related.character} · ${related.relation}\n${related.note}',
                      style: const TextStyle(fontSize: 16, height: 1.6),
                    ),
                  ),
              ],
            ],
            if (entry != null && entry.referenceNotes.isNotEmpty)
              ExpansionTile(
                title: const Text('사전 설명 참고'),
                children: [
                  for (final note in entry.referenceNotes)
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(note),
                    ),
                ],
              ),
            const SizedBox(height: 16),
            OutlinedButton(
              onPressed: onOpenExternal,
              child: const Text('네이버 한자사전에서 보기'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _heading(String title) => Padding(
    padding: const EdgeInsets.only(top: 24, bottom: 12),
    child: Text(
      title,
      style: const TextStyle(fontSize: 19, fontWeight: FontWeight.w600),
    ),
  );
}
