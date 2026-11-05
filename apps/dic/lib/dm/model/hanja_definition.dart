class HanjaDefinition {
  const HanjaDefinition({
    required this.character,
    required this.readings,
    required this.meanings,
    required this.formation,
    required this.radical,
    required this.strokes,
    required this.compounds,
    this.mnemonic = '',
    this.explanations = const [],
    this.historicalForms = const [],
    this.readingNotes = const [],
    this.referenceNotes = const [],
    this.relatedCharacters = const [],
    this.gradeLabels = const [],
    this.strokeOrderUrl = '',
    this.residualStrokes,
  });

  final String character, formation, radical;
  final String mnemonic, strokeOrderUrl;
  final int? residualStrokes;
  final List<HanjaExplanation> explanations;
  final List<HanjaHistoricalForm> historicalForms;
  final List<String> readingNotes, gradeLabels, referenceNotes;
  final List<HanjaRelatedCharacter> relatedCharacters;
  final int strokes;
  final List<String> readings;
  final List<HanjaMeaning> meanings;
  final List<HanjaCompound> compounds;
}

class HanjaMeaning {
  const HanjaMeaning(this.short, this.definition, this.reading);
  final String short, definition, reading;
}

class HanjaCompound {
  const HanjaCompound(
    this.word,
    this.reading,
    this.definition, {
    this.examples = const [],
  });
  final List<String> examples;
  final String word, reading, definition;
}

class HanjaExplanation {
  const HanjaExplanation(this.title, this.text);
  final String title, text;
}

class HanjaHistoricalForm {
  const HanjaHistoricalForm(this.label, this.sourceUrl, this.images);
  final String label, sourceUrl;
  final List<HanjaHistoricalImage> images;
}

class HanjaHistoricalImage {
  const HanjaHistoricalImage(this.url, this.caption);
  final String url, caption;
}

class HanjaRelatedCharacter {
  const HanjaRelatedCharacter(this.character, this.relation, this.note);
  final String character, relation, note;
}
