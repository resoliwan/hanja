import 'dart:io';

/// The selected photo is held in memory; remove its temporary capture file.
Future<void> deleteCapturedFile(String path) async {
  try {
    await File(path).delete();
  } on FileSystemException {
    /* Already removed. */
  }
}
