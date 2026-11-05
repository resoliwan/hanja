abstract interface class HanjaBookmarks {
  Future<Set<String>> read();
  Future<void> write(Set<String> characters);
}
