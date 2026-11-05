import 'package:flutter/material.dart';

class DictionarySearchField extends StatelessWidget {
  const DictionarySearchField({
    super.key,
    required this.controller,
    required this.onChanged,
  });
  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) => TextField(
    controller: controller,
    onChanged: onChanged,
    onTapOutside: (_) => FocusScope.of(context).unfocus(),
    textInputAction: TextInputAction.search,
    onSubmitted: (_) => FocusScope.of(context).unfocus(),
    style: const TextStyle(fontSize: 16),
    decoration: InputDecoration(
      filled: true,
      fillColor: const Color(0xff252525),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
    ),
  );
}
