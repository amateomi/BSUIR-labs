#include <iostream>
#include <fstream>

#include <list>
#include <unordered_map>
#include <stack>

using Text = std::list<std::string>;
using ParseResult = std::list<bool>;

Text GetStringsFromFile(std::ifstream &source);
void WriteResults(std::ofstream &target, const ParseResult &results);

void PrintText(const Text &text);
void PrintResult(const ParseResult &result);

ParseResult ParseText(const Text &text);
bool ParseString(const std::string &string);

bool IsOpenBracket(char c);
bool IsCloseBracket(char c);

// 'argv[1]' - source file path
// 'argv[2]' - target file path
int main(int argc, char *argv[]) {
  try {
    if (argc != 3) throw std::logic_error("Two arguments expected");

    std::cout << "Source file: " << argv[1] << std::endl;
    std::cout << "Target file: " << argv[2] << std::endl;

    std::cout << "Opening source file..." << std::endl;
    std::ifstream source(argv[1], std::ios::in);

    std::cout << "Getting strings from source file..." << std::endl;
    auto text = GetStringsFromFile(source);
    source.close();

    std::cout << "Source file strings:" << std::endl;
    PrintText(text);

    std::cout << "Parsing text..." << std::endl;
    auto results = ParseText(text);

    std::cout << "Result:" << std::endl;
    PrintResult(results);

    std::cout << "Opening target file..." << std::endl;
    std::ofstream target(argv[2], std::ios::out | std::ios::trunc);

    std::cout << "Writing results to target..." << std::endl;
    WriteResults(target, results);
    target.close();

  } catch (std::exception &error) {
    std::cout << "Error: " << error.what() << std::endl;
  }
  return 0;
}

Text GetStringsFromFile(std::ifstream &source) {
  Text text;
  std::string record;
  while (!source.eof()) {
    std::getline(source, record, '\n');
    text.push_back(record);
  }
  return text;
}

void WriteResults(std::ofstream &target, const ParseResult &results) {
  for (auto record : results) {
    target << std::boolalpha << record << std::endl;
  }
}

void PrintText(const Text &text) {
  for (const auto &string: text) {
    std::cout << string << std::endl;
  }
}

void PrintResult(const ParseResult &result) {
  for (const auto &alpha: result) {
    std::cout << std::boolalpha << alpha << std::endl;
  }
}

ParseResult ParseText(const Text &text) {
  ParseResult result;
  for (const auto &string: text) {
    result.push_back(ParseString(string));
  }
  return result;
}

bool ParseString(const std::string &string) {
  static const std::unordered_map<char, char> kBrackets = {
      {')', '('},
      {']', '['},
      {'}', '{'},
      {'>', '<'}
  };

  std::stack<char> stack;

  for (auto item: string) {
    if (IsOpenBracket(item)) {
      stack.push(item);
    } else if (IsCloseBracket(item)) {
      if (stack.empty()) return false;
      if (stack.top() != kBrackets.at(item)) return false;
      stack.pop();
    }
  }

  return stack.empty();
}

bool IsOpenBracket(char c) {
  return c == '('
      || c == '['
      || c == '{'
      || c == '<';
}

bool IsCloseBracket(char c) {
  return c == ')'
      || c == ']'
      || c == '}'
      || c == '>';
}