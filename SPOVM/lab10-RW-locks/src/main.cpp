#include "app.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    throw std::domain_error("Usage: database <table path>");
  }
  App app(argv[1]);
  app.run();
}
