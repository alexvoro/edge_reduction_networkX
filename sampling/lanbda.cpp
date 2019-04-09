//Program to test the C++ lambda syntax and initializer lists
#include <iostream>
#include <vector>

int main()
{
  // Test lambda
  std::cout << [](int m, int n) { return m + n;} (2,4) << '\n';

  // Test initializer lists and range based for loop
  std::vector<int> V{1,2,3};

  std::cout << "V =\n";
  for(auto e : V) {
    std::cout << e << '\n';
  }

  return 0;
}