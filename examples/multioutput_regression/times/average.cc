#include <bits/stdc++.h>

using namespace std;

int main() {
  string line;
  unordered_map<string, int> frec;
  unordered_map<string, double> val;
  while (getline(cin, line)) {
    string key;
    double value;
    stringstream ss(line);
    ss >> key >> value;
    frec[key]++;
    val[key] += value;
  }
  for (auto &i : frec) {
    val[i.first] /= (double) i.second;
    cout << i.first << " " << val[i.first] << endl;
  }
  return 0;
}
