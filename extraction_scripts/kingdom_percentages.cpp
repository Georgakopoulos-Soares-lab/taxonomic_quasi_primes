#include <iostream>
#include <fstream>
#include <string>
#include <google/sparse_hash_map>
#include <chrono>
#include <unordered_map>
#include <utility>
#include <iomanip>

using namespace std;
using google::sparse_hash_map;
using namespace std::chrono;

struct str_hash {
    size_t operator()(const string& str) const {
        return hash<string>()(str);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];

    sparse_hash_map<string, pair<string, int>, str_hash> kmerMap;
    ifstream mappingsFile(input_file);
    if (!mappingsFile) {
        cerr << "Error: Cannot open input file " << input_file << endl;
        return 1;
    }

    string line;
    int fileCounter = 0;
    unordered_map<string, int> kingdomFileCounts;

    while (getline(mappingsFile, line)) {
        size_t spacePos = line.find_first_of(" ");
        string kingdom = line.substr(spacePos + 1);
        kingdomFileCounts[kingdom]++;
    }
    cout << "Total number of kingdoms are: " << kingdomFileCounts.size() << endl;
    mappingsFile.close();

    mappingsFile.open(input_file);
    if (!mappingsFile) {
        cerr << "Error: Cannot reopen input file " << input_file << endl;
        return 1;
    }

    auto lastTime = system_clock::now();

    while (getline(mappingsFile, line)) {
        size_t spacePos = line.find_first_of(" ");
        string filePath = line.substr(0, spacePos);
        string kingdom = line.substr(spacePos + 1);

        ifstream file(filePath);
        if (!file) {
            cerr << "Error: Cannot open file " << filePath << endl;
            continue;
        }

        string kmer;
        while (file >> kmer) {
            auto it = kmerMap.find(kmer);
            if (it != kmerMap.end()) {
                if (it->second.first != kingdom) it->second.first = "cross-kingdom";
                else ++it->second.second;
            } else {
                kmerMap[kmer] = {kingdom, 1};
            }
        }
        file.close();

        fileCounter++;
        auto currentTime = system_clock::now();
        auto duration = duration_cast<minutes>(currentTime - lastTime);

        if (duration.count() >= 15) {
            cout << "Files processed: " << fileCounter << ", Hashmap size: " << kmerMap.size() << endl;
            lastTime = currentTime;
        }
    }
    mappingsFile.close();

    cout << "Final count - Files processed: " << fileCounter << ", Hashmap size: " << kmerMap.size() << endl;

    ofstream output(output_file);
    if (!output) {
        cerr << "Error: Cannot open output file " << output_file << endl;
        return 1;
    }

    for (const auto& kv : kmerMap) {
        if (kv.second.first != "cross-kingdom") {
            float percentage = static_cast<float>(kv.second.second) /
                               static_cast<float>(kingdomFileCounts[kv.second.first]) * 100.0f;

            output << kv.first << " " << kv.second.first << " " << fixed << setprecision(2) << percentage << "%" << "\n";
        }
    }
    output.close();
    return 0;
}

