#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace Utils {

// File Error Handling

inline FILE *fopenCheck(const std::string &path, const std::string &mode,
                        const char *file, int line) {
  FILE *fp = fopen(path.c_str(), mode.c_str());
  if (fp == nullptr) {
    std::cerr << "Error: Failed to open file '" << path << "' at " << file
              << ":" << line << "\n";
    std::cerr << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n"
              << "  Path: " << path << "\n"
              << "  Mode: " << mode << "\n"
              << "---> HINT 1: dataset files/code have moved to dev/data "
                 "recently (May 20, 2024). "
              << "You may have to mv them from the legacy data/ dir to "
                 "dev/data/(dataset), "
              << "or re-run the data preprocessing script. Refer back to the "
                 "main README\n"
              << "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n";
    std::exit(EXIT_FAILURE);
  }
  return fp;
}

#define fopenCheck(path, mode) Utils::fopenCheck(path, mode, __FILE__, __LINE__)

inline void freadCheck(void *ptr, size_t size, size_t nmemb, FILE *stream,
                       const char *file, int line) {
  size_t result = fread(ptr, size, nmemb, stream);
  if (result != nmemb) {
    if (feof(stream)) {
      std::cerr << "Error: Unexpected end of file at " << file << ":" << line
                << "\n";
    } else if (ferror(stream)) {
      std::cerr << "Error: File read error at " << file << ":" << line << "\n";
    } else {
      std::cerr << "Error: Partial read at " << file << ":" << line
                << ". Expected " << nmemb << " elements, read " << result
                << "\n";
    }
    std::cerr << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n"
              << "  Expected elements: " << nmemb << "\n"
              << "  Read elements: " << result << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define freadCheck(ptr, size, nmemb, stream)                                   \
  Utils::freadCheck(ptr, size, nmemb, stream, __FILE__, __LINE__)

inline void fcloseCheck(FILE *fp, const char *file, int line) {
  if (fclose(fp) != 0) {
    std::cerr << "Error: Failed to close file at " << file << ":" << line
              << "\n"
              << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define fcloseCheck(fp) Utils::fcloseCheck(fp, __FILE__, __LINE__)

inline void scloseCheck(int sockfd, const char *file, int line) {
  if (close(sockfd) != 0) {
    std::cerr << "Error: Failed to close socket at " << file << ":" << line
              << "\n"
              << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define scloseCheck(sockfd) Utils::scloseCheck(sockfd, __FILE__, __LINE__)

inline void fseekCheck(FILE *fp, long off, int whence, const char *file,
                       int line) {
  if (fseek(fp, off, whence) != 0) {
    std::cerr << "Error: Failed to seek in file at " << file << ":" << line
              << "\n"
              << "Error details:\n"
              << "  Offset: " << off << "\n"
              << "  Whence: " << whence << "\n"
              << "  File:   " << file << "\n"
              << "  Line:   " << line << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define fseekCheck(fp, off, whence)                                            \
  Utils::fseekCheck(fp, off, whence, __FILE__, __LINE__)

inline void fwriteCheck(const void *ptr, size_t size, size_t nmemb,
                        FILE *stream, const char *file, int line) {
  size_t result = fwrite(ptr, size, nmemb, stream);
  if (result != nmemb) {
    if (feof(stream)) {
      std::cerr << "Error: Unexpected end of file at " << file << ":" << line
                << "\n";
    } else if (ferror(stream)) {
      std::cerr << "Error: File write error at " << file << ":" << line << "\n";
    } else {
      std::cerr << "Error: Partial write at " << file << ":" << line
                << ". Expected " << nmemb << " elements, wrote " << result
                << "\n";
    }
    std::cerr << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n"
              << "  Expected elements: " << nmemb << "\n"
              << "  Written elements: " << result << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define fwriteCheck(ptr, size, nmemb, stream)                                  \
  Utils::fwriteCheck(ptr, size, nmemb, stream, __FILE__, __LINE__)

// Malloc Error Handling

inline void *mallocCheck(size_t size, const char *file, int line) {
  void *ptr = std::malloc(size);
  if (ptr == nullptr) {
    std::cerr << "Error: Memory allocation failed at " << file << ":" << line
              << "\n"
              << "Error details:\n"
              << "  File: " << file << "\n"
              << "  Line: " << line << "\n"
              << "  Size: " << size << " bytes\n";
    std::exit(EXIT_FAILURE);
  }
  return ptr;
}

#define mallocCheck(size) Utils::mallocCheck(size, __FILE__, __LINE__)

// check that all tokens are within range

inline void tokenCheck(const int *tokens, int token_count, int vocab_size,
                       const char *file, int line) {
  for (int i = 0; i < token_count; ++i) {
    if (!(0 <= tokens[i] && tokens[i] < vocab_size)) {
      std::cerr << "Error: Token out of vocabulary at " << file << ":" << line
                << "\n"
                << "Error details:\n"
                << "  File: " << file << "\n"
                << "  Line: " << line << "\n"
                << "  Token: " << tokens[i] << "\n"
                << "  Position: " << i << "\n"
                << "  Vocab: " << vocab_size << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

#define tokenCheck(tokens, count, vocab)                                       \
  Utils::tokenCheck(tokens, count, vocab, __FILE__, __LINE__)

// I/O ops

inline void createDirIfNotExists(const std::string &dir) {
  if (dir.empty()) {
    return;
  }
  struct stat st = {};
  if (stat(dir.c_str(), &st) == -1) {
    if (mkdir(dir.c_str(), 0700) == -1) {
      std::cerr << "ERROR: could not create directory: " << dir << "\n";
      std::exit(EXIT_FAILURE);
    }
    std::cout << "Created directory: " << dir << "\n";
  }
}

inline int findMaxStep(const std::string &output_log_dir) {
  // Find the DONE file in the log dir with the highest step count
  if (output_log_dir.empty()) {
    return -1;
  }
  DIR *dir;
  struct dirent *entry;
  int max_step = -1;
  dir = opendir(output_log_dir.c_str());
  if (dir == nullptr) {
    return -1;
  }
  while ((entry = readdir(dir)) != nullptr) {
    if (std::strncmp(entry->d_name, "DONE_", 5) == 0) {
      int step = std::atoi(entry->d_name + 5);
      if (step > max_step) {
        max_step = step;
      }
    }
  }
  closedir(dir);
  return max_step;
}

inline bool endsWithBin(const std::string &str) {
  // Checks if str ends with ".bin". Could be generalized in the future.
  const std::string suffix = ".bin";
  if (str.size() < suffix.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

} // namespace Utils

#endif // UTILS_HPP
