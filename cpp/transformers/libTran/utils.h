#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace std;

extern inline FILE *fopen_check(const char *path, const char *mode,
                                const char *file, int line) {

  FILE *fp = fopen(path, mode);

  if (fp == NULL) {

    cerr << "Error: Failed to open file " << path << " at " << file << " : "
         << line << endl;

    exit(EXIT_FAILURE);
  }
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

extern inline void fread_check(void *ptr, size_t size, size_t nmemb,
                               FILE *stream, const char *file, int line) {

  size_t result = fread(ptr, size, nmemb, stream);

  if (result != nmemb) {

    if (feof(stream)) {
      cerr << "Error: Unexpected end of file at " << file << " : " << line
           << endl;
    } else if (ferror(stream)) {
      cerr << "Error: File read error at " << file << ": " << line << endl;
    } else {
      cerr << "Error: Partial read at " << file << " : " << line
           << ". Expected " << nmemb << " elements, read " << result << endl;
    }

    exit(EXIT_FAILURE);
  }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__);

#endif
