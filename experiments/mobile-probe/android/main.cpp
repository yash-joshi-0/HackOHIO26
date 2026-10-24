#include <cstdio>

extern "C" int meshthing_native_smoke(const char *, const char *);

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s fixture.gguf result.json\n", argv[0]);
        return 2;
    }
    return meshthing_native_smoke(argv[1], argv[2]);
}
