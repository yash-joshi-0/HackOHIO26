#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

int mesh_shard_main(int argc, char ** argv);

// One session, loopback only. The Iroh bridge owns peer authorization.
int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr, "Usage: worker PORT MODEL BEGIN END CTX GPU BUDGET PROMPT LOGITS\n");
        return 2;
    }
    char * end = nullptr;
    const long port = strtol(argv[1], &end, 10);
    if (!*argv[1] || *end || port < 1024 || port > 65535) return 2;
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) return 1;
    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(listener, 1)) {
        perror("native listener"); close(listener); return 1;
    }
    fprintf(stderr, "native shard listening on loopback:%ld\n", port);
    const int connection = accept(listener, nullptr, nullptr);
    close(listener);
    if (connection < 0) return 1;
    timeval timeout{30, 0};
    setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char preface[5];
    size_t received = 0;
    while (received < sizeof(preface)) {
        const auto n = recv(connection, preface + received, sizeof(preface) - received, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(connection); return 1; }
        received += size_t(n);
    }
    if (memcmp(preface, "MSH1\n", sizeof(preface))) { close(connection); return 1; }
    timeout.tv_sec = 120;
    setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(connection, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    if (dup2(connection, STDIN_FILENO) < 0 || dup2(connection, STDOUT_FILENO) < 0) {
        close(connection); return 1;
    }
    close(connection);
    return mesh_shard_main(argc - 1, argv + 1);
}
