/*
 * C-Kernel-Engine HTTP Streaming Server
 *
 * A simple HTTP server for streaming token generation.
 * Supports Server-Sent Events (SSE) for real-time streaming.
 *
 * API Endpoints:
 *   POST /v1/completions - Text completion with streaming
 *   GET  /health         - Health check
 *   GET  /               - Welcome message
 *
 * Build:
 *   make tools/ck_server
 *
 * Usage:
 *   ./build/ck_server --port 8080 --model weights.bin
 *
 * By Anthony Shivakumar
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sys/select.h>
#include <fcntl.h>

#define CK_VERSION "0.1.0"
#define MAX_REQUEST_SIZE (1024 * 1024)
#define MAX_RESPONSE_SIZE (1024 * 1024)
#define MAX_CONNECTIONS 64

static volatile sig_atomic_t g_shutdown = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
}

/* HTTP Response helpers */
static void send_response(int fd, int status, const char *content_type, const char *body) {
    const char *status_text = "OK";
    if (status == 400) status_text = "Bad Request";
    if (status == 404) status_text = "Not Found";
    if (status == 500) status_text = "Internal Server Error";

    char header[1024];
    int header_len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, strlen(body));

    write(fd, header, header_len);
    write(fd, body, strlen(body));
}

static void send_json(int fd, int status, const char *json) {
    send_response(fd, status, "application/json", json);
}

static void start_sse(int fd) {
    const char *header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    write(fd, header, strlen(header));
}

static void send_sse_event(int fd, const char *data) {
    char buf[4096];
    int len = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    write(fd, buf, len);
}

/* Simple JSON parsing helpers */
static const char *json_find_key(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return NULL;
    pos = strchr(pos, ':');
    if (!pos) return NULL;
    pos++;
    while (*pos == ' ' || *pos == '\t') pos++;
    return pos;
}

static int json_parse_string(const char *pos, char *buf, int max_len) {
    if (*pos != '"') return -1;
    pos++;
    int len = 0;
    while (*pos && *pos != '"' && len < max_len - 1) {
        if (*pos == '\\' && pos[1]) {
            pos++;
            switch (*pos) {
                case 'n': buf[len++] = '\n'; break;
                case 'r': buf[len++] = '\r'; break;
                case 't': buf[len++] = '\t'; break;
                default: buf[len++] = *pos; break;
            }
        } else {
            buf[len++] = *pos;
        }
        pos++;
    }
    buf[len] = '\0';
    return len;
}

static int json_parse_int(const char *pos) {
    return atoi(pos);
}

static double json_parse_float(const char *pos) {
    return atof(pos);
}

static bool json_parse_bool(const char *pos) {
    return strncmp(pos, "true", 4) == 0;
}

/* Request handler context */
typedef struct {
    int client_fd;
    char *request;
    int request_len;
} RequestContext;

/* Demo token generator (placeholder) */
static const char *demo_tokens[] = {
    "Once", " upon", " a", " time", ",", " in", " a", " land",
    " far", " away", ",", " there", " lived", " a", " brave",
    " knight", ".", " He", " was", " known", " for", " his",
    " courage", " and", " wisdom", "."
};
static int demo_token_count = sizeof(demo_tokens) / sizeof(demo_tokens[0]);

static void handle_completions(int fd, const char *body, bool stream) {
    /* Parse request */
    char prompt[4096] = "";
    int max_tokens = 128;
    float temperature = 0.7f;

    const char *pos;
    if ((pos = json_find_key(body, "prompt"))) {
        json_parse_string(pos, prompt, sizeof(prompt));
    }
    if ((pos = json_find_key(body, "max_tokens"))) {
        max_tokens = json_parse_int(pos);
    }
    if ((pos = json_find_key(body, "temperature"))) {
        temperature = json_parse_float(pos);
    }
    if ((pos = json_find_key(body, "stream"))) {
        stream = json_parse_bool(pos);
    }

    printf("[INFO] Completion request: prompt='%.50s...' max_tokens=%d stream=%d\n",
           prompt, max_tokens, stream);

    if (stream) {
        start_sse(fd);

        /* Stream tokens */
        for (int i = 0; i < max_tokens && i < demo_token_count; i++) {
            char event[1024];
            snprintf(event, sizeof(event),
                "{\"choices\":[{\"text\":\"%s\",\"index\":0}],\"model\":\"ck-demo\"}",
                demo_tokens[i]);
            send_sse_event(fd, event);

            /* Simulate generation delay */
            usleep(50000);  /* 50ms per token */
        }

        send_sse_event(fd, "[DONE]");
    } else {
        /* Non-streaming: collect all tokens */
        char response[MAX_RESPONSE_SIZE];
        char text[8192] = "";
        int text_len = 0;

        for (int i = 0; i < max_tokens && i < demo_token_count; i++) {
            int tok_len = strlen(demo_tokens[i]);
            if (text_len + tok_len < (int)sizeof(text) - 1) {
                strcpy(text + text_len, demo_tokens[i]);
                text_len += tok_len;
            }
        }

        snprintf(response, sizeof(response),
            "{\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"length\"}],"
            "\"model\":\"ck-demo\",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}",
            text, (int)strlen(prompt) / 4, max_tokens);

        send_json(fd, 200, response);
    }
}

static void handle_request(int client_fd, const char *request, int request_len) {
    /* Parse request line */
    char method[16] = "";
    char path[256] = "";

    sscanf(request, "%15s %255s", method, path);

    printf("[INFO] %s %s\n", method, path);

    /* Find request body */
    const char *body = strstr(request, "\r\n\r\n");
    if (body) body += 4;

    /* Route request */
    if (strcmp(method, "OPTIONS") == 0) {
        /* CORS preflight */
        const char *response =
            "HTTP/1.1 200 OK\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Content-Length: 0\r\n"
            "\r\n";
        write(client_fd, response, strlen(response));
    }
    else if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        /* Welcome page */
        char html[2048];
        snprintf(html, sizeof(html),
            "<!DOCTYPE html>\n"
            "<html><head><title>C-Kernel-Engine Server</title></head>\n"
            "<body>\n"
            "<h1>C-Kernel-Engine Inference Server v%s</h1>\n"
            "<p>By Anthony Shivakumar</p>\n"
            "<h2>API Endpoints:</h2>\n"
            "<ul>\n"
            "<li>POST /v1/completions - Text completion</li>\n"
            "<li>GET /health - Health check</li>\n"
            "</ul>\n"
            "<h2>Example:</h2>\n"
            "<pre>\n"
            "curl -X POST http://localhost:8080/v1/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"prompt\": \"Hello\", \"max_tokens\": 50, \"stream\": true}'\n"
            "</pre>\n"
            "</body></html>\n",
            CK_VERSION);
        send_response(client_fd, 200, "text/html", html);
    }
    else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        char json[256];
        snprintf(json, sizeof(json),
            "{\"status\":\"ok\",\"version\":\"%s\"}", CK_VERSION);
        send_json(client_fd, 200, json);
    }
    else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        send_json(client_fd, 200,
            "{\"data\":[{\"id\":\"ck-demo\",\"object\":\"model\",\"owned_by\":\"c-kernel-engine\"}]}");
    }
    else if (strcmp(method, "POST") == 0 &&
             (strcmp(path, "/v1/completions") == 0 ||
              strcmp(path, "/v1/chat/completions") == 0)) {
        if (body) {
            handle_completions(client_fd, body, false);
        } else {
            send_json(client_fd, 400, "{\"error\":\"No request body\"}");
        }
    }
    else {
        send_json(client_fd, 404, "{\"error\":\"Not found\"}");
    }
}

static void *client_thread(void *arg) {
    RequestContext *ctx = (RequestContext *)arg;

    handle_request(ctx->client_fd, ctx->request, ctx->request_len);

    close(ctx->client_fd);
    free(ctx->request);
    free(ctx);
    return NULL;
}

static void print_banner(void) {
    printf("\n");
    printf("\033[36m\033[1m");
    printf("   ____      _  __                    _   ____                           \n");
    printf("  / ___|    | |/ /___ _ __ _ __   ___| | / ___|  ___ _ ____   _____ _ __ \n");
    printf(" | |   _____| ' // _ \\ '__| '_ \\ / _ \\ | \\___ \\ / _ \\ '__\\ \\ / / _ \\ '__|\n");
    printf(" | |__|_____| . \\  __/ |  | | | |  __/ |  ___) |  __/ |   \\ V /  __/ |   \n");
    printf("  \\____|    |_|\\_\\___|_|  |_| |_|\\___|_| |____/ \\___|_|    \\_/ \\___|_|   \n");
    printf("\033[0m");
    printf("\n");
    printf("\033[32m  C-Kernel-Engine Server v%s\033[0m\n", CK_VERSION);
    printf("  By Anthony Shivakumar\n");
    printf("\n");
    printf("  Streaming Inference API Server\n");
    printf("  ============================================================\n");
    printf("\n");
}

int main(int argc, char **argv) {
    int port = 8080;
    const char *host = "0.0.0.0";
    const char *model_path = NULL;
    bool show_banner = true;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--no-banner") == 0) {
            show_banner = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --port N      Port to listen on (default: 8080)\n");
            printf("  --host ADDR   Host to bind to (default: 0.0.0.0)\n");
            printf("  --model PATH  Path to model weights\n");
            printf("  --no-banner   Don't show startup banner\n");
            printf("  --help        Show this help\n");
            return 0;
        }
    }

    if (show_banner) {
        print_banner();
    }

    /* Set up signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    if (model_path) {
        printf("Loading model from: %s\n", model_path);
        /* TODO: Load actual model */
    } else {
        printf("\033[33mWarning: No model loaded, using demo responses.\033[0m\n");
        printf("Use --model PATH to load actual weights.\n\n");
    }

    /* Create socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(host);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, MAX_CONNECTIONS) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    printf("Server listening on http://%s:%d\n", host, port);
    printf("\nEndpoints:\n");
    printf("  POST /v1/completions      - Text completion\n");
    printf("  POST /v1/chat/completions - Chat completion\n");
    printf("  GET  /v1/models           - List models\n");
    printf("  GET  /health              - Health check\n");
    printf("\nPress Ctrl+C to stop.\n\n");

    /* Accept loop with select() for interruptible waiting */
    while (!g_shutdown) {
        fd_set read_fds;
        struct timeval tv;

        FD_ZERO(&read_fds);
        FD_SET(server_fd, &read_fds);

        /* Wait up to 1 second for a connection */
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int sel = select(server_fd + 1, &read_fds, NULL, NULL, &tv);
        if (sel < 0) {
            if (errno == EINTR) continue;  /* Signal interrupted, check g_shutdown */
            perror("select");
            break;
        }
        if (sel == 0) {
            /* Timeout - loop back and check g_shutdown */
            continue;
        }

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        /* Read request */
        char *request = (char *)malloc(MAX_REQUEST_SIZE);
        if (!request) {
            close(client_fd);
            continue;
        }

        int request_len = read(client_fd, request, MAX_REQUEST_SIZE - 1);
        if (request_len <= 0) {
            free(request);
            close(client_fd);
            continue;
        }
        request[request_len] = '\0';

        /* Handle in thread */
        RequestContext *ctx = (RequestContext *)malloc(sizeof(RequestContext));
        ctx->client_fd = client_fd;
        ctx->request = request;
        ctx->request_len = request_len;

        pthread_t thread;
        if (pthread_create(&thread, NULL, client_thread, ctx) != 0) {
            handle_request(client_fd, request, request_len);
            close(client_fd);
            free(request);
            free(ctx);
        } else {
            pthread_detach(thread);
        }
    }

    printf("\nShutting down...\n");
    close(server_fd);
    return 0;
}
