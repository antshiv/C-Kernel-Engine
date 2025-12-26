/*
 * C-Kernel-Engine Interactive Inference CLI
 *
 * A llama.cpp-style interactive interface for transformer inference.
 * This is the main entry point that provides:
 *   - Interactive chat mode with continuous prompting
 *   - Batch inference mode
 *   - Streaming token output
 *
 * Build:
 *   make tools/ck_main
 *
 * Usage:
 *   ./build/ck_main --model-dir ~/.cache/huggingface/hub/SmolLM-135M
 *
 * By Anthony Shivakumar
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#define CK_VERSION "0.1.0"

/* ANSI color codes */
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"

static volatile sig_atomic_t g_interrupted = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

static void print_banner(void) {
    printf("\n");
    printf(ANSI_CYAN ANSI_BOLD);
    printf("   ____      _  __                    _   _____             _            \n");
    printf("  / ___|    | |/ /___ _ __ _ __   ___| | | ____|_ __   __ _(_)_ __   ___ \n");
    printf(" | |   _____| ' // _ \\ '__| '_ \\ / _ \\ | |  _| | '_ \\ / _` | | '_ \\ / _ \\\n");
    printf(" | |__|_____| . \\  __/ |  | | | |  __/ | | |___| | | | (_| | | | | |  __/\n");
    printf("  \\____|    |_|\\_\\___|_|  |_| |_|\\___|_| |_____|_| |_|\\__, |_|_| |_|\\___|\n");
    printf("                                                      |___/              \n");
    printf(ANSI_RESET);
    printf("\n");
    printf(ANSI_GREEN "  C-Kernel-Engine v%s" ANSI_RESET "\n", CK_VERSION);
    printf("  By Anthony Shivakumar\n");
    printf("\n");
    printf("  Pure C Transformer Inference Engine\n");
    printf("  ============================================================\n");
    printf("\n");
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model PATH       Path to model weights (bump format)\n");
    printf("  --config PATH      Path to model config JSON\n");
    printf("  --tokenizer PATH   Path to tokenizer.json (HuggingFace format)\n");
    printf("  --context N        Context window size (default: 512)\n");
    printf("  --prompt TEXT      Single prompt (non-interactive mode)\n");
    printf("  --interactive      Interactive chat mode (default if no prompt)\n");
    printf("  --temperature F    Sampling temperature (default: 0.7)\n");
    printf("  --top-k N          Top-k sampling (default: 40)\n");
    printf("  --top-p F          Top-p (nucleus) sampling (default: 0.9)\n");
    printf("  --max-tokens N     Maximum tokens to generate (default: 128)\n");
    printf("  --no-banner        Don't show startup banner\n");
    printf("  --color            Enable colored output (default)\n");
    printf("  --no-color         Disable colored output\n");
    printf("  --help             Show this help\n");
    printf("\n");
    printf("Interactive Commands:\n");
    printf("  /help     Show help\n");
    printf("  /clear    Clear conversation\n");
    printf("  /exit     Exit\n");
    printf("  /stats    Show statistics\n");
    printf("  /temp N   Set temperature\n");
    printf("  /top_k N  Set top-k\n");
    printf("  /top_p N  Set top-p\n");
    printf("\n");
}

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * Simple word-level tokenizer for demo purposes.
 * For production, use the Python wrapper with HuggingFace tokenizers,
 * or implement BPE in C (like llama.cpp does).
 */
typedef struct {
    char **vocab;
    int vocab_size;
    int unk_id;
    int eos_id;
    int bos_id;
} SimpleTokenizer;

static int simple_tokenize(const char *text, int *tokens, int max_tokens) {
    /* Very simple whitespace tokenizer - for demo only */
    int count = 0;
    const char *p = text;

    while (*p && count < max_tokens) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Find word end */
        const char *start = p;
        while (*p && !isspace((unsigned char)*p)) p++;

        /* Hash the word to a token ID (very crude) */
        unsigned int hash = 0;
        for (const char *c = start; c < p; c++) {
            hash = hash * 31 + (unsigned char)*c;
        }
        tokens[count++] = hash % 32000;  /* Assume 32k vocab */
    }

    return count;
}

typedef struct {
    /* Model config */
    int num_layers;
    int embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    int context_window;
    float rms_norm_eps;
    float rope_theta;

    /* Inference state */
    uint8_t *memory_base;
    size_t total_bytes;
    float *logits;

    /* Sampling params */
    float temperature;
    int top_k;
    float top_p;
    int max_tokens;

    /* Stats */
    int tokens_generated;
    double total_gen_time_ms;
} CKModel;

/* Placeholder - in real implementation, this calls the generated forward pass */
static int run_forward_pass(CKModel *model, const int *tokens, int num_tokens) {
    (void)model;
    (void)tokens;
    (void)num_tokens;
    /*
     * This would call into the generated model code.
     * For now, we just fill logits with random values for demo.
     */
    if (!model->logits) {
        model->logits = (float*)malloc(model->vocab_size * sizeof(float));
    }

    /* Generate random logits for demo */
    for (int i = 0; i < model->vocab_size; i++) {
        model->logits[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }

    return 0;
}

static int sample_token(CKModel *model) {
    float *logits = model->logits;
    int vocab_size = model->vocab_size;
    float temperature = model->temperature;
    int top_k = model->top_k;

    if (temperature <= 0.0f) {
        /* Greedy: return argmax */
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    /* Apply temperature */
    float *scaled = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        scaled[i] = logits[i] / temperature;
    }

    /* Find top-k */
    int *indices = (int*)malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) indices[i] = i;

    /* Partial sort for top-k */
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        int max_idx = i;
        for (int j = i + 1; j < vocab_size; j++) {
            if (scaled[indices[j]] > scaled[indices[max_idx]]) {
                max_idx = j;
            }
        }
        int tmp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = tmp;
    }

    /* Softmax over top-k */
    float max_val = scaled[indices[0]];
    float sum = 0.0f;
    float *probs = (float*)malloc(top_k * sizeof(float));
    for (int i = 0; i < top_k; i++) {
        probs[i] = expf(scaled[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum;
    }

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int chosen = indices[0];
    for (int i = 0; i < top_k; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            chosen = indices[i];
            break;
        }
    }

    free(scaled);
    free(indices);
    free(probs);

    return chosen;
}

/* Simple token-to-text (demo placeholder) */
static const char* token_to_text(int token_id) {
    static char buf[32];
    /* In real implementation, look up in vocab */
    snprintf(buf, sizeof(buf), "[%d]", token_id);
    return buf;
}

static void generate_stream(CKModel *model, const char *prompt, bool use_color) {
    int tokens[1024];
    int num_tokens = simple_tokenize(prompt, tokens, 1024);

    if (num_tokens == 0) {
        printf("(empty prompt)\n");
        return;
    }

    if (use_color) {
        printf(ANSI_GREEN);
    }

    double start_time = get_time_ms();
    int generated = 0;

    while (generated < model->max_tokens && !g_interrupted) {
        run_forward_pass(model, tokens, num_tokens);
        int next_token = sample_token(model);

        /* Check for EOS */
        if (next_token == 0) break;  /* Assume 0 is EOS for demo */

        /* Append token */
        if (num_tokens < 1024) {
            tokens[num_tokens++] = next_token;
        }

        /* Stream output */
        const char *text = token_to_text(next_token);
        printf("%s", text);
        fflush(stdout);

        generated++;
        model->tokens_generated++;
    }

    double elapsed = get_time_ms() - start_time;
    model->total_gen_time_ms += elapsed;

    if (use_color) {
        printf(ANSI_RESET);
    }
    printf("\n");

    if (g_interrupted) {
        printf("\n(interrupted)\n");
        g_interrupted = 0;
    }

    /* Print stats */
    printf(ANSI_YELLOW "[%d tokens, %.1f ms, %.1f tok/s]" ANSI_RESET "\n",
           generated, elapsed, generated * 1000.0 / elapsed);
}

static void interactive_loop(CKModel *model, bool use_color) {
    char input[4096];

    printf("Type /help for commands, or enter your prompt.\n\n");

    while (!g_interrupted) {
        if (use_color) {
            printf(ANSI_CYAN "You: " ANSI_RESET);
        } else {
            printf("You: ");
        }
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) {
            printf("\n");
            break;
        }

        /* Remove trailing newline */
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') {
            input[len-1] = '\0';
            len--;
        }

        if (len == 0) continue;

        /* Handle commands */
        if (input[0] == '/') {
            float fval;
            int ival;

            if (strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
                printf("Goodbye!\n");
                break;
            } else if (strcmp(input, "/help") == 0) {
                printf("\nCommands:\n");
                printf("  /help     - Show this help\n");
                printf("  /clear    - Clear screen\n");
                printf("  /exit     - Exit\n");
                printf("  /stats    - Show generation statistics\n");
                printf("  /temp N   - Set temperature (current: %.2f)\n", model->temperature);
                printf("  /top_k N  - Set top-k (current: %d)\n", model->top_k);
                printf("  /top_p N  - Set top-p (current: %.2f)\n", model->top_p);
                printf("\n");
            } else if (strcmp(input, "/clear") == 0) {
                printf("\033[2J\033[H");  /* ANSI clear screen */
                print_banner();
            } else if (strcmp(input, "/stats") == 0) {
                printf("\nGeneration Statistics:\n");
                printf("  Total tokens generated: %d\n", model->tokens_generated);
                printf("  Total generation time:  %.1f ms\n", model->total_gen_time_ms);
                if (model->tokens_generated > 0) {
                    printf("  Average speed:          %.1f tok/s\n",
                           model->tokens_generated * 1000.0 / model->total_gen_time_ms);
                }
                printf("  Context window:         %d\n", model->context_window);
                printf("  Vocabulary size:        %d\n", model->vocab_size);
                printf("\n");
            } else if (sscanf(input, "/temp %f", &fval) == 1) {
                model->temperature = fval;
                printf("Temperature set to %.2f\n", fval);
            } else if (sscanf(input, "/top_k %d", &ival) == 1) {
                model->top_k = ival;
                printf("Top-k set to %d\n", ival);
            } else if (sscanf(input, "/top_p %f", &fval) == 1) {
                model->top_p = fval;
                printf("Top-p set to %.2f\n", fval);
            } else {
                printf("Unknown command: %s\n", input);
            }
            continue;
        }

        /* Generate response */
        if (use_color) {
            printf(ANSI_GREEN "Assistant: " ANSI_RESET);
        } else {
            printf("Assistant: ");
        }

        generate_stream(model, input, use_color);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int context = 512;
    int max_tokens = 128;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    bool interactive = true;
    bool show_banner = true;
    bool use_color = true;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            i++;  /* Skip config path - will be used when loading model */
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
            interactive = false;
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            context = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--interactive") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--no-banner") == 0) {
            show_banner = false;
        } else if (strcmp(argv[i], "--color") == 0) {
            use_color = true;
        } else if (strcmp(argv[i], "--no-color") == 0) {
            use_color = false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Set up signal handler */
    signal(SIGINT, signal_handler);

    /* Seed RNG */
    srand(time(NULL));

    if (show_banner) {
        print_banner();
    }

    /* Initialize model (placeholder) */
    CKModel model = {0};
    model.num_layers = 12;
    model.embed_dim = 768;
    model.num_heads = 12;
    model.num_kv_heads = 12;
    model.head_dim = 64;
    model.intermediate_size = 3072;
    model.vocab_size = 32000;
    model.context_window = context;
    model.rms_norm_eps = 1e-5f;
    model.rope_theta = 10000.0f;
    model.temperature = temperature;
    model.top_k = top_k;
    model.top_p = top_p;
    model.max_tokens = max_tokens;

    printf("Model configuration:\n");
    printf("  Layers:       %d\n", model.num_layers);
    printf("  Hidden size:  %d\n", model.embed_dim);
    printf("  Heads:        %d\n", model.num_heads);
    printf("  Context:      %d\n", model.context_window);
    printf("  Vocab size:   %d\n", model.vocab_size);
    printf("\n");

    if (model_path) {
        printf("Loading model from: %s\n", model_path);
        /* TODO: Actually load the model weights */
    } else {
        printf(ANSI_YELLOW "Warning: No model loaded, using random weights for demo.\n" ANSI_RESET);
        printf("Use --model PATH to load actual weights.\n\n");
    }

    if (interactive) {
        interactive_loop(&model, use_color);
    } else if (prompt) {
        printf("Prompt: %s\n", prompt);
        printf("Response: ");
        generate_stream(&model, prompt, use_color);
    }

    /* Cleanup */
    if (model.logits) free(model.logits);

    return 0;
}
