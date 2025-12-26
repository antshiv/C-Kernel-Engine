/*
 * ck.c - C-Kernel-Engine Main Orchestrator
 *
 * Usage:
 *   ck run https://huggingface.co/HuggingFaceTB/SmolLM-135M
 *   ck run HuggingFaceTB/SmolLM-135M
 *   ck run ./path/to/config.json
 *   ck run SmolLM-135M --server --port 8080
 *   ck list                    # list cached models
 *   ck remove <model>          # remove cached model
 *
 * By Anthony Shivakumar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <stdbool.h>

#define CK_VERSION "0.1.0"
#define CK_CACHE_DIR ".cache/ck-engine/models"
#define MAX_PATH 4096
#define MAX_CMD 8192

/* ANSI Colors */
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_ORANGE  "\033[38;5;214m"
#define C_GREEN   "\033[38;5;114m"
#define C_BLUE    "\033[38;5;75m"
#define C_RED     "\033[38;5;203m"
#define C_YELLOW  "\033[38;5;227m"
#define C_CYAN    "\033[38;5;87m"

/* ═══════════════════════════════════════════════════════════════════════════
 * Data Structures
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    INPUT_LOCAL_PATH,      /* ./path/to/config.json */
    INPUT_HF_URL,          /* https://huggingface.co/org/model */
    INPUT_HF_MODEL_ID,     /* org/model or just model-name */
} InputType;

typedef enum {
    MODE_CHAT,
    MODE_SERVER,
} RunMode;

/* Supported model architectures */
typedef enum {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA,           /* LlamaForCausalLM - LLaMA, SmolLM, Mistral, etc. */
    ARCH_QWEN2,           /* Qwen2ForCausalLM */
    ARCH_GPT2,            /* GPT2LMHeadModel */
    ARCH_GPTNEOX,         /* GPTNeoXForCausalLM */
    ARCH_PHI,             /* PhiForCausalLM */
    ARCH_GEMMA,           /* GemmaForCausalLM */
    ARCH_UNSUPPORTED,
} ModelArch;

typedef struct {
    char model_input[MAX_PATH];    /* Original input */
    char model_id[512];            /* org/model format */
    char org[256];                 /* Organization/user */
    char name[256];                /* Model name */
    char cache_dir[MAX_PATH];      /* ~/.cache/ck-engine/models/org--model */
    char config_path[MAX_PATH];    /* Path to config.json */
    char weights_path[MAX_PATH];   /* Path to weights.bump */
    char tokenizer_path[MAX_PATH]; /* Path to tokenizer.json */
    InputType input_type;
    RunMode mode;
    int port;
    float temperature;
    int max_tokens;
    bool verbose;
    bool force_download;
    bool force_convert;
    bool debug;              /* Run pre-flight checks and show diagnostics */
    char prompt[4096];       /* Optional prompt for non-interactive mode */

    /* Model info (parsed from config.json) */
    ModelArch arch;
    char arch_name[64];            /* e.g., "LlamaForCausalLM" */
    char model_type[64];           /* e.g., "llama" */
    int hidden_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int intermediate_size;
    int vocab_size;
    long param_count;              /* Estimated parameter count */
} CKConfig;

/* Architecture name mappings */
static const struct {
    const char *hf_name;
    ModelArch arch;
    bool supported;
} ARCH_MAP[] = {
    {"LlamaForCausalLM",     ARCH_LLAMA,    true},
    {"MistralForCausalLM",   ARCH_LLAMA,    true},   /* Mistral uses Llama arch */
    {"Qwen2ForCausalLM",     ARCH_QWEN2,    true},
    {"GPT2LMHeadModel",      ARCH_GPT2,     false},  /* TODO */
    {"GPTNeoXForCausalLM",   ARCH_GPTNEOX,  false},  /* TODO */
    {"PhiForCausalLM",       ARCH_PHI,      false},  /* TODO */
    {"GemmaForCausalLM",     ARCH_GEMMA,    false},  /* TODO */
    {NULL, ARCH_UNKNOWN, false}
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Utility Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_banner(void) {
    printf("\n");
    printf(C_ORANGE "  ╔═══════════════════════════════════════════════════════════╗\n" C_RESET);
    printf(C_ORANGE "  ║" C_RESET C_BOLD "  C-Kernel-Engine" C_RESET C_DIM " v%-41s" C_RESET C_ORANGE "║\n" C_RESET, CK_VERSION);
    printf(C_ORANGE "  ║" C_RESET C_DIM "  %-57s" C_RESET C_ORANGE "║\n" C_RESET, "High-Performance ML Inference");
    printf(C_ORANGE "  ║" C_RESET C_DIM "  %-57s" C_RESET C_ORANGE "║\n" C_RESET, "By Anthony Shivakumar");
    printf(C_ORANGE "  ╚═══════════════════════════════════════════════════════════╝\n" C_RESET);
    printf("\n");
}

static void print_usage(void) {
    print_banner();
    printf(C_BOLD "USAGE:" C_RESET "\n");
    printf("  ck run <model>           Run model in chat mode\n");
    printf("  ck run <model> --server  Run model as HTTP server\n");
    printf("  ck list                  List cached models\n");
    printf("  ck remove <model>        Remove cached model\n");
    printf("  ck info <model>          Show model info\n");
    printf("\n");
    printf(C_BOLD "MODEL FORMATS:" C_RESET "\n");
    printf("  " C_CYAN "https://huggingface.co/org/model" C_RESET "  HuggingFace URL\n");
    printf("  " C_CYAN "org/model" C_RESET "                         HuggingFace model ID\n");
    printf("  " C_CYAN "./path/to/config.json" C_RESET "             Local config file\n");
    printf("\n");
    printf(C_BOLD "OPTIONS:" C_RESET "\n");
    printf("  --server          Run as HTTP server (default: chat mode)\n");
    printf("  --port <port>     Server port (default: 8080)\n");
    printf("  --temp <float>    Temperature (default: 0.7)\n");
    printf("  --max-tokens <n>  Max tokens to generate (default: 512)\n");
    printf("  --prompt <text>   Run single prompt (non-interactive mode)\n");
    printf("  --force-download  Re-download model files\n");
    printf("  --force-convert   Re-convert weights\n");
    printf("  --verbose         Verbose output\n");
    printf("  --debug           Run pre-flight diagnostics before starting\n");
    printf("\n");
    printf(C_BOLD "EXAMPLES:" C_RESET "\n");
    printf("  ck run HuggingFaceTB/SmolLM-135M\n");
    printf("  ck run https://huggingface.co/Qwen/Qwen2-0.5B --server --port 3000\n");
    printf("  ck run ./my-model/config.json --temp 0.9\n");
    printf("\n");
}

static void log_info(const char *msg) {
    printf(C_BLUE "[INFO]" C_RESET " %s\n", msg);
}

static void log_ok(const char *msg) {
    printf(C_GREEN "[OK]" C_RESET " %s\n", msg);
}

static void log_warn(const char *msg) {
    printf(C_YELLOW "[WARN]" C_RESET " %s\n", msg);
}

static void log_error(const char *msg) {
    fprintf(stderr, C_RED "[ERROR]" C_RESET " %s\n", msg);
}

static void log_step(int step, int total, const char *msg) {
    printf(C_ORANGE "[%d/%d]" C_RESET " %s\n", step, total, msg);
}

static bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int mkdir_p(const char *path) {
    char tmp[MAX_PATH];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int run_cmd(const char *cmd, bool verbose) {
    if (verbose) {
        printf(C_DIM "$ %s" C_RESET "\n", cmd);
    }
    return system(cmd);
}

static char *get_home_dir(void) {
    char *home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");  /* Windows fallback */
    return home;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Input Parsing
 * ═══════════════════════════════════════════════════════════════════════════ */

static InputType detect_input_type(const char *input) {
    /* Check for HuggingFace URL */
    if (strstr(input, "huggingface.co/") != NULL) {
        return INPUT_HF_URL;
    }

    /* Check for local path (starts with ./ or / or contains .json) */
    if (input[0] == '/' || input[0] == '.' || strstr(input, ".json") != NULL) {
        return INPUT_LOCAL_PATH;
    }

    /* Otherwise assume HuggingFace model ID */
    return INPUT_HF_MODEL_ID;
}

static int parse_hf_url(const char *url, char *org, char *name, size_t size) {
    /* Parse: https://huggingface.co/org/model or https://huggingface.co/org/model/... */
    const char *start = strstr(url, "huggingface.co/");
    if (!start) return -1;

    start += strlen("huggingface.co/");

    /* Find org/name */
    const char *slash = strchr(start, '/');
    if (!slash) return -1;

    size_t org_len = slash - start;
    if (org_len >= size) org_len = size - 1;
    strncpy(org, start, org_len);
    org[org_len] = '\0';

    /* Get model name (up to next / or end) */
    start = slash + 1;
    const char *end = strchr(start, '/');
    size_t name_len = end ? (size_t)(end - start) : strlen(start);
    if (name_len >= size) name_len = size - 1;
    strncpy(name, start, name_len);
    name[name_len] = '\0';

    return 0;
}

static int parse_model_id(const char *id, char *org, char *name, size_t size) {
    /* Parse: org/model */
    const char *slash = strchr(id, '/');
    if (!slash) {
        /* No org, just model name - assume default org */
        strncpy(org, "models", size);
        strncpy(name, id, size);
        return 0;
    }

    size_t org_len = slash - id;
    if (org_len >= size) org_len = size - 1;
    strncpy(org, id, org_len);
    org[org_len] = '\0';

    strncpy(name, slash + 1, size);
    name[size - 1] = '\0';

    return 0;
}

static int setup_config(CKConfig *cfg, const char *input) {
    strncpy(cfg->model_input, input, sizeof(cfg->model_input) - 1);
    cfg->input_type = detect_input_type(input);

    char *home = get_home_dir();
    if (!home) {
        log_error("Cannot determine home directory");
        return -1;
    }

    switch (cfg->input_type) {
        case INPUT_HF_URL:
            if (parse_hf_url(input, cfg->org, cfg->name, sizeof(cfg->org)) != 0) {
                log_error("Failed to parse HuggingFace URL");
                return -1;
            }
            break;

        case INPUT_HF_MODEL_ID:
            if (parse_model_id(input, cfg->org, cfg->name, sizeof(cfg->org)) != 0) {
                log_error("Failed to parse model ID");
                return -1;
            }
            break;

        case INPUT_LOCAL_PATH:
            /* For local paths, extract model name from directory */
            strncpy(cfg->config_path, input, sizeof(cfg->config_path) - 1);
            if (!file_exists(cfg->config_path)) {
                log_error("Config file not found");
                return -1;
            }
            /* Use parent directory name as model name */
            char *dir = strdup(input);
            char *base = strrchr(dir, '/');
            if (base && base != dir) {
                *base = '\0';
                char *name_start = strrchr(dir, '/');
                strncpy(cfg->name, name_start ? name_start + 1 : dir, sizeof(cfg->name) - 1);
            } else {
                strncpy(cfg->name, "local-model", sizeof(cfg->name) - 1);
            }
            strncpy(cfg->org, "local", sizeof(cfg->org) - 1);
            free(dir);

            /* Set cache dir to same directory as config */
            char *config_dir = strdup(input);
            char *last_slash = strrchr(config_dir, '/');
            if (last_slash) *last_slash = '\0';
            strncpy(cfg->cache_dir, config_dir, sizeof(cfg->cache_dir) - 1);
            free(config_dir);
            return 0;
    }

    /* Build model ID and cache directory */
    snprintf(cfg->model_id, sizeof(cfg->model_id), "%s/%s", cfg->org, cfg->name);
    snprintf(cfg->cache_dir, sizeof(cfg->cache_dir), "%s/%s/%s--%s",
             home, CK_CACHE_DIR, cfg->org, cfg->name);
    snprintf(cfg->config_path, sizeof(cfg->config_path), "%s/config.json", cfg->cache_dir);
    snprintf(cfg->weights_path, sizeof(cfg->weights_path), "%s/weights.bump", cfg->cache_dir);
    snprintf(cfg->tokenizer_path, sizeof(cfg->tokenizer_path), "%s/tokenizer.json", cfg->cache_dir);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Simple JSON Parser (for ck_download.py output)
 * ═══════════════════════════════════════════════════════════════════════════ */

static char *json_get_string(const char *json, const char *key, char *out, size_t out_size) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return NULL;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;

    if (*pos != '"') return NULL;
    pos++;  /* Skip opening quote */

    size_t i = 0;
    while (*pos && *pos != '"' && i < out_size - 1) {
        out[i++] = *pos++;
    }
    out[i] = '\0';
    return out;
}

static int json_get_int(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return default_val;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;

    return atoi(pos);
}

static long json_get_long(const char *json, const char *key, long default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return default_val;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;

    return atol(pos);
}

static bool json_get_bool(const char *json, const char *key, bool default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return default_val;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;

    if (strncmp(pos, "true", 4) == 0) return true;
    if (strncmp(pos, "false", 5) == 0) return false;
    return default_val;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Model Download & Convert (via Python helper)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int download_and_convert(CKConfig *cfg) {
    if (cfg->input_type == INPUT_LOCAL_PATH) {
        log_info("Using local model, skipping download");
        /* Still need to get model info from config */
        /* TODO: Parse config.json directly in C */
        return 0;
    }

    /* Find the Python helper script */
    const char *script_paths[] = {
        "./scripts/ck_download.py",
        "../scripts/ck_download.py",
        "scripts/ck_download.py",
        NULL
    };

    const char *script = NULL;
    for (int i = 0; script_paths[i]; i++) {
        if (file_exists(script_paths[i])) {
            script = script_paths[i];
            break;
        }
    }

    if (!script) {
        log_error("Download script not found (scripts/ck_download.py)");
        return -1;
    }

    /* Build command */
    char cmd[MAX_CMD];
    char *home = get_home_dir();

    snprintf(cmd, sizeof(cmd),
        "python3 %s --model '%s' --cache-dir '%s/%s' --quiet %s %s 2>&1",
        script,
        cfg->model_input,
        home, CK_CACHE_DIR,
        cfg->force_download ? "--force" : "",
        cfg->verbose ? "" : "");

    if (cfg->verbose) {
        printf(C_DIM "$ %s" C_RESET "\n", cmd);
    }

    /* Run and capture output */
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        log_error("Failed to run download script");
        return -1;
    }

    /* Read JSON output */
    char json_output[16384] = {0};
    size_t total_read = 0;
    char buffer[1024];

    while (fgets(buffer, sizeof(buffer), fp) != NULL && total_read < sizeof(json_output) - 1) {
        size_t len = strlen(buffer);
        if (total_read + len < sizeof(json_output)) {
            strcat(json_output, buffer);
            total_read += len;
        }
    }

    int ret = pclose(fp);
    if (ret != 0 && !strstr(json_output, "\"status\": \"ok\"")) {
        log_error("Download/convert failed");
        if (cfg->verbose) {
            fprintf(stderr, "%s\n", json_output);
        }

        /* Try to extract error message */
        char error_msg[512];
        if (json_get_string(json_output, "error", error_msg, sizeof(error_msg))) {
            log_error(error_msg);
        }
        return -1;
    }

    /* Parse JSON output */
    char status[64];
    if (!json_get_string(json_output, "status", status, sizeof(status)) ||
        strcmp(status, "ok") != 0) {

        char error_msg[512];
        if (json_get_string(json_output, "error", error_msg, sizeof(error_msg))) {
            log_error(error_msg);
        } else {
            log_error("Unknown error from download script");
        }
        return -1;
    }

    /* Extract model info */
    json_get_string(json_output, "model_id", cfg->model_id, sizeof(cfg->model_id));
    json_get_string(json_output, "cache_dir", cfg->cache_dir, sizeof(cfg->cache_dir));
    json_get_string(json_output, "config_path", cfg->config_path, sizeof(cfg->config_path));
    json_get_string(json_output, "tokenizer_path", cfg->tokenizer_path, sizeof(cfg->tokenizer_path));
    json_get_string(json_output, "weights_path", cfg->weights_path, sizeof(cfg->weights_path));

    /* Extract architecture info */
    json_get_string(json_output, "architecture", cfg->arch_name, sizeof(cfg->arch_name));
    json_get_string(json_output, "model_type", cfg->model_type, sizeof(cfg->model_type));

    bool supported = json_get_bool(json_output, "supported", false);
    cfg->hidden_size = json_get_int(json_output, "hidden_size", 0);
    cfg->num_layers = json_get_int(json_output, "num_layers", 0);
    cfg->num_heads = json_get_int(json_output, "num_heads", 0);
    cfg->num_kv_heads = json_get_int(json_output, "num_kv_heads", 0);
    cfg->intermediate_size = json_get_int(json_output, "intermediate_size", 0);
    cfg->vocab_size = json_get_int(json_output, "vocab_size", 0);
    cfg->param_count = json_get_long(json_output, "param_count", 0);

    /* Determine architecture enum */
    cfg->arch = ARCH_UNKNOWN;
    for (int i = 0; ARCH_MAP[i].hf_name != NULL; i++) {
        if (strcmp(cfg->arch_name, ARCH_MAP[i].hf_name) == 0) {
            cfg->arch = ARCH_MAP[i].arch;
            break;
        }
    }

    /* Print model info */
    char param_str[32];
    if (cfg->param_count >= 1000000000) {
        snprintf(param_str, sizeof(param_str), "%.1fB", cfg->param_count / 1e9);
    } else if (cfg->param_count >= 1000000) {
        snprintf(param_str, sizeof(param_str), "%.0fM", cfg->param_count / 1e6);
    } else {
        snprintf(param_str, sizeof(param_str), "%ld", cfg->param_count);
    }

    printf("\n");
    printf(C_BOLD "  Model:" C_RESET " %s\n", cfg->model_id);
    printf(C_BOLD "  Arch:" C_RESET "  %s %s\n", cfg->arch_name,
           supported ? C_GREEN "(supported)" C_RESET : C_RED "(not supported)" C_RESET);
    printf(C_BOLD "  Size:" C_RESET "  %s params | %d layers | %d heads\n",
           param_str, cfg->num_layers, cfg->num_heads);
    printf("\n");

    if (!supported) {
        log_error("This model architecture is not yet supported by C-Kernel-Engine");
        printf(C_DIM "  Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM\n" C_RESET);
        return -1;
    }

    log_ok("Model ready");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Code Generation & Compilation
 * ═══════════════════════════════════════════════════════════════════════════ */

static int codegen_and_compile(CKConfig *cfg) {
    char model_c[MAX_PATH];
    char model_so[MAX_PATH];

    snprintf(model_c, sizeof(model_c), "%s/model.c", cfg->cache_dir);
    snprintf(model_so, sizeof(model_so), "%s/libmodel.so", cfg->cache_dir);

    /* Check if already compiled (unless force_convert is set) */
    if (file_exists(model_so) && !cfg->force_convert) {
        /* TODO: Check if config changed */
        log_ok("Model already compiled");
        return 0;
    }

    if (cfg->force_convert && file_exists(model_so)) {
        log_info("Force recompiling model...");
    }

    log_info("Generating model code...");

    /* Find ck_ir_demo binary */
    const char *ir_paths[] = {
        "./build/ck_ir_demo",
        "../build/ck_ir_demo",
        "build/ck_ir_demo",
        NULL
    };

    const char *ir_bin = NULL;
    for (int i = 0; ir_paths[i]; i++) {
        if (file_exists(ir_paths[i])) {
            ir_bin = ir_paths[i];
            break;
        }
    }

    if (!ir_bin) {
        log_warn("ck_ir_demo not found, run 'make build/ck_ir_demo' first");
        log_info("Skipping codegen for now...");
        return 0;  /* Non-fatal for now */
    }

    char cmd[MAX_CMD];
    /* Generate library-mode code (no main, exports API functions) */
    snprintf(cmd, sizeof(cmd), "%s %s --emit %s --emit-lib", ir_bin, cfg->config_path, model_c);

    int ret = run_cmd(cmd, cfg->verbose);
    if (ret != 0) {
        log_warn("Code generation failed (non-fatal)");
        return 0;
    }

    log_info("Compiling model...");

    /* Find project root (where include/ and build/ are) */
    const char *root_paths[] = {
        ".",
        "..",
        "../..",
        NULL
    };

    const char *project_root = NULL;
    char include_check[MAX_PATH];

    for (int i = 0; root_paths[i]; i++) {
        snprintf(include_check, sizeof(include_check), "%s/include/ckernel_engine.h", root_paths[i]);
        if (file_exists(include_check)) {
            project_root = root_paths[i];
            break;
        }
    }

    if (!project_root) {
        log_warn("Project root not found, skipping compilation");
        log_info("Run from project directory or set up include paths");
        return 0;
    }

    /* Check if library exists */
    char lib_path[MAX_PATH];
    snprintf(lib_path, sizeof(lib_path), "%s/build/libckernel_engine.so", project_root);

    if (!file_exists(lib_path)) {
        log_warn("libckernel_engine.so not found, building it...");
        snprintf(cmd, sizeof(cmd), "make -C %s build/libckernel_engine.so", project_root);
        run_cmd(cmd, cfg->verbose);
    }

    /* Read the .kernels manifest if it exists (one file per line) */
    char kernels_file[MAX_PATH];
    char kernel_sources[MAX_CMD] = "";
    snprintf(kernels_file, sizeof(kernels_file), "%s.kernels", model_c);

    if (file_exists(kernels_file)) {
        FILE *f = fopen(kernels_file, "r");
        if (f) {
            char line[512];
            size_t pos = 0;
            while (fgets(line, sizeof(line), f) && pos < sizeof(kernel_sources) - 512) {
                /* Remove trailing newline */
                size_t len = strlen(line);
                if (len > 0 && line[len-1] == '\n') line[--len] = '\0';
                if (len > 0) {
                    /* Add space separator and the source file */
                    if (pos > 0) {
                        kernel_sources[pos++] = ' ';
                    }
                    /* Prepend project_root if path is relative */
                    if (line[0] != '/') {
                        pos += snprintf(kernel_sources + pos, sizeof(kernel_sources) - pos,
                                        "%s/%s", project_root, line);
                    } else {
                        pos += snprintf(kernel_sources + pos, sizeof(kernel_sources) - pos,
                                        "%s", line);
                    }
                }
            }
            fclose(f);
        }
    }

    /* Compile to self-contained shared library (includes kernel sources directly) */
    snprintf(cmd, sizeof(cmd),
        "cc -O3 -fPIC -shared -I%s/include -o %s %s %s -lm 2>&1 || "
        "gcc -O3 -fPIC -shared -I%s/include -o %s %s %s -lm 2>&1",
        project_root, model_so, model_c, kernel_sources,
        project_root, model_so, model_c, kernel_sources);

    ret = run_cmd(cmd, cfg->verbose);
    if (ret != 0) {
        log_warn("Compilation failed (non-fatal)");
        if (cfg->verbose) {
            printf(C_DIM "  Hint: Run 'make' in project root to build kernel library\n" C_RESET);
        }
        return 0;
    }

    log_ok("Model compiled successfully");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Run Chat/Server (using dlopen to load compiled model)
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <dlfcn.h>
#include <math.h>

/* Function pointer types for model API */
typedef int (*ck_model_init_fn)(const char *weights_path);
typedef int (*ck_model_embed_tokens_fn)(const int32_t *tokens, int num_tokens);
typedef int (*ck_model_forward_fn)(float *logits_out);
typedef float* (*ck_model_get_logits_fn)(void);
typedef void (*ck_model_free_fn)(void);
typedef int (*ck_model_get_vocab_size_fn)(void);
typedef int (*ck_model_get_context_window_fn)(void);

/* Simple top-k sampling */
static int sample_top_k(const float *logits, int vocab_size, int k, float temp) {
    if (k <= 0 || k > vocab_size) k = vocab_size;

    /* Find top-k indices */
    int *top_idx = (int *)malloc(k * sizeof(int));
    float *top_val = (float *)malloc(k * sizeof(float));

    for (int i = 0; i < k; i++) {
        top_idx[i] = -1;
        top_val[i] = -1e30f;
    }

    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (v > top_val[k-1]) {
            /* Insert into sorted list */
            int j = k - 1;
            while (j > 0 && v > top_val[j-1]) {
                top_val[j] = top_val[j-1];
                top_idx[j] = top_idx[j-1];
                j--;
            }
            top_val[j] = v;
            top_idx[j] = i;
        }
    }

    /* Apply temperature and softmax */
    float max_v = top_val[0];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        top_val[i] = expf((top_val[i] - max_v) / temp);
        sum += top_val[i];
    }
    for (int i = 0; i < k; i++) {
        top_val[i] /= sum;
    }

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int result = top_idx[0];
    for (int i = 0; i < k; i++) {
        cumsum += top_val[i];
        if (r <= cumsum) {
            result = top_idx[i];
            break;
        }
    }

    free(top_idx);
    free(top_val);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pre-flight Diagnostics (--debug mode)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int run_preflight_checks(CKConfig *cfg) {
    printf("\n");
    printf(C_YELLOW "╔═══════════════════════════════════════════════════════════╗\n" C_RESET);
    printf(C_YELLOW "║" C_RESET C_BOLD "  PRE-FLIGHT DIAGNOSTICS                                  " C_RESET C_YELLOW "║\n" C_RESET);
    printf(C_YELLOW "╚═══════════════════════════════════════════════════════════╝\n" C_RESET);
    printf("\n");

    int errors = 0;
    int warnings = 0;

    /* Check 1: Required files */
    printf(C_BOLD "  [1/4] Checking required files...\n" C_RESET);

    char model_so[MAX_PATH];
    snprintf(model_so, sizeof(model_so), "%s/libmodel.so", cfg->cache_dir);

    struct {
        const char *name;
        const char *path;
        bool required;
    } files[] = {
        {"Config", cfg->config_path, true},
        {"Weights", cfg->weights_path, true},
        {"Tokenizer", cfg->tokenizer_path, false},
        {"Model Library", model_so, true},
    };

    for (int i = 0; i < 4; i++) {
        bool exists = file_exists(files[i].path);
        if (exists) {
            printf("        " C_GREEN "✓" C_RESET " %s: %s\n", files[i].name, files[i].path);
        } else if (files[i].required) {
            printf("        " C_RED "✗" C_RESET " %s: %s " C_RED "(MISSING)" C_RESET "\n", files[i].name, files[i].path);
            errors++;
        } else {
            printf("        " C_YELLOW "○" C_RESET " %s: %s " C_DIM "(optional)" C_RESET "\n", files[i].name, files[i].path);
            warnings++;
        }
    }

    /* Check 2: Library loading */
    printf(C_BOLD "\n  [2/4] Testing library loading...\n" C_RESET);

    void *lib = dlopen(model_so, RTLD_NOW);
    if (!lib) {
        printf("        " C_RED "✗" C_RESET " dlopen failed: %s\n", dlerror());
        errors++;
    } else {
        printf("        " C_GREEN "✓" C_RESET " Library opened successfully\n");

        /* Check required symbols */
        const char *required_syms[] = {
            "ck_model_init", "ck_model_embed_tokens",
            "ck_model_forward", "ck_model_get_logits", "ck_model_free"
        };
        for (int i = 0; i < 5; i++) {
            void *sym = dlsym(lib, required_syms[i]);
            if (sym) {
                printf("        " C_GREEN "✓" C_RESET " %s\n", required_syms[i]);
            } else {
                printf("        " C_RED "✗" C_RESET " %s " C_RED "(missing symbol)" C_RESET "\n", required_syms[i]);
                errors++;
            }
        }
        dlclose(lib);
    }

    /* Check 3: Weights file sanity */
    printf(C_BOLD "\n  [3/4] Checking weights file...\n" C_RESET);

    FILE *wf = fopen(cfg->weights_path, "rb");
    if (wf) {
        fseek(wf, 0, SEEK_END);
        long size = ftell(wf);
        fclose(wf);
        printf("        " C_GREEN "✓" C_RESET " Size: %.2f MB\n", size / (1024.0 * 1024.0));

        /* Estimate expected size based on param count */
        long expected_bytes = cfg->param_count * 4;  /* fp32 */
        if (size < expected_bytes * 0.5) {
            printf("        " C_YELLOW "⚠" C_RESET " File seems small for %ldM params (expected ~%.0f MB)\n",
                   cfg->param_count / 1000000, expected_bytes / (1024.0 * 1024.0));
            warnings++;
        }
    } else {
        printf("        " C_RED "✗" C_RESET " Cannot read weights file\n");
        errors++;
    }

    /* Check 4: CPU capabilities */
    printf(C_BOLD "\n  [4/4] Checking CPU capabilities...\n" C_RESET);

#ifdef __x86_64__
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo) {
        char line[1024];
        bool avx = false, avx2 = false, avx512 = false;
        while (fgets(line, sizeof(line), cpuinfo)) {
            if (strncmp(line, "flags", 5) == 0) {
                avx = strstr(line, " avx ") != NULL;
                avx2 = strstr(line, " avx2 ") != NULL;
                avx512 = strstr(line, " avx512f ") != NULL;
                break;
            }
        }
        fclose(cpuinfo);
        printf("        AVX:     %s\n", avx ? C_GREEN "✓" C_RESET : C_DIM "✗" C_RESET);
        printf("        AVX2:    %s\n", avx2 ? C_GREEN "✓" C_RESET : C_DIM "✗" C_RESET);
        printf("        AVX-512: %s\n", avx512 ? C_GREEN "✓" C_RESET : C_DIM "✗" C_RESET);
    }
#else
    printf("        " C_DIM "(CPU check only available on x86_64)" C_RESET "\n");
#endif

    /* Summary */
    printf("\n");
    printf(C_DIM "  ─────────────────────────────────────────────────────────\n" C_RESET);
    if (errors > 0) {
        printf("  " C_RED "FAILED" C_RESET ": %d error(s), %d warning(s)\n", errors, warnings);
        printf("\n");
        printf(C_DIM "  Troubleshooting:\n" C_RESET);
        printf(C_DIM "    • Run 'make test' to check kernel functionality\n" C_RESET);
        printf(C_DIM "    • Run 'python scripts/smollm_forward_parity.py' for PyTorch comparison\n" C_RESET);
        printf(C_DIM "    • Try 'ck run <model> --force-convert' to regenerate weights\n" C_RESET);
        printf("\n");
        return -1;
    } else if (warnings > 0) {
        printf("  " C_YELLOW "PASSED with warnings" C_RESET ": %d warning(s)\n", warnings);
    } else {
        printf("  " C_GREEN "ALL CHECKS PASSED" C_RESET "\n");
    }
    printf(C_DIM "  ─────────────────────────────────────────────────────────\n" C_RESET);
    printf("\n");

    return 0;
}

static int run_chat(CKConfig *cfg) {
    /* Run pre-flight checks in debug mode */
    if (cfg->debug) {
        if (run_preflight_checks(cfg) != 0) {
            return -1;
        }
    }

    printf("\n");
    printf(C_ORANGE "  ┌─────────────────────────────────────────────────────────┐\n" C_RESET);
    printf(C_ORANGE "  │" C_RESET C_BOLD " Model: " C_RESET C_CYAN "%s" C_RESET "\n", cfg->model_id);
    printf(C_ORANGE "  │" C_RESET C_DIM " Cache: %s" C_RESET "\n", cfg->cache_dir);
    printf(C_ORANGE "  │" C_RESET C_DIM " Temp: %.2f | Max tokens: %d" C_RESET "\n", cfg->temperature, cfg->max_tokens);
    printf(C_ORANGE "  └─────────────────────────────────────────────────────────┘\n" C_RESET);
    printf("\n");

    /* Use Python chat script with proper tokenizer */
    char cmd[MAX_CMD];

    /* Find the scripts directory - try relative to executable or cwd */
    const char *script_paths[] = {
        "./scripts/ck_chat.py",
        "../scripts/ck_chat.py",
        "scripts/ck_chat.py",
        NULL
    };

    const char *script = NULL;
    for (int i = 0; script_paths[i]; i++) {
        if (file_exists(script_paths[i])) {
            script = script_paths[i];
            break;
        }
    }

    if (!script) {
        log_error("Chat script not found (scripts/ck_chat.py)");
        printf(C_DIM "  Searched: ./scripts/ck_chat.py, ../scripts/ck_chat.py\n" C_RESET);
        printf(C_DIM "  Current directory: %s\n" C_RESET, getcwd(NULL, 0));
        printf(C_DIM "  Please run from the C-Kernel-Engine directory.\n" C_RESET);
        return -1;
    }

    if (cfg->verbose) {
        printf(C_DIM "  Using script: %s\n" C_RESET, script);
    }

    if (cfg->prompt[0] != '\0') {
        /* Non-interactive mode with prompt */
        snprintf(cmd, sizeof(cmd),
            "python3 %s --model-dir '%s' --temperature %.2f --max-tokens %d --prompt '%s'",
            script, cfg->cache_dir, cfg->temperature, cfg->max_tokens, cfg->prompt);
    } else {
        /* Interactive mode */
        snprintf(cmd, sizeof(cmd),
            "python3 %s --model-dir '%s' --temperature %.2f --max-tokens %d",
            script, cfg->cache_dir, cfg->temperature, cfg->max_tokens);
    }

    if (cfg->verbose) {
        printf(C_DIM "  Command: %s\n" C_RESET, cmd);
    }

    int ret = system(cmd);
    if (ret != 0) {
        log_error("Chat script failed");
        printf(C_DIM "  Exit code: %d\n" C_RESET, ret);
        printf(C_DIM "  Try: pip install tokenizers numpy\n" C_RESET);
    }
    return ret;
}

static int run_server(CKConfig *cfg) {
    printf("\n");
    printf(C_ORANGE "  ┌─────────────────────────────────────────────────────────┐\n" C_RESET);
    printf(C_ORANGE "  │" C_RESET C_BOLD " C-Kernel-Engine Server" C_RESET "\n");
    printf(C_ORANGE "  │" C_RESET C_CYAN " Model: %s" C_RESET "\n", cfg->model_id);
    printf(C_ORANGE "  │" C_RESET C_GREEN " http://localhost:%d" C_RESET "\n", cfg->port);
    printf(C_ORANGE "  └─────────────────────────────────────────────────────────┘\n" C_RESET);
    printf("\n");
    printf(C_DIM "  Endpoints:\n" C_RESET);
    printf(C_DIM "    POST /v1/completions      - Text completion\n" C_RESET);
    printf(C_DIM "    POST /v1/chat/completions - Chat completion\n" C_RESET);
    printf(C_DIM "    GET  /v1/models           - List models\n" C_RESET);
    printf(C_DIM "    GET  /health              - Health check\n" C_RESET);
    printf("\n");
    printf(C_DIM "  Press Ctrl+C to stop\n" C_RESET);
    printf("\n");

    /* Find ck_server binary */
    const char *server_paths[] = {
        "./build/ck_server",
        "../build/ck_server",
        "build/ck_server",
        NULL
    };

    const char *server_bin = NULL;
    for (int i = 0; server_paths[i]; i++) {
        if (file_exists(server_paths[i])) {
            server_bin = server_paths[i];
            break;
        }
    }

    if (!server_bin) {
        log_error("Server binary not found. Run 'make ck-server' first");
        return -1;
    }

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
        "%s --config %s --weights %s --tokenizer %s --port %d",
        server_bin, cfg->config_path, cfg->weights_path, cfg->tokenizer_path, cfg->port);

    return run_cmd(cmd, cfg->verbose);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * List/Remove Commands
 * ═══════════════════════════════════════════════════════════════════════════ */

static int list_models(void) {
    char *home = get_home_dir();
    if (!home) {
        log_error("Cannot determine home directory");
        return -1;
    }

    char cache_dir[MAX_PATH];
    snprintf(cache_dir, sizeof(cache_dir), "%s/%s", home, CK_CACHE_DIR);

    if (!dir_exists(cache_dir)) {
        printf(C_DIM "No cached models found.\n" C_RESET);
        return 0;
    }

    printf("\n");
    printf(C_BOLD "Cached Models:" C_RESET "\n");
    printf(C_DIM "─────────────────────────────────────────────────\n" C_RESET);

    DIR *dir = opendir(cache_dir);
    if (!dir) {
        printf(C_DIM "No cached models found.\n" C_RESET);
        return 0;
    }

    struct dirent *entry;
    int count = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char model_dir[MAX_PATH];
        snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);

        if (!dir_exists(model_dir)) continue;

        /* Replace -- with / for display */
        char display_name[512];
        strncpy(display_name, entry->d_name, sizeof(display_name));
        char *sep = strstr(display_name, "--");
        if (sep) *sep = '/', *(sep + 1) = '\0', strcat(display_name, sep + 2);

        /* Check what files exist */
        char config[MAX_PATH], weights[MAX_PATH], tokenizer[MAX_PATH];
        snprintf(config, sizeof(config), "%s/config.json", model_dir);
        snprintf(weights, sizeof(weights), "%s/weights.bump", model_dir);
        snprintf(tokenizer, sizeof(tokenizer), "%s/tokenizer.json", model_dir);

        printf("  " C_CYAN "%s" C_RESET "\n", display_name);
        printf("    ");
        printf(file_exists(config) ? C_GREEN "config " C_RESET : C_RED "config " C_RESET);
        printf(file_exists(weights) ? C_GREEN "weights " C_RESET : C_DIM "weights " C_RESET);
        printf(file_exists(tokenizer) ? C_GREEN "tokenizer" C_RESET : C_DIM "tokenizer" C_RESET);
        printf("\n");

        count++;
    }

    closedir(dir);

    if (count == 0) {
        printf(C_DIM "  No cached models found.\n" C_RESET);
    }

    printf(C_DIM "─────────────────────────────────────────────────\n" C_RESET);
    printf(C_DIM "Cache: %s\n" C_RESET, cache_dir);
    printf("\n");

    return 0;
}

static int remove_model(const char *model) {
    CKConfig cfg = {0};
    if (setup_config(&cfg, model) != 0) {
        return -1;
    }

    if (!dir_exists(cfg.cache_dir)) {
        log_error("Model not found in cache");
        return -1;
    }

    printf("Remove " C_CYAN "%s" C_RESET "? [y/N] ", cfg.model_id);
    fflush(stdout);

    char response[16];
    if (!fgets(response, sizeof(response), stdin)) {
        return -1;
    }

    if (response[0] != 'y' && response[0] != 'Y') {
        printf("Cancelled.\n");
        return 0;
    }

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", cfg.cache_dir);

    if (run_cmd(cmd, false) != 0) {
        log_error("Failed to remove model");
        return -1;
    }

    log_ok("Model removed");
    return 0;
}

static int show_model_info(const char *model) {
    CKConfig cfg = {0};
    if (setup_config(&cfg, model) != 0) {
        return -1;
    }

    printf("\n");
    printf(C_BOLD "Model Information:" C_RESET "\n");
    printf(C_DIM "─────────────────────────────────────────────────\n" C_RESET);
    printf("  " C_BOLD "ID:" C_RESET "        %s\n", cfg.model_id);
    printf("  " C_BOLD "Org:" C_RESET "       %s\n", cfg.org);
    printf("  " C_BOLD "Name:" C_RESET "      %s\n", cfg.name);
    printf("  " C_BOLD "Cache:" C_RESET "     %s\n", cfg.cache_dir);
    printf("  " C_BOLD "Config:" C_RESET "    %s %s\n", cfg.config_path,
           file_exists(cfg.config_path) ? C_GREEN "(exists)" C_RESET : C_RED "(missing)" C_RESET);
    printf("  " C_BOLD "Weights:" C_RESET "   %s %s\n", cfg.weights_path,
           file_exists(cfg.weights_path) ? C_GREEN "(exists)" C_RESET : C_DIM "(not converted)" C_RESET);
    printf("  " C_BOLD "Tokenizer:" C_RESET " %s %s\n", cfg.tokenizer_path,
           file_exists(cfg.tokenizer_path) ? C_GREEN "(exists)" C_RESET : C_RED "(missing)" C_RESET);
    printf(C_DIM "─────────────────────────────────────────────────\n" C_RESET);
    printf("\n");

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    const char *cmd = argv[1];

    /* Handle commands */
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage();
        return 0;
    }

    if (strcmp(cmd, "version") == 0 || strcmp(cmd, "--version") == 0 || strcmp(cmd, "-v") == 0) {
        printf("ck version %s\n", CK_VERSION);
        return 0;
    }

    if (strcmp(cmd, "list") == 0) {
        print_banner();
        return list_models();
    }

    if (strcmp(cmd, "remove") == 0) {
        if (argc < 3) {
            log_error("Usage: ck remove <model>");
            return 1;
        }
        return remove_model(argv[2]);
    }

    if (strcmp(cmd, "info") == 0) {
        if (argc < 3) {
            log_error("Usage: ck info <model>");
            return 1;
        }
        return show_model_info(argv[2]);
    }

    if (strcmp(cmd, "run") == 0) {
        if (argc < 3) {
            log_error("Usage: ck run <model> [options]");
            return 1;
        }

        CKConfig cfg = {0};
        cfg.mode = MODE_CHAT;
        cfg.port = 8080;
        cfg.temperature = 0.7f;
        cfg.max_tokens = 512;
        cfg.verbose = false;
        cfg.force_download = false;
        cfg.force_convert = false;
        cfg.debug = false;

        /* Parse model argument */
        if (setup_config(&cfg, argv[2]) != 0) {
            return 1;
        }

        /* Parse options */
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--server") == 0) {
                cfg.mode = MODE_SERVER;
            } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
                cfg.port = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
                cfg.temperature = atof(argv[++i]);
            } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
                cfg.max_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
                cfg.verbose = true;
            } else if (strcmp(argv[i], "--force-download") == 0) {
                cfg.force_download = true;
            } else if (strcmp(argv[i], "--force-convert") == 0) {
                cfg.force_convert = true;
            } else if (strcmp(argv[i], "--debug") == 0) {
                cfg.debug = true;
                cfg.verbose = true;  /* Debug implies verbose */
            } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
                strncpy(cfg.prompt, argv[++i], sizeof(cfg.prompt) - 1);
                cfg.prompt[sizeof(cfg.prompt) - 1] = '\0';
            }
        }

        print_banner();

        /* Run pipeline */
        int total_steps = 3;
        int step = 1;

        log_step(step++, total_steps, "Downloading model & converting weights (Python)...");
        if (download_and_convert(&cfg) != 0) return 1;

        log_step(step++, total_steps, "Generating and compiling (C)...");
        if (codegen_and_compile(&cfg) != 0) return 1;

        log_step(step++, total_steps, cfg.mode == MODE_SERVER ? "Starting server..." : "Starting chat...");

        if (cfg.mode == MODE_SERVER) {
            return run_server(&cfg);
        } else {
            return run_chat(&cfg);
        }
    }

    /* Unknown command */
    log_error("Unknown command. Run 'ck help' for usage.");
    return 1;
}
