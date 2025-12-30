/*
 * Debug CPU flag detection - run this on the Xeon Gold server
 * Compile: gcc -o debug_cpu_flags tools/debug_cpu_flags.c
 * Run: ./debug_cpu_flags
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

static void trim_string(char *str) {
    if (!str) return;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) *end-- = '\0';
    char *start = str;
    while (*start && isspace(*start)) start++;
    if (start != str) memmove(str, start, strlen(start) + 1);
}

static int has_cpu_flag(const char *flags, const char *flag) {
    if (!flags || !flag) return 0;
    size_t flag_len = strlen(flag);
    const char *p = flags;
    while ((p = strstr(p, flag)) != NULL) {
        int start_ok = (p == flags) || (*(p - 1) == ' ');
        char after = *(p + flag_len);
        int end_ok = (after == '\0') || (after == ' ') || (after == '\n');
        if (start_ok && end_ok) return 1;
        p++;
    }
    return 0;
}

int main() {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        printf("ERROR: Cannot open /proc/cpuinfo\n");
        return 1;
    }

    char line[8192];  // Larger buffer for long flag lines
    char flags_value[8192] = "";
    char model_name[256] = "";
    int found_flags = 0;

    printf("=== Reading /proc/cpuinfo ===\n\n");

    while (fgets(line, sizeof(line), f)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;

        char key[64];
        size_t key_len = colon - line;
        if (key_len >= sizeof(key)) key_len = sizeof(key) - 1;
        strncpy(key, line, key_len);
        key[key_len] = '\0';
        trim_string(key);

        char *value = colon + 1;
        while (*value == ' ' || *value == '\t') value++;

        if (strcmp(key, "model name") == 0 && model_name[0] == '\0') {
            strncpy(model_name, value, sizeof(model_name) - 1);
            trim_string(model_name);
        }

        if (strcmp(key, "flags") == 0 && !found_flags) {
            strncpy(flags_value, value, sizeof(flags_value) - 1);
            trim_string(flags_value);
            found_flags = 1;

            printf("Raw flags line length: %zu\n", strlen(line));
            printf("Key extracted: '%s'\n", key);
            printf("Value length after trim: %zu\n\n", strlen(flags_value));
        }
    }
    fclose(f);

    printf("CPU Model: %s\n\n", model_name);

    if (!found_flags) {
        printf("ERROR: No 'flags' line found in /proc/cpuinfo!\n");
        return 1;
    }

    // Print first 500 chars of flags
    printf("First 500 chars of flags:\n%.500s\n\n", flags_value);

    // Check for AVX-512 related strings in raw flags
    printf("=== Raw substring search (strstr) ===\n");
    const char *avx512_indicators[] = {
        "avx512", "avx512f", "avx512bw", "avx512vl", "avx512_bf16",
        "amx", "amx_tile", "amx_bf16", "amx_int8"
    };
    for (int i = 0; i < sizeof(avx512_indicators)/sizeof(avx512_indicators[0]); i++) {
        char *found = strstr(flags_value, avx512_indicators[i]);
        if (found) {
            // Show context around the match
            int offset = found - flags_value;
            int start = offset > 20 ? offset - 20 : 0;
            int len = 50;
            printf("  %-15s: FOUND at offset %d, context: '...%.50s...'\n",
                   avx512_indicators[i], offset, flags_value + start);
        } else {
            printf("  %-15s: NOT FOUND\n", avx512_indicators[i]);
        }
    }

    printf("\n=== Word-boundary detection (has_cpu_flag) ===\n");
    const char *test_flags[] = {
        "sse4_2", "avx", "avx2", "fma",
        "avx512f", "avx512bw", "avx512vl", "avx512cd", "avx512dq",
        "avx512_bf16", "avx512_vnni",
        "amx_tile", "amx_int8", "amx_bf16"
    };

    for (int i = 0; i < sizeof(test_flags)/sizeof(test_flags[0]); i++) {
        int found = has_cpu_flag(flags_value, test_flags[i]);
        printf("  %-15s: %s\n", test_flags[i], found ? "YES" : "no");
    }

    // Determine SIMD level
    printf("\n=== Detected SIMD Level ===\n");
    if (has_cpu_flag(flags_value, "avx512f")) {
        printf("AVX-512");
        if (has_cpu_flag(flags_value, "avx512_bf16")) printf(" (BF16");
        else printf(" (");
        if (has_cpu_flag(flags_value, "avx512bw")) printf("BW, ");
        if (has_cpu_flag(flags_value, "avx512vl")) printf("VL, ");
        printf(")\n");
    } else if (has_cpu_flag(flags_value, "avx2")) {
        printf("AVX2\n");
    } else if (has_cpu_flag(flags_value, "avx")) {
        printf("AVX\n");
    } else {
        printf("SSE only\n");
    }

    if (has_cpu_flag(flags_value, "amx_tile") ||
        has_cpu_flag(flags_value, "amx_bf16") ||
        has_cpu_flag(flags_value, "amx_int8")) {
        printf("AMX detected!\n");
    }

    return 0;
}
