#include "ckernel_ir.h"
#include "ckernel_codegen.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage:\n"
                "  %s /path/to/config.json [--emit out.c]  # parse config, dump + codegen\n"
                "  %s --ir /path/to/ir.json [--emit out.c] # parse IR JSON, dump + codegen\n",
                argv[0], argv[0]);
        return 1;
    }

    CKIRGraph graph = {0};
    CKIRGraph bwd   = {0};

    const char *emit_path = NULL;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--emit") == 0) {
            emit_path = argv[i + 1];
            break;
        }
    }

    if (strcmp(argv[1], "--ir") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Missing IR JSON path after --ir\n");
            return 1;
        }
        const char *ir_path = argv[2];
        if (ck_ir_parse_json(ir_path, &graph) != 0) {
            fprintf(stderr, "Failed to parse IR JSON: %s\n", ir_path);
            return 1;
        }
        if (ck_build_decoder_backward_ir(&graph, &bwd) != 0) {
            fprintf(stderr, "Failed to build backward IR from IR JSON\n");
            ck_ir_free(&graph);
            return 1;
        }
    } else {
        const char *config_path = argv[1];
        CKModelConfig cfg;
        if (ck_model_config_from_hf_json(config_path, &cfg) != 0) {
            fprintf(stderr, "Failed to parse config.json: %s\n", config_path);
            return 1;
        }

        if (ck_build_decoder_ir(&cfg, &graph) != 0) {
            fprintf(stderr, "Failed to build decoder IR\n");
            return 1;
        }

        if (ck_build_decoder_backward_ir(&graph, &bwd) != 0) {
            fprintf(stderr, "Failed to build backward IR\n");
            ck_ir_free(&graph);
            return 1;
        }
    }

    printf("=== Forward IR ===\n");
    ck_ir_dump(&graph, stdout);
    printf("\n=== Backward IR (skeleton) ===\n");
    ck_ir_dump(&bwd, stdout);

    printf("\n=== Generated C Skeleton ===\n");
    ck_codegen_c_skeleton(&graph, &bwd, stdout);

    if (emit_path) {
        if (ck_codegen_emit_runtime(&graph, emit_path) == 0) {
            fprintf(stderr, "\n[ck_ir_demo] runtime written to %s\n", emit_path);
        } else {
            fprintf(stderr, "\n[ck_ir_demo] failed to write runtime to %s\n", emit_path);
        }
    }

    // If we came from config.json, also emit a JSON IR map for tooling.
    if (strcmp(argv[1], "--ir") != 0) {
        if (ck_ir_serialize_json(&graph, "build/ir.json") == 0) {
            fprintf(stderr, "\n[ck_ir_demo] JSON IR written to build/ir.json\n");
        } else {
            fprintf(stderr, "\n[ck_ir_demo] Failed to write JSON IR to build/ir.json\n");
        }
    }

    ck_ir_free(&graph);
    ck_ir_free(&bwd);
    return 0;
}
