#include "ckernel_ir.h"
#include "ckernel_codegen.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s /path/to/config.json\n", argv[0]);
        return 1;
    }

    const char *config_path = argv[1];
    CKModelConfig cfg;
    if (ck_model_config_from_hf_json(config_path, &cfg) != 0) {
        fprintf(stderr, "Failed to parse config.json: %s\n", config_path);
        return 1;
    }

    CKIRGraph graph = {0};
    if (ck_build_decoder_ir(&cfg, &graph) != 0) {
        fprintf(stderr, "Failed to build decoder IR\n");
        return 1;
    }

    CKIRGraph bwd = {0};
    if (ck_build_decoder_backward_ir(&graph, &bwd) == 0) {
        printf("=== Forward IR ===\n");
        ck_ir_dump(&graph, stdout);
        printf("\n=== Backward IR (skeleton) ===\n");
        ck_ir_dump(&bwd, stdout);

        printf("\n=== Generated C Skeleton ===\n");
        ck_codegen_c_skeleton(&graph, &bwd, stdout);

        // Also emit a JSON IR map into build/ir.json for tooling.
        if (ck_ir_serialize_json(&graph, "build/ir.json") == 0) {
            fprintf(stderr, "\n[ck_ir_demo] JSON IR written to build/ir.json\n");
        } else {
            fprintf(stderr, "\n[ck_ir_demo] Failed to write JSON IR to build/ir.json\n");
        }

        ck_ir_free(&bwd);
    } else {
        fprintf(stderr, "Warning: failed to build backward IR\n");
    }

    ck_ir_free(&graph);
    return 0;
}
