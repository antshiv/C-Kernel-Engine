/*
 * show_config.c - Display system configuration for C-Kernel-Engine
 *
 * Main program for `make show_config` that displays comprehensive
 * hardware topology and recommendations for distributed training.
 */

#define _GNU_SOURCE
#include "system_topology.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════════════════
// ANSI Color Codes
// ═══════════════════════════════════════════════════════════════════════════════

#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define DIM         "\033[2m"

#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"

#define BG_RED      "\033[41m"
#define BG_GREEN    "\033[42m"
#define BG_YELLOW   "\033[43m"

// Check if terminal supports colors
static int use_colors = 1;

#define C(color) (use_colors ? color : "")

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

static const char* format_size(uint64_t size_mb, char *buf, size_t buf_size) {
    if (size_mb >= 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f TB", size_mb / (1024.0 * 1024.0));
    } else if (size_mb >= 1024) {
        snprintf(buf, buf_size, "%.1f GB", size_mb / 1024.0);
    } else {
        snprintf(buf, buf_size, "%lu MB", (unsigned long)size_mb);
    }
    return buf;
}

static const char* format_bandwidth(float bw_gbs, char *buf, size_t buf_size) {
    if (bw_gbs >= 1.0f) {
        snprintf(buf, buf_size, "%.1f GB/s", bw_gbs);
    } else {
        snprintf(buf, buf_size, "%.0f MB/s", bw_gbs * 1024);
    }
    return buf;
}

static void print_header(const char *title) {
    printf("\n%s", C(BOLD));
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("═══════════════════════════════════════════════════════════════════════════════%s\n",
           C(RESET));
}

static void print_section(const char *title) {
    printf("\n%s  %s%s\n", C(CYAN), title, C(RESET));
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
}

static void print_tree_item(int level, int is_last, const char *fmt, ...) {
    for (int i = 0; i < level; i++) {
        printf("  │   ");
    }
    printf("  %s── ", is_last ? "└" : "├");

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

static void print_warning(const char *msg) {
    printf("  %s⚠️  %s%s\n", C(YELLOW), msg, C(RESET));
}

static void print_ok(const char *msg) {
    printf("  %s✓  %s%s\n", C(GREEN), msg, C(RESET));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Print Functions
// ═══════════════════════════════════════════════════════════════════════════════

void topology_print_cpu(const CPUInfo *cpu) {
    print_section("CPU");

    printf("  %s%s%s\n", C(BOLD), cpu->model_name, C(RESET));

    // Build SIMD string with detailed AVX-512 sub-features
    char simd_buf[256] = "";
    if (cpu->has_avx512f) {
        strcat(simd_buf, "AVX-512");
        // Add AVX-512 sub-features in parentheses
        char sub_features[64] = "";
        if (cpu->has_avx512_bf16) strcat(sub_features, "BF16, ");
        if (cpu->has_avx512bw) strcat(sub_features, "BW, ");
        if (cpu->has_avx512vl) strcat(sub_features, "VL, ");
        if (sub_features[0]) {
            // Remove trailing ", "
            sub_features[strlen(sub_features) - 2] = '\0';
            strcat(simd_buf, " (");
            strcat(simd_buf, sub_features);
            strcat(simd_buf, ")");
        }
        strcat(simd_buf, " ");
    } else if (cpu->has_avx2) {
        strcat(simd_buf, "AVX2 ");
    } else if (cpu->has_avx) {
        strcat(simd_buf, "AVX ");
    } else if (cpu->has_sse4_2) {
        strcat(simd_buf, "SSE4.2 ");
    }

    if (cpu->has_vnni) strcat(simd_buf, "VNNI ");

    // AMX details
    if (cpu->has_amx) {
        strcat(simd_buf, "AMX");
        char amx_features[32] = "";
        if (cpu->has_amx_bf16) strcat(amx_features, "BF16,");
        if (cpu->has_amx_int8) strcat(amx_features, "INT8,");
        if (amx_features[0]) {
            amx_features[strlen(amx_features) - 1] = '\0';  // Remove trailing comma
            strcat(simd_buf, "(");
            strcat(simd_buf, amx_features);
            strcat(simd_buf, ") ");
        } else {
            strcat(simd_buf, " ");
        }
    }

    print_tree_item(0, 0, "Sockets: %d", cpu->sockets);
    print_tree_item(0, 0, "Cores: %d physical, %d logical %s",
                    cpu->physical_cores, cpu->logical_cores,
                    cpu->threads_per_core > 1 ? "(HT/SMT enabled)" : "");
    print_tree_item(0, 0, "Frequency: %.0f MHz (max: %.0f MHz)",
                    cpu->base_freq_mhz, cpu->max_freq_mhz);
    print_tree_item(0, 0, "SIMD: %s%s%s",
                    C(cpu->has_avx512f ? GREEN : (cpu->has_avx2 ? GREEN : YELLOW)),
                    simd_buf[0] ? simd_buf : "Basic",
                    C(RESET));
    print_tree_item(0, 1, "PCIe: Gen %d, ~%d lanes from CPU",
                    cpu->pcie_generation, cpu->pcie_lanes_total);

    if (!cpu->has_avx2) {
        print_warning("No AVX2 support - kernel performance will be limited");
    }
}

void topology_print_cache(const CacheTopology *cache, int logical_cores) {
    print_section("CACHE HIERARCHY");

    // Show data source
    printf("  %sSource: /sys/devices/system/cpu/cpu0/cache/%s\n\n",
           C(DIM), C(RESET));

    for (int i = 0; i < cache->num_levels; i++) {
        const CacheInfo *c = &cache->levels[i];
        int is_last = (i == cache->num_levels - 1);

        // Calculate number of instances (how many of this cache exist)
        int instances = 1;
        if (c->shared_by_cores > 0 && logical_cores > 0) {
            instances = logical_cores / c->shared_by_cores;
            if (instances < 1) instances = 1;
        }

        // Calculate total size across all instances
        int total_kb = c->size_kb * instances;

        // Format size nicely (KB or MB)
        char size_str[32];
        if (total_kb >= 1024) {
            snprintf(size_str, sizeof(size_str), "%d MiB", total_kb / 1024);
        } else {
            snprintf(size_str, sizeof(size_str), "%d KiB", total_kb);
        }

        // Format instance info like lscpu
        char instance_str[32] = "";
        if (instances > 1) {
            snprintf(instance_str, sizeof(instance_str), " (%d instances)", instances);
        } else {
            snprintf(instance_str, sizeof(instance_str), " (%d instance)", instances);
        }

        print_tree_item(0, is_last, "L%d%c: %s%s",
                        c->level,
                        c->type[0] == 'D' ? 'd' : (c->type[0] == 'I' ? 'i' : ' '),
                        size_str, instance_str);
    }
}

void topology_print_numa(const NUMATopology *numa, int sockets) {
    print_section("NUMA TOPOLOGY");

    // Show source
    printf("  %sSource: /sys/devices/system/node/%s\n", C(DIM), C(RESET));

    // Single NUMA node - UMA system
    if (numa->num_nodes <= 1) {
        printf("\n  %s✓ Single NUMA node (Uniform Memory Access)%s\n", C(GREEN), C(RESET));
        printf("  %s  All memory is local - no NUMA penalties%s\n", C(DIM), C(RESET));
        printf("\n  %sNote: Sub-NUMA Clustering (SNC) / NUMA-Per-Socket (NPS) not detected.%s\n",
               C(DIM), C(RESET));
        printf("  %s  On Xeon/EPYC, check BIOS settings or run: numactl --hardware%s\n",
               C(DIM), C(RESET));
        return;
    }

    // Detect potential Sub-NUMA Clustering (SNC) or NUMA-Per-Socket (NPS)
    // If num_nodes > sockets, SNC/NPS is likely enabled
    // SNC divides each socket's memory channels into groups, one per sub-NUMA node
    if (sockets > 0 && numa->num_nodes > sockets) {
        int nodes_per_socket = numa->num_nodes / sockets;
        printf("\n  %s⚠ Sub-NUMA detected: %d NUMA nodes on %d socket(s) = SNC%d or NPS%d%s\n",
               C(YELLOW), numa->num_nodes, sockets, nodes_per_socket, nodes_per_socket, C(RESET));
        printf("  %s  Intel: Sub-NUMA Clustering (SNC) | AMD: NUMA-Per-Socket (NPS)%s\n",
               C(DIM), C(RESET));
        printf("  %s  Each sub-node has its own memory channels for lower latency%s\n",
               C(DIM), C(RESET));
    } else if (sockets > 1) {
        // Multi-socket without SNC
        printf("\n  %sMulti-socket system: %d sockets, %d NUMA nodes%s\n",
               C(CYAN), sockets, numa->num_nodes, C(RESET));
        printf("  %s  SNC/NPS not enabled - each socket is one NUMA node%s\n",
               C(DIM), C(RESET));
        printf("  %s  💡 Enable SNC in BIOS to partition channels for lower latency%s\n",
               C(DIM), C(RESET));
    }

    printf("\n");
    char size_buf[32];

    for (int i = 0; i < numa->num_nodes; i++) {
        const NUMANode *n = &numa->nodes[i];
        int is_last = (i == numa->num_nodes - 1);

        format_size(n->memory_total_mb, size_buf, sizeof(size_buf));
        print_tree_item(0, is_last, "Node %d: %s, CPUs %d-%d",
                        n->node_id, size_buf,
                        n->cpu_list[0],
                        n->cpu_list[n->num_cpus - 1]);
    }

    // Print distance matrix if more than 2 nodes
    if (numa->num_nodes >= 2 && numa->distances[0][1] > 0) {
        printf("\n  NUMA Distances (10=local, higher=remote):\n");
        printf("       ");
        for (int i = 0; i < numa->num_nodes; i++) {
            printf(" N%d ", i);
        }
        printf("\n");
        for (int i = 0; i < numa->num_nodes; i++) {
            printf("    N%d", i);
            for (int j = 0; j < numa->num_nodes; j++) {
                int dist = numa->distances[i][j];
                if (dist == 10) {
                    printf(" %s%2d%s ", C(GREEN), dist, C(RESET));
                } else {
                    printf(" %s%2d%s ", C(YELLOW), dist, C(RESET));
                }
            }
            printf("\n");
        }
    }

    // Tip for accurate per-node bandwidth
    printf("\n  %s💡 Per-node bandwidth: numactl --cpunodebind=0 --membind=0 ./build/show_config%s\n",
           C(CYAN), C(RESET));
}

void topology_print_memory(const MemoryInfo *mem) {
    print_section("MEMORY");

    // Show data sources
    printf("  %sSource: /proc/meminfo, dmidecode (if root), STREAM benchmark%s\n\n",
           C(DIM), C(RESET));

    char total_buf[32], avail_buf[32], theo_bw_buf[32], meas_bw_buf[32];
    format_size(mem->total_mb, total_buf, sizeof(total_buf));
    format_size(mem->available_mb, avail_buf, sizeof(avail_buf));
    format_bandwidth(mem->theoretical_bandwidth_gbs, theo_bw_buf, sizeof(theo_bw_buf));
    format_bandwidth(mem->measured_bandwidth_gbs, meas_bw_buf, sizeof(meas_bw_buf));

    printf("  %sTotal: %s%s (%s available)\n",
           C(BOLD), total_buf, C(RESET), avail_buf);

    if (mem->memory_type[0]) {
        print_tree_item(0, 0, "Type: %s @ %d MT/s", mem->memory_type, mem->memory_speed_mhz);
    }

    print_tree_item(0, 0, "Configuration: %s", mem->channel_config);

    if (mem->num_slots > 0) {
        print_tree_item(0, 0, "Slots: %d/%d populated",
                        mem->slots_populated, mem->num_slots);
    }

    // Show bandwidth measurements with explanation
    printf("\n  %sBandwidth Analysis:%s\n", C(CYAN), C(RESET));

    // Theoretical bandwidth calculation
    if (mem->memory_speed_mhz > 0 && mem->channels_populated > 0) {
        printf("  %s├── Theoretical: %d MT/s × 8 bytes × %d channel(s) = %s%s\n",
               C(DIM), mem->memory_speed_mhz, mem->channels_populated,
               theo_bw_buf, C(RESET));

        // Show SNC relationship for multi-channel configs
        if (mem->channels_populated >= 4) {
            printf("  %s│   └── SNC potential: %d ch ÷ 2 = SNC2 (%d ch/node), ÷ 4 = SNC4 (%d ch/node)%s\n",
                   C(DIM), mem->channels_populated,
                   mem->channels_populated / 2,
                   mem->channels_populated / 4,
                   C(RESET));
        } else if (mem->channels_populated >= 2) {
            printf("  %s│   └── SNC potential: %d ch ÷ 2 = SNC2 (%d ch/node)%s\n",
                   C(DIM), mem->channels_populated,
                   mem->channels_populated / 2,
                   C(RESET));
        }
    } else {
        printf("  %s├── Theoretical: %s (estimated)%s\n",
               C(DIM), theo_bw_buf, C(RESET));
    }

    // Measured bandwidth with methodology
    if (mem->measured_bandwidth_gbs > 0) {
        float efficiency = (mem->theoretical_bandwidth_gbs > 0) ?
            (mem->measured_bandwidth_gbs / mem->theoretical_bandwidth_gbs * 100.0f) : 0;

        printf("  %s├── Measured: %s%s%s (%.0f%% efficiency)%s\n",
               C(DIM),
               C(mem->measured_bandwidth_gbs > 40 ? GREEN : YELLOW),
               meas_bw_buf, C(RESET), efficiency, C(RESET));
        printf("  %s│   Method: STREAM Triad (c[i] = a[i] + s*b[i])%s\n",
               C(DIM), C(RESET));
        printf("  %s│   Buffer: 256 MB × 3 arrays, 3 iterations%s\n",
               C(DIM), C(RESET));
        printf("  %s└── Formula: (256MB × 3 × 3) / time = GB/s%s\n",
               C(DIM), C(RESET));
    }

    // Show DIMM details if available
    if (mem->num_slots > 0 && mem->slots[0].locator[0]) {
        printf("\n  DIMM Layout:\n");
        for (int i = 0; i < mem->num_slots; i++) {
            const MemorySlot *s = &mem->slots[i];
            if (s->populated) {
                char dimm_size[32];
                format_size(s->size_mb, dimm_size, sizeof(dimm_size));
                printf("    %s[%s]%s %s: %s%s @ %d MT/s%s\n",
                       C(GREEN), s->locator, C(RESET),
                       s->type, C(BOLD), dimm_size, s->speed_mhz, C(RESET));
            } else {
                printf("    %s[%s]%s EMPTY\n",
                       C(DIM), s->locator, C(RESET));
            }
        }
    }

    if (mem->channels_populated == 1 && mem->num_slots > 1) {
        print_warning("Single-channel mode - bandwidth reduced by ~50%%");
    }

    if (mem->num_slots > 0 && mem->slots_populated < mem->num_slots) {
        printf("  %s💡 Tip:%s Add %d more DIMM(s) for better bandwidth\n",
               C(CYAN), C(RESET), mem->num_slots - mem->slots_populated);
    }
}

void topology_print_pcie(const PCIeTopology *pcie) {
    print_section("PCIe DEVICES");

    int gpu_count = 0, nic_count = 0, nvme_count = 0;
    char bw_buf[32];

    for (int i = 0; i < pcie->num_devices; i++) {
        const PCIeDevice *d = &pcie->devices[i];

        // Skip bridges and other infrastructure devices
        if (d->link_width == 0) continue;
        if (strstr(d->device_name, "bridge") ||
            strstr(d->device_name, "Bridge") ||
            strstr(d->device_name, "Host") ||
            strstr(d->device_name, "PCI")) continue;

        const char *type_icon = "   ";
        const char *type_color = "";
        if (d->is_gpu) {
            type_icon = "🎮 ";
            type_color = GREEN;
            gpu_count++;
        } else if (d->is_nic) {
            type_icon = "🌐 ";
            type_color = CYAN;
            nic_count++;
        } else if (d->is_nvme) {
            type_icon = "💾 ";
            type_color = MAGENTA;
            nvme_count++;
        }

        format_bandwidth(d->bandwidth_gbs, bw_buf, sizeof(bw_buf));

        // Truncate long device names
        char name[48];
        strncpy(name, d->device_name, sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
        if (strlen(d->device_name) > 45) {
            strcpy(name + 42, "...");
        }

        printf("  %s%s%s%-45s%s x%d Gen%d %s%s%s",
               type_icon, C(type_color), C(BOLD), name, C(RESET),
               d->link_width, d->link_speed,
               C(DIM), bw_buf, C(RESET));

        // Show if not running at max capability
        if (d->link_width < d->link_width_max || d->link_speed < d->link_speed_max) {
            printf(" %s(capable: x%d Gen%d)%s",
                   C(YELLOW), d->link_width_max, d->link_speed_max, C(RESET));
        }
        printf("\n");
    }

    if (gpu_count == 0 && nic_count == 0 && nvme_count == 0) {
        printf("  %sNo significant PCIe devices detected%s\n", C(DIM), C(RESET));
    }

    printf("\n  Summary: %d GPU(s), %d NIC(s), %d NVMe(s)\n",
           gpu_count, nic_count, nvme_count);
}

void topology_print_network(const NetworkTopology *net) {
    print_section("NETWORK INTERFACES");

    if (net->num_interfaces == 0) {
        printf("  %sNo network interfaces detected%s\n", C(DIM), C(RESET));
        return;
    }

    char bw_buf[32];

    for (int i = 0; i < net->num_interfaces; i++) {
        const NetworkInterface *n = &net->interfaces[i];

        const char *status_icon = n->is_up && n->has_link ? "✓" : "✗";
        const char *status_color = n->is_up && n->has_link ? GREEN : RED;

        float bw_gbs = n->speed_mbps / 8000.0f;
        format_bandwidth(bw_gbs, bw_buf, sizeof(bw_buf));

        printf("  %s%s%s %s%-10s%s ",
               C(status_color), status_icon, C(RESET),
               C(BOLD), n->name, C(RESET));

        if (n->has_link) {
            // Color code speed
            const char *speed_color = "";
            if (n->speed_mbps >= 100000) speed_color = GREEN;      // 100 GbE+
            else if (n->speed_mbps >= 10000) speed_color = GREEN;  // 10 GbE
            else if (n->speed_mbps >= 1000) speed_color = YELLOW;  // 1 GbE
            else speed_color = RED;                                 // < 1 GbE

            printf("%s%6lu Mbps%s (%s) ",
                   C(speed_color), (unsigned long)n->speed_mbps, C(RESET), bw_buf);
        } else {
            printf("%sno link    %s ", C(RED), C(RESET));
        }

        if (n->driver[0]) {
            printf("%s[%s]%s ", C(DIM), n->driver, C(RESET));
        }

        if (n->supports_rdma) {
            printf("%s[RDMA]%s ", C(GREEN), C(RESET));
        }
        if (n->is_infiniband) {
            printf("%s[IB]%s ", C(MAGENTA), C(RESET));
        }

        printf("\n");
    }

    // Network capability summary
    printf("\n  For Distributed Training:\n");

    if (net->max_bandwidth_gbs >= 12.5f) {
        print_ok("100 GbE+ available - excellent for distributed training");
    } else if (net->max_bandwidth_gbs >= 1.25f) {
        printf("  %s✓%s 10 GbE available - good for small clusters\n",
               C(GREEN), C(RESET));
    } else if (net->max_bandwidth_gbs >= 0.125f) {
        print_warning("Only 1 GbE - significant bottleneck for distributed training");
    } else {
        print_warning("Very slow network - distributed training not recommended");
    }

    if (net->has_rdma) {
        print_ok("RDMA capable NIC detected - low-latency gradient sync possible");
    }
}

void topology_print_affinity(const AffinityInfo *aff) {
    print_section("THREAD AFFINITY (OpenMP)");

    print_tree_item(0, 0, "OMP_NUM_THREADS: %d", aff->omp_num_threads);
    print_tree_item(0, 0, "OMP_PROC_BIND: %s%s%s",
                    aff->affinity_set ? C(GREEN) : C(YELLOW),
                    aff->omp_proc_bind, C(RESET));
    print_tree_item(0, 1, "OMP_PLACES: %s", aff->omp_places);

    if (!aff->affinity_set) {
        printf("\n");
        print_warning("Thread affinity not configured");
        printf("  %s💡 Recommendation:%s export OMP_PROC_BIND=close OMP_PLACES=cores\n",
               C(CYAN), C(RESET));
    }
}

void topology_print_recommendations(const RecommendationList *recs) {
    if (recs->num_recommendations == 0) {
        print_section("RECOMMENDATIONS");
        print_ok("No significant issues detected!");
        return;
    }

    print_section("RECOMMENDATIONS");

    for (int i = 0; i < recs->num_recommendations; i++) {
        const Recommendation *r = &recs->recommendations[i];

        const char *priority_icon = "";
        const char *priority_color = "";
        switch (r->priority) {
            case REC_PRIORITY_CRITICAL:
                priority_icon = "🔴";
                priority_color = RED;
                break;
            case REC_PRIORITY_HIGH:
                priority_icon = "🟠";
                priority_color = RED;
                break;
            case REC_PRIORITY_MEDIUM:
                priority_icon = "🟡";
                priority_color = YELLOW;
                break;
            case REC_PRIORITY_LOW:
                priority_icon = "🟢";
                priority_color = GREEN;
                break;
        }

        printf("\n  %s %s%s%s\n", priority_icon, C(priority_color), r->title, C(RESET));
        printf("     %s\n", r->description);
        printf("     %s→ %s%s\n", C(CYAN), r->action, C(RESET));
    }
}

void topology_print_distributed_potential(const SystemTopology *topo) {
    print_section("DISTRIBUTED TRAINING POTENTIAL");

    char mem_bw_buf[32], net_bw_buf[32];
    format_bandwidth(topo->memory.theoretical_bandwidth_gbs, mem_bw_buf, sizeof(mem_bw_buf));
    format_bandwidth(topo->network.max_bandwidth_gbs, net_bw_buf, sizeof(net_bw_buf));

    printf("  Single Node Capacity:\n");
    print_tree_item(0, 0, "Compute: %d cores @ %s",
                    topo->cpu.physical_cores,
                    topo->cpu.has_avx512f ? "AVX-512" :
                    (topo->cpu.has_avx2 ? "AVX2" : "AVX"));
    print_tree_item(0, 0, "Memory: %lu GB @ %s",
                    (unsigned long)(topo->memory.total_mb / 1024), mem_bw_buf);
    print_tree_item(0, 1, "Network: %s", net_bw_buf);

    // Estimate sync times for various model sizes
    printf("\n  Estimated Gradient Sync Time (single allreduce):\n");

    uint64_t model_sizes[] = {100, 500, 1000, 7000};  // MB
    const char *model_names[] = {"100 MB (BERT-base)", "500 MB (GPT-2)",
                                  "1 GB (ResNet-50 batch)", "7 GB (LLaMA-7B)"};

    for (int i = 0; i < 4; i++) {
        float sync_time = topology_estimate_network_training_time(
            &topo->network, model_sizes[i]);

        const char *time_color = "";
        if (sync_time < 0.1f) time_color = GREEN;
        else if (sync_time < 1.0f) time_color = YELLOW;
        else time_color = RED;

        printf("    %-25s %s%8.2f sec%s\n",
               model_names[i], C(time_color), sync_time, C(RESET));
    }

    // Multi-node projection
    printf("\n  Multi-Node Projection (assuming identical nodes):\n");
    int nodes[] = {2, 4, 8, 16};
    for (int i = 0; i < 4; i++) {
        int n = nodes[i];
        uint64_t total_mem = topo->memory.total_mb * n;
        int total_cores = topo->cpu.physical_cores * n;

        char total_mem_buf[32];
        format_size(total_mem, total_mem_buf, sizeof(total_mem_buf));

        printf("    %2d nodes: %4d cores, %s memory\n",
               n, total_cores, total_mem_buf);
    }

    // Ring-allreduce topology diagram
    printf("\n  Ring-AllReduce Topology (4 nodes):\n");
    printf("     %s┌─────────┐     ┌─────────┐%s\n", C(CYAN), C(RESET));
    printf("     %s│ Node 0  │────→│ Node 1  │%s\n", C(CYAN), C(RESET));
    printf("     %s│ Worker  │     │ Worker  │%s\n", C(CYAN), C(RESET));
    printf("     %s└────↑────┘     └────│────┘%s\n", C(CYAN), C(RESET));
    printf("     %s     │               │     %s\n", C(CYAN), C(RESET));
    printf("     %s     │               ↓     %s\n", C(CYAN), C(RESET));
    printf("     %s┌────│────┐     ┌────↓────┐%s\n", C(CYAN), C(RESET));
    printf("     %s│ Node 3  │←────│ Node 2  │%s\n", C(CYAN), C(RESET));
    printf("     %s│ Worker  │     │ Worker  │%s\n", C(CYAN), C(RESET));
    printf("     %s└─────────┘     └─────────┘%s\n", C(CYAN), C(RESET));
}

void topology_print_summary(const SystemTopology *topo) {
    print_header("C-Kernel-Engine System Configuration");

    printf("  %sHostname:%s %s\n", C(DIM), C(RESET), topo->hostname);
    printf("  %sKernel:%s   %s\n", C(DIM), C(RESET), topo->kernel_version);
    if (!topo->has_root_access) {
        printf("  %sNote:%s Running without root - some info may be unavailable\n",
               C(YELLOW), C(RESET));
    }

    topology_print_cpu(&topo->cpu);
    topology_print_cache(&topo->cache, topo->cpu.logical_cores);
    topology_print_numa(&topo->numa, topo->cpu.sockets);
    topology_print_memory(&topo->memory);
    topology_print_pcie(&topo->pcie);
    topology_print_network(&topo->network);
    topology_print_affinity(&topo->affinity);

    RecommendationList recs;
    topology_generate_recommendations(topo, &recs);
    topology_print_recommendations(&recs);

    topology_print_distributed_potential(topo);

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
    // Check for --no-color flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-color") == 0) {
            use_colors = 0;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("\nDisplay system hardware configuration for C-Kernel-Engine\n");
            printf("\nOptions:\n");
            printf("  --no-color    Disable colored output\n");
            printf("  --help, -h    Show this help message\n");
            return 0;
        }
    }

    // Check if stdout is a terminal
    if (!isatty(1)) {
        use_colors = 0;
    }

    SystemTopology topo;
    if (topology_discover(&topo) < 0) {
        fprintf(stderr, "Error: Failed to discover system topology\n");
        return 1;
    }

    topology_print_summary(&topo);

    return 0;
}
