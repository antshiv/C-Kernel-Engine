/*
 * system_topology.h - System hardware topology discovery for distributed training
 *
 * Probes CPU, memory, NUMA, PCIe, and network configuration to provide
 * recommendations for optimal distributed training setup.
 */

#ifndef SYSTEM_TOPOLOGY_H
#define SYSTEM_TOPOLOGY_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

#define MAX_NUMA_NODES      8
#define MAX_CPUS            256
#define MAX_CACHE_LEVELS    4
#define MAX_NICS            8
#define MAX_PCIE_DEVICES    32
#define MAX_MEMORY_SLOTS    16
#define MAX_STR_LEN         256

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char model_name[MAX_STR_LEN];
    char vendor[64];
    int  family;
    int  model;
    int  stepping;

    int  physical_cores;
    int  logical_cores;
    int  sockets;
    int  cores_per_socket;
    int  threads_per_core;

    float base_freq_mhz;
    float max_freq_mhz;

    // SIMD capabilities
    bool has_sse4_2;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512vl;
    bool has_avx512_bf16;   // AVX-512 BF16 instructions
    bool has_amx;           // Intel AMX (any variant)
    bool has_amx_tile;      // AMX tile operations
    bool has_amx_int8;      // AMX INT8 matrix multiply
    bool has_amx_bf16;      // AMX BF16 matrix multiply
    bool has_vnni;          // Vector Neural Network Instructions

    // PCIe lanes (from CPU)
    int  pcie_lanes_total;
    int  pcie_generation;
} CPUInfo;

// ═══════════════════════════════════════════════════════════════════════════════
// Cache Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int  level;             // L1, L2, L3
    char type[16];          // "Data", "Instruction", "Unified"
    int  size_kb;
    int  line_size_bytes;
    int  ways_of_associativity;
    int  shared_by_cores;   // How many cores share this cache
} CacheInfo;

typedef struct {
    int       num_levels;
    CacheInfo levels[MAX_CACHE_LEVELS];
    int       l3_total_kb;  // Total L3 across all sockets
} CacheTopology;

// ═══════════════════════════════════════════════════════════════════════════════
// NUMA Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int      node_id;
    uint64_t memory_total_mb;
    uint64_t memory_free_mb;
    int      cpu_list[MAX_CPUS];
    int      num_cpus;
} NUMANode;

typedef struct {
    int      num_nodes;
    NUMANode nodes[MAX_NUMA_NODES];
    // Inter-node distances (for NUMA-aware allocation)
    int      distances[MAX_NUMA_NODES][MAX_NUMA_NODES];
} NUMATopology;

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int      slot_number;
    char     locator[64];       // "DIMM_A1", "P1-DIMMA1", etc.
    bool     populated;
    uint64_t size_mb;
    int      speed_mhz;
    char     type[32];          // "DDR4", "DDR5"
    int      rank;
    int      data_width_bits;
} MemorySlot;

typedef struct {
    uint64_t total_mb;
    uint64_t available_mb;
    uint64_t cached_mb;

    // DIMM configuration (requires dmidecode/root)
    int         num_slots;
    int         slots_populated;
    MemorySlot  slots[MAX_MEMORY_SLOTS];

    // Channel configuration
    int      num_channels;
    int      channels_populated;
    char     channel_config[64];    // "Dual-channel", "Quad-channel", etc.

    // Theoretical bandwidth
    float    theoretical_bandwidth_gbs;
    float    measured_bandwidth_gbs;    // If we run a quick test

    // Memory type detected
    char     memory_type[32];
    int      memory_speed_mhz;
} MemoryInfo;

// ═══════════════════════════════════════════════════════════════════════════════
// PCIe Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char     slot_id[32];
    char     device_name[MAX_STR_LEN];
    char     vendor[64];
    int      domain;
    int      bus;
    int      device;
    int      function;

    int      link_width;        // x1, x4, x8, x16
    int      link_width_max;    // Maximum supported
    int      link_speed;        // PCIe generation (3, 4, 5, 6)
    int      link_speed_max;

    float    bandwidth_gbs;     // Current bandwidth
    float    bandwidth_max_gbs; // Maximum possible

    bool     is_gpu;
    bool     is_nic;
    bool     is_nvme;
} PCIeDevice;

typedef struct {
    int         num_devices;
    PCIeDevice  devices[MAX_PCIE_DEVICES];

    // Summary
    int         total_lanes_used;
    int         total_lanes_available;
    int         empty_x16_slots;
    int         empty_x8_slots;
    int         empty_x4_slots;
} PCIeTopology;

// ═══════════════════════════════════════════════════════════════════════════════
// Network Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char     name[32];          // eth0, enp3s0, ib0
    char     driver[64];
    char     pci_address[32];

    uint64_t speed_mbps;        // Link speed in Mbps
    bool     is_up;
    bool     has_link;

    // Capabilities
    bool     supports_rdma;     // RoCE or InfiniBand
    bool     supports_roce;
    bool     is_infiniband;
    int      ib_port;

    char     mac_address[24];
    int      mtu;

    // PCIe info for this NIC
    int      pcie_width;
    int      pcie_gen;
    float    pcie_bandwidth_gbs;
} NetworkInterface;

typedef struct {
    int               num_interfaces;
    NetworkInterface  interfaces[MAX_NICS];

    // Best interface for distributed training
    int               best_interface_idx;
    float             max_bandwidth_gbs;
    bool              has_rdma;
} NetworkTopology;

// ═══════════════════════════════════════════════════════════════════════════════
// OpenMP / Affinity Information
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int      omp_num_threads;
    char     omp_proc_bind[32];     // "close", "spread", "master", etc.
    char     omp_places[64];        // "cores", "threads", "sockets"
    bool     affinity_set;

    // Current process affinity
    int      affinity_cpus[MAX_CPUS];
    int      num_affinity_cpus;
} AffinityInfo;

// ═══════════════════════════════════════════════════════════════════════════════
// Complete System Topology
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    CPUInfo         cpu;
    CacheTopology   cache;
    NUMATopology    numa;
    MemoryInfo      memory;
    PCIeTopology    pcie;
    NetworkTopology network;
    AffinityInfo    affinity;

    // Flags
    bool            has_root_access;
    char            hostname[256];
    char            kernel_version[128];
} SystemTopology;

// ═══════════════════════════════════════════════════════════════════════════════
// Recommendations
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    REC_PRIORITY_LOW,
    REC_PRIORITY_MEDIUM,
    REC_PRIORITY_HIGH,
    REC_PRIORITY_CRITICAL
} RecommendationPriority;

typedef enum {
    REC_CATEGORY_MEMORY,
    REC_CATEGORY_CPU,
    REC_CATEGORY_NETWORK,
    REC_CATEGORY_AFFINITY,
    REC_CATEGORY_PCIE
} RecommendationCategory;

#define MAX_RECOMMENDATIONS 32

typedef struct {
    RecommendationPriority priority;
    RecommendationCategory category;
    char                   title[128];
    char                   description[512];
    char                   action[256];
} Recommendation;

typedef struct {
    int            num_recommendations;
    Recommendation recommendations[MAX_RECOMMENDATIONS];
} RecommendationList;

// ═══════════════════════════════════════════════════════════════════════════════
// API Functions
// ═══════════════════════════════════════════════════════════════════════════════

// Main discovery function
int  topology_discover(SystemTopology *topo);

// Individual discovery functions
int  topology_discover_cpu(CPUInfo *cpu);
int  topology_discover_cache(CacheTopology *cache);
int  topology_discover_numa(NUMATopology *numa);
int  topology_discover_memory(MemoryInfo *mem);
int  topology_discover_pcie(PCIeTopology *pcie);
int  topology_discover_network(NetworkTopology *net);
int  topology_discover_affinity(AffinityInfo *aff);

// Generate recommendations
int  topology_generate_recommendations(const SystemTopology *topo,
                                        RecommendationList *recs);

// Display functions
void topology_print_summary(const SystemTopology *topo);
void topology_print_cpu(const CPUInfo *cpu);
void topology_print_cache(const CacheTopology *cache, int logical_cores);
void topology_print_numa(const NUMATopology *numa);
void topology_print_memory(const MemoryInfo *mem);
void topology_print_pcie(const PCIeTopology *pcie);
void topology_print_network(const NetworkTopology *net);
void topology_print_affinity(const AffinityInfo *aff);
void topology_print_recommendations(const RecommendationList *recs);

// Distributed training info
void topology_print_distributed_potential(const SystemTopology *topo);

// Utility
float topology_estimate_memory_bandwidth(const MemoryInfo *mem);
float topology_estimate_network_training_time(const NetworkTopology *net,
                                               uint64_t model_size_mb);

#ifdef __cplusplus
}
#endif

#endif // SYSTEM_TOPOLOGY_H
