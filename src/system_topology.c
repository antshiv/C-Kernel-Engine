/*
 * system_topology.c - System hardware topology discovery implementation
 *
 * Probes system hardware via /proc, /sys, and external tools to provide
 * comprehensive information for distributed training optimization.
 */

#define _GNU_SOURCE
#include "system_topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <ctype.h>
#include <sched.h>
#include <sys/utsname.h>
#include <sys/sysinfo.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

static void trim_string(char *str) {
    if (!str) return;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) *end-- = '\0';
    char *start = str;
    while (*start && isspace(*start)) start++;
    if (start != str) memmove(str, start, strlen(start) + 1);
}

static int read_file_string(const char *path, char *buf, size_t buf_size) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, buf_size, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    trim_string(buf);
    return 0;
}

static int read_file_int(const char *path) {
    char buf[64];
    if (read_file_string(path, buf, sizeof(buf)) < 0) return -1;
    return atoi(buf);
}

static uint64_t read_file_uint64(const char *path) {
    char buf[64];
    if (read_file_string(path, buf, sizeof(buf)) < 0) return 0;
    return strtoull(buf, NULL, 10);
}

static int run_command(const char *cmd, char *output, size_t output_size) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;

    size_t total = 0;
    while (total < output_size - 1) {
        size_t n = fread(output + total, 1, output_size - 1 - total, fp);
        if (n == 0) break;
        total += n;
    }
    output[total] = '\0';
    int status = pclose(fp);
    return WEXITSTATUS(status);
}

static int count_set_bits(const char *hex_mask) {
    int count = 0;
    for (const char *p = hex_mask; *p; p++) {
        if (*p == ',' || *p == '\n') continue;
        int val = 0;
        if (*p >= '0' && *p <= '9') val = *p - '0';
        else if (*p >= 'a' && *p <= 'f') val = *p - 'a' + 10;
        else if (*p >= 'A' && *p <= 'F') val = *p - 'A' + 10;
        while (val) { count += val & 1; val >>= 1; }
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_cpu(CPUInfo *cpu) {
    memset(cpu, 0, sizeof(*cpu));

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;

    char line[512];
    int processor_count = 0;
    int physical_id_max = -1;
    int core_id_max = -1;

    while (fgets(line, sizeof(line), f)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;

        char *key = line;
        char *value = colon + 1;
        *colon = '\0';
        trim_string(key);
        trim_string(value);

        if (strcmp(key, "processor") == 0) {
            processor_count++;
        } else if (strcmp(key, "model name") == 0 && cpu->model_name[0] == '\0') {
            strncpy(cpu->model_name, value, sizeof(cpu->model_name) - 1);
        } else if (strcmp(key, "vendor_id") == 0 && cpu->vendor[0] == '\0') {
            strncpy(cpu->vendor, value, sizeof(cpu->vendor) - 1);
        } else if (strcmp(key, "cpu family") == 0 && cpu->family == 0) {
            cpu->family = atoi(value);
        } else if (strcmp(key, "model") == 0 && cpu->model == 0) {
            cpu->model = atoi(value);
        } else if (strcmp(key, "stepping") == 0 && cpu->stepping == 0) {
            cpu->stepping = atoi(value);
        } else if (strcmp(key, "cpu MHz") == 0 && cpu->base_freq_mhz == 0) {
            cpu->base_freq_mhz = atof(value);
        } else if (strcmp(key, "physical id") == 0) {
            int id = atoi(value);
            if (id > physical_id_max) physical_id_max = id;
        } else if (strcmp(key, "core id") == 0) {
            int id = atoi(value);
            if (id > core_id_max) core_id_max = id;
        } else if (strcmp(key, "flags") == 0) {
            cpu->has_sse4_2   = strstr(value, "sse4_2") != NULL;
            cpu->has_avx      = strstr(value, " avx ") != NULL || strstr(value, " avx\n") != NULL;
            cpu->has_avx2     = strstr(value, "avx2") != NULL;
            cpu->has_avx512f  = strstr(value, "avx512f") != NULL;
            cpu->has_avx512bw = strstr(value, "avx512bw") != NULL;
            cpu->has_avx512vl = strstr(value, "avx512vl") != NULL;
            cpu->has_amx      = strstr(value, "amx") != NULL;
            cpu->has_vnni     = strstr(value, "vnni") != NULL;
        }
    }
    fclose(f);

    cpu->logical_cores = processor_count;
    cpu->sockets = physical_id_max + 1;
    if (cpu->sockets < 1) cpu->sockets = 1;

    // Read from /sys for more accurate core count
    int cores_per_socket = read_file_int("/sys/devices/system/cpu/cpu0/topology/core_cpus_list");
    if (cores_per_socket < 0) {
        // Fallback: estimate from logical cores and sockets
        cpu->physical_cores = cpu->logical_cores / 2;  // Assume HT
        cpu->cores_per_socket = cpu->physical_cores / cpu->sockets;
    } else {
        // Count unique core IDs
        char path[256];
        int unique_cores = 0;
        int seen_cores[MAX_CPUS] = {0};

        for (int i = 0; i < cpu->logical_cores && i < MAX_CPUS; i++) {
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
            int core_id = read_file_int(path);
            if (core_id >= 0 && core_id < MAX_CPUS && !seen_cores[core_id]) {
                seen_cores[core_id] = 1;
                unique_cores++;
            }
        }
        cpu->physical_cores = unique_cores > 0 ? unique_cores : cpu->logical_cores / 2;
        cpu->cores_per_socket = cpu->physical_cores / cpu->sockets;
    }

    cpu->threads_per_core = cpu->logical_cores / cpu->physical_cores;

    // Try to get max frequency
    int max_freq = read_file_int("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (max_freq > 0) {
        cpu->max_freq_mhz = max_freq / 1000.0f;
    }

    // Estimate PCIe lanes based on CPU model
    if (strstr(cpu->model_name, "Xeon") || strstr(cpu->model_name, "EPYC")) {
        cpu->pcie_lanes_total = 64;  // Server CPUs typically have more
        cpu->pcie_generation = 4;
    } else if (strstr(cpu->model_name, "i9") || strstr(cpu->model_name, "i7")) {
        cpu->pcie_lanes_total = 20;
        cpu->pcie_generation = cpu->has_avx512f ? 4 : 3;
    } else {
        cpu->pcie_lanes_total = 16;
        cpu->pcie_generation = 3;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cache Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_cache(CacheTopology *cache) {
    memset(cache, 0, sizeof(*cache));

    const char *base = "/sys/devices/system/cpu/cpu0/cache";
    DIR *dir = opendir(base);
    if (!dir) return -1;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "index", 5) != 0) continue;

        char path[512];
        CacheInfo *ci = &cache->levels[cache->num_levels];

        snprintf(path, sizeof(path), "%s/%s/level", base, entry->d_name);
        ci->level = read_file_int(path);

        snprintf(path, sizeof(path), "%s/%s/type", base, entry->d_name);
        read_file_string(path, ci->type, sizeof(ci->type));

        snprintf(path, sizeof(path), "%s/%s/size", base, entry->d_name);
        char size_str[32];
        if (read_file_string(path, size_str, sizeof(size_str)) == 0) {
            ci->size_kb = atoi(size_str);  // Usually in KB with 'K' suffix
        }

        snprintf(path, sizeof(path), "%s/%s/coherency_line_size", base, entry->d_name);
        ci->line_size_bytes = read_file_int(path);

        snprintf(path, sizeof(path), "%s/%s/ways_of_associativity", base, entry->d_name);
        ci->ways_of_associativity = read_file_int(path);

        snprintf(path, sizeof(path), "%s/%s/shared_cpu_map", base, entry->d_name);
        char cpu_map[256];
        if (read_file_string(path, cpu_map, sizeof(cpu_map)) == 0) {
            ci->shared_by_cores = count_set_bits(cpu_map);
        }

        if (ci->level == 3) {
            cache->l3_total_kb = ci->size_kb;  // Will be multiplied if multiple
        }

        cache->num_levels++;
        if (cache->num_levels >= MAX_CACHE_LEVELS) break;
    }
    closedir(dir);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// NUMA Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_numa(NUMATopology *numa) {
    memset(numa, 0, sizeof(*numa));

    const char *base = "/sys/devices/system/node";
    DIR *dir = opendir(base);
    if (!dir) {
        // No NUMA, single node system
        numa->num_nodes = 1;
        numa->nodes[0].node_id = 0;

        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            numa->nodes[0].memory_total_mb = si.totalram / (1024 * 1024);
            numa->nodes[0].memory_free_mb = si.freeram / (1024 * 1024);
        }
        return 0;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "node", 4) != 0) continue;
        if (!isdigit(entry->d_name[4])) continue;

        int node_id = atoi(entry->d_name + 4);
        if (node_id >= MAX_NUMA_NODES) continue;

        NUMANode *node = &numa->nodes[numa->num_nodes];
        node->node_id = node_id;

        char path[512];

        // Memory info
        snprintf(path, sizeof(path), "%s/%s/meminfo", base, entry->d_name);
        FILE *f = fopen(path, "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                uint64_t val;
                if (sscanf(line, "Node %*d MemTotal: %lu kB", &val) == 1) {
                    node->memory_total_mb = val / 1024;
                } else if (sscanf(line, "Node %*d MemFree: %lu kB", &val) == 1) {
                    node->memory_free_mb = val / 1024;
                }
            }
            fclose(f);
        }

        // CPU list
        snprintf(path, sizeof(path), "%s/%s/cpulist", base, entry->d_name);
        char cpulist[512];
        if (read_file_string(path, cpulist, sizeof(cpulist)) == 0) {
            // Parse CPU list (e.g., "0-7,16-23")
            char *saveptr;
            char *token = strtok_r(cpulist, ",", &saveptr);
            while (token && node->num_cpus < MAX_CPUS) {
                int start, end;
                if (sscanf(token, "%d-%d", &start, &end) == 2) {
                    for (int i = start; i <= end && node->num_cpus < MAX_CPUS; i++) {
                        node->cpu_list[node->num_cpus++] = i;
                    }
                } else if (sscanf(token, "%d", &start) == 1) {
                    node->cpu_list[node->num_cpus++] = start;
                }
                token = strtok_r(NULL, ",", &saveptr);
            }
        }

        numa->num_nodes++;
    }
    closedir(dir);

    // Read NUMA distances
    char path[512];
    snprintf(path, sizeof(path), "%s/node0/distance", base);
    char dist_str[256];
    if (read_file_string(path, dist_str, sizeof(dist_str)) == 0) {
        char *saveptr;
        char *token = strtok_r(dist_str, " ", &saveptr);
        int col = 0;
        while (token && col < numa->num_nodes) {
            numa->distances[0][col++] = atoi(token);
            token = strtok_r(NULL, " ", &saveptr);
        }
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_memory(MemoryInfo *mem) {
    memset(mem, 0, sizeof(*mem));

    // Basic memory info from /proc/meminfo
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            uint64_t val;
            if (sscanf(line, "MemTotal: %lu kB", &val) == 1) {
                mem->total_mb = val / 1024;
            } else if (sscanf(line, "MemAvailable: %lu kB", &val) == 1) {
                mem->available_mb = val / 1024;
            } else if (sscanf(line, "Cached: %lu kB", &val) == 1) {
                mem->cached_mb = val / 1024;
            }
        }
        fclose(f);
    }

    // Try to get DIMM info via dmidecode (requires root)
    char output[8192];
    if (run_command("dmidecode -t memory 2>/dev/null", output, sizeof(output)) == 0 &&
        strlen(output) > 100) {

        char *line = strtok(output, "\n");
        MemorySlot *current_slot = NULL;

        while (line) {
            trim_string(line);

            if (strstr(line, "Memory Device")) {
                if (mem->num_slots < MAX_MEMORY_SLOTS) {
                    current_slot = &mem->slots[mem->num_slots++];
                    memset(current_slot, 0, sizeof(*current_slot));
                    current_slot->slot_number = mem->num_slots;
                }
            } else if (current_slot) {
                uint64_t val;
                int ival;
                char str[64];

                if (sscanf(line, "Size: %lu MB", &val) == 1) {
                    current_slot->size_mb = val;
                    current_slot->populated = true;
                    mem->slots_populated++;
                } else if (sscanf(line, "Size: %lu GB", &val) == 1) {
                    current_slot->size_mb = val * 1024;
                    current_slot->populated = true;
                    mem->slots_populated++;
                } else if (strstr(line, "Size: No Module")) {
                    current_slot->populated = false;
                } else if (sscanf(line, "Speed: %d MT/s", &ival) == 1 ||
                           sscanf(line, "Speed: %d MHz", &ival) == 1) {
                    current_slot->speed_mhz = ival;
                    if (mem->memory_speed_mhz == 0) mem->memory_speed_mhz = ival;
                } else if (sscanf(line, "Type: %63s", str) == 1) {
                    strncpy(current_slot->type, str, sizeof(current_slot->type) - 1);
                    if (mem->memory_type[0] == '\0') {
                        strncpy(mem->memory_type, str, sizeof(mem->memory_type) - 1);
                    }
                } else if (sscanf(line, "Locator: %63s", str) == 1) {
                    strncpy(current_slot->locator, str, sizeof(current_slot->locator) - 1);
                } else if (sscanf(line, "Rank: %d", &ival) == 1) {
                    current_slot->rank = ival;
                } else if (sscanf(line, "Data Width: %d bits", &ival) == 1) {
                    current_slot->data_width_bits = ival;
                }
            }

            line = strtok(NULL, "\n");
        }
    }

    // Estimate channel configuration
    if (mem->slots_populated > 0) {
        if (mem->slots_populated == 1) {
            strcpy(mem->channel_config, "Single-channel");
            mem->num_channels = 1;
            mem->channels_populated = 1;
        } else if (mem->slots_populated == 2) {
            strcpy(mem->channel_config, "Dual-channel");
            mem->num_channels = 2;
            mem->channels_populated = 2;
        } else if (mem->slots_populated == 4) {
            strcpy(mem->channel_config, "Quad-channel");
            mem->num_channels = 4;
            mem->channels_populated = 4;
        } else if (mem->slots_populated >= 6) {
            strcpy(mem->channel_config, "Hexa-channel or more");
            mem->num_channels = 6;
            mem->channels_populated = mem->slots_populated;
        } else {
            snprintf(mem->channel_config, sizeof(mem->channel_config),
                     "%d DIMMs", mem->slots_populated);
            mem->num_channels = mem->slots_populated;
            mem->channels_populated = mem->slots_populated;
        }

        // Estimate bandwidth
        // DDR4: speed * 8 bytes * channels
        // DDR5: speed * 8 bytes * channels (but DDR5 has 2 channels per DIMM)
        float bytes_per_transfer = 8.0f;
        if (strstr(mem->memory_type, "DDR5")) {
            mem->theoretical_bandwidth_gbs =
                (mem->memory_speed_mhz * bytes_per_transfer * mem->channels_populated * 2) / 1000.0f;
        } else {
            mem->theoretical_bandwidth_gbs =
                (mem->memory_speed_mhz * bytes_per_transfer * mem->channels_populated) / 1000.0f;
        }
    } else {
        // Fallback estimation
        strcpy(mem->channel_config, "Unknown");
        mem->theoretical_bandwidth_gbs = 50.0f;  // Conservative estimate
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PCIe Discovery
// ═══════════════════════════════════════════════════════════════════════════════

static float pcie_bandwidth_gbs(int gen, int width) {
    // GB/s per lane per generation (accounting for encoding overhead)
    float per_lane[] = {0, 0.25f, 0.5f, 0.985f, 1.969f, 3.938f, 7.877f};
    if (gen < 1 || gen > 6) gen = 3;
    return per_lane[gen] * width;
}

int topology_discover_pcie(PCIeTopology *pcie) {
    memset(pcie, 0, sizeof(*pcie));

    char output[32768];
    if (run_command("lspci -vvv 2>/dev/null", output, sizeof(output)) != 0) {
        return -1;
    }

    PCIeDevice *current = NULL;
    char *line = strtok(output, "\n");

    while (line) {
        // New device line: "00:1f.0 ISA bridge: Intel..."
        if (strlen(line) > 0 && isxdigit(line[0]) && line[2] == ':') {
            if (pcie->num_devices < MAX_PCIE_DEVICES) {
                current = &pcie->devices[pcie->num_devices++];
                memset(current, 0, sizeof(*current));

                // Parse BDF
                sscanf(line, "%x:%x.%x", &current->bus, &current->device, &current->function);

                // Get device name (after the type)
                char *name_start = strchr(line, ':');
                if (name_start) {
                    name_start = strchr(name_start + 1, ':');
                    if (name_start) {
                        name_start++;
                        while (*name_start == ' ') name_start++;
                        strncpy(current->device_name, name_start,
                                sizeof(current->device_name) - 1);
                    }
                }

                // Check device type
                current->is_gpu = (strstr(line, "VGA") != NULL ||
                                   strstr(line, "3D controller") != NULL ||
                                   strstr(line, "Display") != NULL);
                current->is_nic = (strstr(line, "Ethernet") != NULL ||
                                   strstr(line, "Network") != NULL ||
                                   strstr(line, "InfiniBand") != NULL);
                current->is_nvme = (strstr(line, "Non-Volatile memory") != NULL);
            }
        } else if (current) {
            // Parse LnkCap and LnkSta for PCIe info
            if (strstr(line, "LnkCap:")) {
                char *speed = strstr(line, "Speed ");
                char *width = strstr(line, "Width x");
                if (speed) {
                    float gts;
                    if (sscanf(speed, "Speed %fGT/s", &gts) == 1) {
                        if (gts >= 64) current->link_speed_max = 6;
                        else if (gts >= 32) current->link_speed_max = 5;
                        else if (gts >= 16) current->link_speed_max = 4;
                        else if (gts >= 8) current->link_speed_max = 3;
                        else if (gts >= 5) current->link_speed_max = 2;
                        else current->link_speed_max = 1;
                    }
                }
                if (width) {
                    sscanf(width, "Width x%d", &current->link_width_max);
                }
            } else if (strstr(line, "LnkSta:")) {
                char *speed = strstr(line, "Speed ");
                char *width = strstr(line, "Width x");
                if (speed) {
                    float gts;
                    if (sscanf(speed, "Speed %fGT/s", &gts) == 1) {
                        if (gts >= 64) current->link_speed = 6;
                        else if (gts >= 32) current->link_speed = 5;
                        else if (gts >= 16) current->link_speed = 4;
                        else if (gts >= 8) current->link_speed = 3;
                        else if (gts >= 5) current->link_speed = 2;
                        else current->link_speed = 1;
                    }
                }
                if (width) {
                    sscanf(width, "Width x%d", &current->link_width);
                }
            }
        }

        line = strtok(NULL, "\n");
    }

    // Calculate bandwidths and summary
    for (int i = 0; i < pcie->num_devices; i++) {
        PCIeDevice *d = &pcie->devices[i];
        d->bandwidth_gbs = pcie_bandwidth_gbs(d->link_speed, d->link_width);
        d->bandwidth_max_gbs = pcie_bandwidth_gbs(d->link_speed_max, d->link_width_max);

        if (d->link_width > 0) {
            pcie->total_lanes_used += d->link_width;
        }
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Network Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_network(NetworkTopology *net) {
    memset(net, 0, sizeof(*net));

    const char *base = "/sys/class/net";
    DIR *dir = opendir(base);
    if (!dir) return -1;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (strcmp(entry->d_name, "lo") == 0) continue;  // Skip loopback

        if (net->num_interfaces >= MAX_NICS) break;
        NetworkInterface *nic = &net->interfaces[net->num_interfaces];
        memset(nic, 0, sizeof(*nic));

        strncpy(nic->name, entry->d_name, sizeof(nic->name) - 1);

        char path[512];

        // Check if interface is up
        snprintf(path, sizeof(path), "%s/%s/operstate", base, entry->d_name);
        char state[32];
        if (read_file_string(path, state, sizeof(state)) == 0) {
            nic->is_up = (strcmp(state, "up") == 0);
        }

        // Get speed
        snprintf(path, sizeof(path), "%s/%s/speed", base, entry->d_name);
        int speed = read_file_int(path);
        if (speed > 0) {
            nic->speed_mbps = speed;
            nic->has_link = true;
        }

        // Get MTU
        snprintf(path, sizeof(path), "%s/%s/mtu", base, entry->d_name);
        nic->mtu = read_file_int(path);

        // Get MAC address
        snprintf(path, sizeof(path), "%s/%s/address", base, entry->d_name);
        read_file_string(path, nic->mac_address, sizeof(nic->mac_address));

        // Get driver
        snprintf(path, sizeof(path), "%s/%s/device/driver", base, entry->d_name);
        char driver_link[512];
        ssize_t len = readlink(path, driver_link, sizeof(driver_link) - 1);
        if (len > 0) {
            driver_link[len] = '\0';
            char *driver_name = strrchr(driver_link, '/');
            if (driver_name) {
                strncpy(nic->driver, driver_name + 1, sizeof(nic->driver) - 1);
            }
        }

        // Check for InfiniBand
        snprintf(path, sizeof(path), "/sys/class/infiniband/%s", entry->d_name);
        if (access(path, F_OK) == 0) {
            nic->is_infiniband = true;
            nic->supports_rdma = true;
        }

        // Check for RoCE capability
        if (strstr(nic->driver, "mlx") || strstr(nic->driver, "bnxt") ||
            strstr(nic->driver, "qed")) {
            nic->supports_roce = true;
            nic->supports_rdma = true;
        }

        // Get PCI address
        snprintf(path, sizeof(path), "%s/%s/device", base, entry->d_name);
        char pci_link[512];
        len = readlink(path, pci_link, sizeof(pci_link) - 1);
        if (len > 0) {
            pci_link[len] = '\0';
            char *pci = strrchr(pci_link, '/');
            if (pci) {
                strncpy(nic->pci_address, pci + 1, sizeof(nic->pci_address) - 1);
            }
        }

        // Calculate bandwidth
        float bandwidth = nic->speed_mbps / 8000.0f;  // Mbps to GB/s

        if (bandwidth > net->max_bandwidth_gbs) {
            net->max_bandwidth_gbs = bandwidth;
            net->best_interface_idx = net->num_interfaces;
        }

        if (nic->supports_rdma) {
            net->has_rdma = true;
        }

        net->num_interfaces++;
    }
    closedir(dir);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Affinity Discovery
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover_affinity(AffinityInfo *aff) {
    memset(aff, 0, sizeof(*aff));

    // OpenMP settings
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        aff->omp_num_threads = atoi(omp_threads);
    } else {
        aff->omp_num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    }

    const char *omp_bind = getenv("OMP_PROC_BIND");
    if (omp_bind) {
        strncpy(aff->omp_proc_bind, omp_bind, sizeof(aff->omp_proc_bind) - 1);
        aff->affinity_set = true;
    } else {
        strcpy(aff->omp_proc_bind, "not set");
    }

    const char *omp_places = getenv("OMP_PLACES");
    if (omp_places) {
        strncpy(aff->omp_places, omp_places, sizeof(aff->omp_places) - 1);
    } else {
        strcpy(aff->omp_places, "not set");
    }

    // Current process affinity
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int i = 0; i < MAX_CPUS && aff->num_affinity_cpus < MAX_CPUS; i++) {
            if (CPU_ISSET(i, &mask)) {
                aff->affinity_cpus[aff->num_affinity_cpus++] = i;
            }
        }
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Discovery Function
// ═══════════════════════════════════════════════════════════════════════════════

int topology_discover(SystemTopology *topo) {
    memset(topo, 0, sizeof(*topo));

    // Get hostname and kernel version
    gethostname(topo->hostname, sizeof(topo->hostname));

    struct utsname uts;
    if (uname(&uts) == 0) {
        snprintf(topo->kernel_version, sizeof(topo->kernel_version),
                 "%s %s", uts.sysname, uts.release);
    }

    // Check for root access
    topo->has_root_access = (geteuid() == 0);

    // Run all discovery functions
    topology_discover_cpu(&topo->cpu);
    topology_discover_cache(&topo->cache);
    topology_discover_numa(&topo->numa);
    topology_discover_memory(&topo->memory);
    topology_discover_pcie(&topo->pcie);
    topology_discover_network(&topo->network);
    topology_discover_affinity(&topo->affinity);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Recommendations Generation
// ═══════════════════════════════════════════════════════════════════════════════

int topology_generate_recommendations(const SystemTopology *topo,
                                       RecommendationList *recs) {
    memset(recs, 0, sizeof(*recs));

    // Memory recommendations
    if (topo->memory.slots_populated > 0 &&
        topo->memory.slots_populated < topo->memory.num_slots) {

        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_MEDIUM;
        r->category = REC_CATEGORY_MEMORY;
        strcpy(r->title, "Memory Slots Available");
        snprintf(r->description, sizeof(r->description),
                 "Only %d of %d memory slots populated. Adding more DIMMs "
                 "could increase memory bandwidth.",
                 topo->memory.slots_populated, topo->memory.num_slots);
        snprintf(r->action, sizeof(r->action),
                 "Consider adding %d more matching DIMMs for better bandwidth",
                 topo->memory.num_slots - topo->memory.slots_populated);
    }

    // Single-channel warning
    if (topo->memory.channels_populated == 1 && topo->memory.num_slots > 1) {
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_HIGH;
        r->category = REC_CATEGORY_MEMORY;
        strcpy(r->title, "Single-Channel Memory");
        strcpy(r->description,
               "Running in single-channel mode significantly reduces memory bandwidth. "
               "This will impact training performance.");
        strcpy(r->action,
               "Add a second DIMM in the correct slot to enable dual-channel mode");
    }

    // Affinity recommendations
    if (!topo->affinity.affinity_set) {
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_MEDIUM;
        r->category = REC_CATEGORY_AFFINITY;
        strcpy(r->title, "OpenMP Affinity Not Set");
        strcpy(r->description,
               "OpenMP thread affinity is not configured. Threads may migrate "
               "between cores causing cache misses and NUMA penalties.");
        strcpy(r->action,
               "export OMP_PROC_BIND=close OMP_PLACES=cores");
    }

    // Network recommendations
    if (topo->network.max_bandwidth_gbs < 1.0f) {  // Less than 10 GbE
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_MEDIUM;
        r->category = REC_CATEGORY_NETWORK;
        strcpy(r->title, "Slow Network for Distributed Training");
        snprintf(r->description, sizeof(r->description),
                 "Maximum network bandwidth is %.2f GB/s. This will be a "
                 "significant bottleneck for distributed training.",
                 topo->network.max_bandwidth_gbs);
        strcpy(r->action,
               "Consider upgrading to 10GbE+ or InfiniBand for distributed training");
    }

    // RDMA recommendation
    if (!topo->network.has_rdma && topo->network.num_interfaces > 0) {
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_LOW;
        r->category = REC_CATEGORY_NETWORK;
        strcpy(r->title, "No RDMA-Capable NICs");
        strcpy(r->description,
               "No RDMA-capable network adapters detected. RDMA enables direct "
               "memory access between nodes, reducing latency for gradient sync.");
        strcpy(r->action,
               "Consider Mellanox ConnectX or Intel E810 for RDMA support");
    }

    // SIMD recommendations
    if (!topo->cpu.has_avx2) {
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_LOW;
        r->category = REC_CATEGORY_CPU;
        strcpy(r->title, "Limited SIMD Support");
        strcpy(r->description,
               "CPU does not support AVX2. Kernel performance will be limited.");
        strcpy(r->action, "AVX2+ CPUs provide significantly better performance");
    }

    // NUMA warning for multi-socket
    if (topo->numa.num_nodes > 1) {
        Recommendation *r = &recs->recommendations[recs->num_recommendations++];
        r->priority = REC_PRIORITY_MEDIUM;
        r->category = REC_CATEGORY_CPU;
        strcpy(r->title, "Multi-NUMA System Detected");
        snprintf(r->description, sizeof(r->description),
                 "System has %d NUMA nodes. Cross-node memory access is slower. "
                 "Ensure data locality for best performance.",
                 topo->numa.num_nodes);
        strcpy(r->action,
               "Use numactl --localalloc or NUMA-aware memory allocation");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

float topology_estimate_memory_bandwidth(const MemoryInfo *mem) {
    return mem->theoretical_bandwidth_gbs;
}

float topology_estimate_network_training_time(const NetworkTopology *net,
                                               uint64_t model_size_mb) {
    if (net->max_bandwidth_gbs <= 0) return -1;

    // Time to transfer model_size_mb in seconds
    // Account for protocol overhead (~10%)
    float effective_bw = net->max_bandwidth_gbs * 0.9f * 1024;  // Convert to MB/s
    return model_size_mb / effective_bw;
}
