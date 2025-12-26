/*
 * C-Kernel-Engine BPE Tokenizer
 *
 * Pure C implementation of Byte-Pair Encoding tokenizer
 * compatible with HuggingFace tokenizer.json format.
 *
 * The tokenizer uses a memory pool for allocations and maps tokens
 * directly to dense embedding indices (token_id == embedding row).
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TOKENIZER_H
#define CK_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum token length in bytes */
#define CK_MAX_TOKEN_LEN 256

/* Maximum vocabulary size */
#define CK_MAX_VOCAB_SIZE 256000

/* Memory pool block size */
#define CK_POOL_BLOCK_SIZE (1024 * 1024)  /* 1MB */

/*
 * Memory pool for tokenizer allocations.
 * Avoids malloc overhead for small allocations.
 */
typedef struct CKPoolBlock {
    uint8_t *data;
    size_t used;
    size_t capacity;
    struct CKPoolBlock *next;
} CKPoolBlock;

typedef struct {
    CKPoolBlock *head;
    CKPoolBlock *current;
    size_t total_allocated;
} CKMemPool;

/*
 * Vocabulary entry.
 * Token string -> ID mapping.
 * IDs are dense indices into the embedding table.
 */
typedef struct CKVocabEntry {
    char *token;           /* Token string (UTF-8) */
    int token_len;         /* Length in bytes */
    int32_t id;            /* Dense embedding index */
    struct CKVocabEntry *next;  /* Hash chain */
} CKVocabEntry;

/*
 * BPE merge rule.
 * Pair of token IDs -> merged token ID.
 */
typedef struct {
    int32_t left;
    int32_t right;
    int32_t merged;
    int priority;  /* Lower = higher priority (earlier in merges list) */
} CKMergeRule;

/*
 * Tokenizer state.
 */
typedef struct {
    /* Memory pool */
    CKMemPool pool;

    /* Vocabulary: token string -> ID */
    int vocab_size;
    CKVocabEntry **vocab_hash;  /* Hash table for string -> ID */
    int vocab_hash_size;

    /* Reverse vocabulary: ID -> token string */
    char **id_to_token;

    /* BPE merge rules */
    CKMergeRule *merges;
    int num_merges;

    /* Merge lookup: (left_id, right_id) -> merge index */
    int *merge_hash;
    int merge_hash_size;

    /* Special tokens */
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;

    /* Config */
    bool add_bos;
    bool add_eos;
} CKTokenizer;

/*
 * Initialize memory pool.
 */
void ck_pool_init(CKMemPool *pool);

/*
 * Allocate from memory pool.
 */
void *ck_pool_alloc(CKMemPool *pool, size_t size);

/*
 * Allocate and copy string.
 */
char *ck_pool_strdup(CKMemPool *pool, const char *s, int len);

/*
 * Free memory pool.
 */
void ck_pool_free(CKMemPool *pool);

/*
 * Initialize tokenizer.
 * Returns 0 on success, -1 on error.
 */
int ck_tokenizer_init(CKTokenizer *tok);

/*
 * Load tokenizer from HuggingFace tokenizer.json.
 * Returns 0 on success, -1 on error.
 */
int ck_tokenizer_load(CKTokenizer *tok, const char *path);

/*
 * Add a token to the vocabulary.
 * Returns the token ID.
 */
int32_t ck_tokenizer_add_token(CKTokenizer *tok, const char *token, int len);

/*
 * Look up token ID by string.
 * Returns token ID or unk_id if not found.
 */
int32_t ck_tokenizer_lookup(const CKTokenizer *tok, const char *token, int len);

/*
 * Add a BPE merge rule.
 */
int ck_tokenizer_add_merge(CKTokenizer *tok, int32_t left, int32_t right, int32_t merged);

/*
 * Look up merge rule for a pair.
 * Returns merge index or -1 if no merge.
 */
int ck_tokenizer_lookup_merge(const CKTokenizer *tok, int32_t left, int32_t right);

/*
 * Encode text to token IDs.
 * Returns number of tokens written to `ids`.
 */
int ck_tokenizer_encode(const CKTokenizer *tok,
                        const char *text,
                        int text_len,
                        int32_t *ids,
                        int max_ids);

/*
 * Decode token IDs to text.
 * Returns number of bytes written to `text`.
 */
int ck_tokenizer_decode(const CKTokenizer *tok,
                        const int32_t *ids,
                        int num_ids,
                        char *text,
                        int max_len);

/*
 * Get token string for an ID.
 * Returns NULL if ID is invalid.
 */
const char *ck_tokenizer_id_to_token(const CKTokenizer *tok, int32_t id);

/*
 * Free tokenizer resources.
 */
void ck_tokenizer_free(CKTokenizer *tok);

/*
 * Get vocabulary size.
 */
static inline int ck_tokenizer_vocab_size(const CKTokenizer *tok) {
    return tok->vocab_size;
}

#ifdef __cplusplus
}
#endif

#endif /* CK_TOKENIZER_H */
