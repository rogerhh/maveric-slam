/* local_feature_pool.h
 * A local feature pool is a hash map that stores local features indexed by their word id
 **/

#include "types.h"

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define MAX_LOCAL_FRAMES 8
#define MAX_LOCAL_FEATURES 1000
#define LOCAL_FEATURE_POOL_MAX_LOAD_FACTOR 0.75
#define LOCAL_FEATURE_POOL_CAPACITY 3000

typedef struct {
    int word_id;                // If -1, then the entry is empty
    int frame_ptr;
    int num_frames;
    int frames[MAX_LOCAL_FRAMES];
    Vector3f coords_3D;
} LocalFeature;

void init_local_feature(LocalFeature* feature) {
    feature->word_id = -1;
    feature->frame_ptr = 0;
    feature->num_frames = 0;
}

void init_local_feature_with_id(LocalFeature* feature, int word_id, int frame_num) {
    feature->word_id = word_id;
    feature->frame_ptr = 0;
    feature->num_frames = 1;
    feature->frames[0] = frame_num;
}

void update_local_feature(LocalFeature* feature, int frame_num) {
    if(feature->num_frames < MAX_LOCAL_FRAMES) {
        feature->frames[(feature->frame_ptr + feature->num_frames) % MAX_LOCAL_FRAMES] 
            = frame_num;
        feature->num_frames++;
    }
    else {
        feature->frames[feature->frame_ptr] = frame_num;
        feature->frame_ptr = (feature->frame_ptr + 1) % MAX_LOCAL_FRAMES;
    }
}

bool remove_old_frame(LocalFeature* feature, int oldest_keep_frame) {
    if(feature->word_id == -1) {
        return false;
    }
    int oldest_frame = feature->frames[feature->frame_ptr];
    if(oldest_frame < oldest_keep_frame) {
        feature->frame_ptr = (feature->frame_ptr + 1) % MAX_LOCAL_FRAMES;
        feature->num_frames--;
    }
    if(feature->num_frames == 0) {
        return true;
    }
    return false;
}

typedef struct {
    int key;
    LocalFeature value;
    bool is_occupied;
} HashEntry;

void init_hash_entry(HashEntry* entry) {
    entry->key = -1;
    entry->is_occupied = false;
    init_local_feature(&entry->value);
}

void delete_hash_entry(HashEntry* entry) {
    entry->key = -1;
    entry->is_occupied = false;
    init_local_feature(&entry->value);
}

typedef struct {
    HashEntry entries[LOCAL_FEATURE_POOL_CAPACITY];
    int size;
    int capacity;
} LocalFeaturePool;

typedef struct {
    LocalFeature* feature;
    bool inserted;
} LocalFeaturePoolInsertResult;

int hash(int key, int capacity) {
    return key % capacity;
}

void init_local_feature_pool(LocalFeaturePool* pool) {
    pool->size = 0;
    pool->capacity = LOCAL_FEATURE_POOL_CAPACITY;
    for(int i = 0; i < pool->capacity; i++) {
        init_hash_entry(&pool->entries[i]);
    }
}

// Try to insert a local feature into the pool
// Returns true if the feature was inserted, false otherwise
// feature points to the feature that was inserted
LocalFeaturePoolInsertResult local_feature_pool_insert(LocalFeaturePool* pool, 
                                                       int key, LocalFeature value) {
    if(pool->size >= pool->capacity) {
        LocalFeaturePoolInsertResult result = {NULL, false};
        return result;
    }

    int index = hash(key, pool->capacity);
    for(int i = 0; i < pool->capacity; i++) {
        if(pool->entries[index].key == key) {
            LocalFeaturePoolInsertResult result = {&pool->entries[index].value, false};
            return result;
        }
        if(!pool->entries[index].is_occupied) {
            pool->size++;
            pool->entries[index].key = key;
            pool->entries[index].value = value;
            pool->entries[index].is_occupied = true;
            LocalFeaturePoolInsertResult result = {&pool->entries[index].value, true};
            return result;
        }
        index = (index + 1) % pool->capacity;
    }
}

// For each replacement, find a new index between hash(key[remove_index]) and remove_index
// Then do the same for the replacement as well
// Find the last valid index to replace to minimize the number of replacements
// The final replace index is the one that needs to be removed
int chain_replacement(LocalFeaturePool* pool, int remove_index) {
    int last_replace_index = remove_index;
    while(1) {
        int replace_index = -1;
        int index = (remove_index + 1) % pool->capacity;
        bool wrap_around = (index == 0);
        for(int i = 0; i < pool->capacity; i++) {
            if(!pool->entries[index].is_occupied) {
                break;
            }
            int hash_index = hash(pool->entries[index].key, pool->capacity);
            if(!wrap_around && hash_index <= remove_index) {
                replace_index = index;
            }
            else if(wrap_around && hash_index > index && hash_index <= remove_index) {
                replace_index = index;
            }

            index = (index + 1) % pool->capacity;
            if(index == 0) {
                wrap_around = true;
            }
        }
        if(replace_index == -1) {
            break;
        }
        pool->entries[remove_index] = pool->entries[replace_index];
        remove_index = replace_index;
        last_replace_index = replace_index;
    }
    return last_replace_index;
}

bool local_feature_pool_delete(LocalFeaturePool* pool, int key) {
    int index = hash(key, pool->capacity);
    int start_index = index;
    int remove_index = -1;
    // First find the key
    for(int i = 0; i < pool->capacity; i++) {
        // If entry is empty, then the key is not in the pool
        if(!pool->entries[index].is_occupied) {
            printf("Key not found\n");
            exit(0);
        }
        if(pool->entries[index].key == key) {
            remove_index = index;
            break;
        }
        index = (index + 1) % pool->capacity;
    }

    int replace_index = chain_replacement(pool, remove_index);

    delete_hash_entry(&pool->entries[replace_index]);
    pool->size--;


    // int replace_index = -1;
    // if(remove_index >= start_index) {
    //     // Find a replacement for the removed entry
    //     // The replacement's hash needs to be the last available hash between start_index and remove_index (inclusive)
    //     index = (remove_index + 1) % pool->capacity;
    //     for(int i = 0; i < pool->capacity; i++) {
    //         if(!pool->entries[index].is_occupied) {
    //             break;
    //         }
    //         int hash_index = hash(pool->entries[index].key, pool->capacity);
    //         if(hash_index >= start_index && hash_index <= remove_index) {
    //             replace_index = index;
    //             break;
    //         }
    //         index = (index + 1) % pool->capacity;
    //     }
    // }
    // else {
    //     index = (remove_index + 1) % pool->capacity;
    //     // Find a replacement for the removed entry
    //     // The replacement's hash needs to be the last available hash after start_index or before remove_index (inclusive)
    //     for(int i = 0; i < pool->capacity; i++) {
    //         if(!pool->entries[index].is_occupied) {
    //             break;
    //         }
    //         int hash_index = hash(pool->entries[index].key, pool->capacity);
    //         if(hash_index >= start_index || hash_index <= remove_index) {
    //             replace_index = index;
    //             break;
    //         }
    //         index = (index + 1) % pool->capacity;
    //     }
    // }

    // if(replace_index != -1) {
    //     pool->entries[remove_index] = pool->entries[replace_index];
    //     delete_hash_entry(&pool->entries[replace_index]);
    // }
}

void local_feature_pool_rehash(LocalFeaturePool* pool) {
    HashEntry old_entries[pool->capacity];
    for(int i = 0; i < pool->capacity; i++) {
        old_entries[i] = pool->entries[i];
    }

    pool->size = 0;
    for(int i = 0; i < pool->capacity; i++) {
        init_hash_entry(&pool->entries[i]);
    }

    for(int i = 0; i < pool->capacity; i++) {
        if(old_entries[i].is_occupied) {
            local_feature_pool_insert(pool, old_entries[i].key, old_entries[i].value);
        }
    }

}

float local_feature_pool_load_factor(LocalFeaturePool* pool) {
    return (float)pool->size / pool->capacity;
}

// Delete all features that are last seen before current_frame_num - MAX_LOCAL_FRAMES
void local_feature_pool_remove_old(LocalFeaturePool* pool, int current_frame_num) {
    int i = 0;
    while(i < pool->capacity) {
        if(pool->entries[i].is_occupied) {
            if(remove_old_frame(&pool->entries[i].value, current_frame_num - MAX_LOCAL_FRAMES + 1)) {
                local_feature_pool_delete(pool, pool->entries[i].key);
                i--;
            }
        }
        i++;
    }
}

void local_feature_pool_valid_keys(LocalFeaturePool* pool, int* num_keys, int* keys) {
    for(int i = 0; i < pool->capacity; i++) {
        if(pool->entries[i].is_occupied) {
            keys[(*num_keys)++] = pool->entries[i].key;
        }
    }
}

void local_feature_pool_check_invariant(LocalFeaturePool* pool, int cur_frame, bool print) {
    int size = 0;
    for(int i = 0; i < pool->capacity; i++) {
        HashEntry* entry = &pool->entries[i];
        if(entry->is_occupied) {
            if(print) {
                LocalFeature* feature = &pool->entries[i].value;
                printf("Index %d, Feature %d: ", i, entry->key);
                int ptr = feature->frame_ptr;
                for(int j = 0; j < feature->num_frames; j++) {
                    printf("Frame %d ", feature->frames[ptr]);
                    ptr = (ptr + 1) % MAX_LOCAL_FRAMES;
                }
                printf("\n");
            }

            size++;
            if(entry->key == -1) {
                printf("Entry key is -1\n");
                exit(0);
            }
            if(entry->value.word_id != entry->key) {
                printf("Entry key %d does not match value word id %d\n", entry->key, entry->value.word_id);
                fflush(stdout);
                exit(0);
            }
            LocalFeature* feature = &pool->entries[i].value;

            if(feature->num_frames < 1) {
                printf("Feature has no frames\n");
                exit(0);
            }


            int ptr = feature->frame_ptr;
            if(feature->frames[ptr] < cur_frame - MAX_LOCAL_FRAMES + 1) {
                printf("Frame %d is too old\n", feature->frames[ptr]);
                exit(0);
            }
            for(int j = 1; j < feature->num_frames; j++) {
                int last_ptr = ptr;
                ptr = (ptr + 1) % MAX_LOCAL_FRAMES;
                if(feature->frames[ptr] <= feature->frames[last_ptr]) {
                    printf("Frames are not in increasing order\n");
                    exit(0);
                }
            }
        }
    }
    
    if(size != pool->size) {
        printf("Size count is incorrect: %d %d\n", size, pool->size);
        exit(0);
    }
    if(print) {
        printf("Load factor: %f\n", local_feature_pool_load_factor(pool));
    }
}
