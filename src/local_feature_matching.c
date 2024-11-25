/**
 * local_feature_matching.c
 * Given a keyframe with a set of features and a local feature pool
 * Iteratively check if the keyframe features are in the feature pool
 * Each feature has a counter that stores the frame number it was last seen
 * If a feature has not been matched for a certain number of frames, it is removed
 *
 **/

#include "local_feature_pool.h"

#include <stdio.h>

#define NUM_FEATURE_PER_FRAME 200
#define NUM_OVERLAPPING_FEATURES 75
#define MAX_FEATURE_ID 100000

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int word_id_list_len = MAX_FEATURE_ID;
int word_id_list[MAX_FEATURE_ID];

bool check_duplicate_ids(int len, int* v) {
    bool found[MAX_FEATURE_ID] = {0};
    for(int i = 0; i < len; i++) {
        if(found[v[i]]) {
            printf("Duplicate ID %d\n", v[i]);
            return true;
        }
        found[v[i]] = true;
    }
    return false;
}

void bubble_sort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void generate_word_ids(LocalFeaturePool* pool, int num_id, int* ids) {

    /*************** INITIALIZATION ***************/
    word_id_list_len = MAX_FEATURE_ID;
    for(int i = 0; i < MAX_FEATURE_ID; i++) {
        word_id_list[i] = i;
    }
    /*************** END INITIALIZATION ***************/

    int valid_keys[LOCAL_FEATURE_POOL_CAPACITY];
    int num_valid_keys = 0;
    local_feature_pool_valid_keys(pool, &num_valid_keys, valid_keys);

    bubble_sort(valid_keys, num_valid_keys);

    if(check_duplicate_ids(num_valid_keys, valid_keys)) {
        printf("Duplicate overlap keys found!\n");
        exit(1);
    }

    for(int i = num_valid_keys - 1; i >= 0; i--) {
        int key = valid_keys[i];
        word_id_list[key] = word_id_list[word_id_list_len - 1];
        word_id_list_len--;
    }

    // Check word id list
    bool found[MAX_FEATURE_ID] = {0};
    bool found_overlap[MAX_FEATURE_ID] = {0};
    for(int i = 0; i < word_id_list_len; i++) {
        if(found[word_id_list[i]]) {
            printf("Duplicate new ID %d\n", word_id_list[i]);
            exit(1);
        }
        found[word_id_list[i]] = true;
    }
    for(int i = 0; i < num_valid_keys; i++) {
        if(found[valid_keys[i]]) {
            printf("Duplicate new with overlap ID %d\n", valid_keys[i]);
            exit(1);
        }
        if(found_overlap[valid_keys[i]]) {
            printf("Duplicate overlap with overlap ID %d\n", valid_keys[i]);
            exit(1);
        }
        found[valid_keys[i]] = true;
        found_overlap[valid_keys[i]] = true;
    }


    int num_overlap_features = min(NUM_OVERLAPPING_FEATURES, num_valid_keys);
    
    for(int i = 0; i < num_id; i++) {
        bool generate_overlap = (rand() % num_id) < num_overlap_features && num_valid_keys > 0;
        if(generate_overlap) {
            int index = rand() % num_valid_keys;
            ids[i] = valid_keys[index];
            valid_keys[index] = valid_keys[num_valid_keys - 1];
            num_valid_keys--;
        }
        else {
            if(word_id_list_len == 0) {
                printf("No more new features. Increase MAX_FEATURE_ID\n");
                exit(1);
            }
            int index = rand() % word_id_list_len;
            ids[i] = word_id_list[index];
            word_id_list[index] = word_id_list[word_id_list_len - 1];
            word_id_list_len--;
        }
    }

}

int main() {
    srand(0);

    /********** CHANGE THESE PARAMETERS **********/
    int num_frames = 100;
    /********** END CHANGE **********/

    int frame_features[NUM_FEATURE_PER_FRAME];


    LocalFeaturePool pool;
    init_local_feature_pool(&pool);

    for(int frame = 0; frame < num_frames; frame++) {
        printf("frame = %d\n", frame);
        fflush(stdout);
        // Generate NUM_FEATURE_PER_FRAME features for each frame
        // of which NUM_OVERLAPPING_FEATURES features are shared with the local pool
        generate_word_ids(&pool, NUM_FEATURE_PER_FRAME, frame_features);
        if(check_duplicate_ids(NUM_FEATURE_PER_FRAME, frame_features)) {
            printf("Duplicate IDs found!\n");
            exit(1);
        }

        /***************** MEASURE THIS *****************/
        for(int i = 0; i < NUM_FEATURE_PER_FRAME; i++) {
            LocalFeature feature;
            init_local_feature_with_id(&feature, frame_features[i], frame);
            LocalFeaturePoolInsertResult result = local_feature_pool_insert(&pool, frame_features[i], feature);
            if(!result.inserted) {
                update_local_feature(result.feature, frame);
            }

        }
        local_feature_pool_remove_old(&pool, frame);
        /***************** END MEASURE *****************/


        bool debug_print = false;
        local_feature_pool_check_invariant(&pool, frame, debug_print);
        printf("Load factor: %f\n", local_feature_pool_load_factor(&pool));
    }


}
