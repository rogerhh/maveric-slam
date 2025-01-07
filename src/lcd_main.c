#include <stdio.h>
#include <stdbool.h>

#define NUM_FRAMES 100
#define FEATURES_PER_FRAME 200
#define MAX_FEATURE_ID 5000
#define SOME_SEED 97
#define SOME_PRIME 29
#define SOME_PRIME2 73
#define SOME_PRIME3 997

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int prev_frames_num_features[NUM_FRAMES] = {0};
int prev_frames[NUM_FRAMES][FEATURES_PER_FRAME];
int cur_frame_num_features = 0;
int cur_frame[FEATURES_PER_FRAME];

int main() {
    // First populate the frames with BoW features
    for(int i = 0; i < NUM_FRAMES; i++) {
        int features[MAX_FEATURE_ID] = {0};
        int seed = i * SOME_SEED;
        for(int j = 0; j < FEATURES_PER_FRAME; j++) {
            seed = ((seed * SOME_PRIME + SOME_PRIME2 + i) % SOME_PRIME3) % MAX_FEATURE_ID;
            features[seed] = 1;
        }
        for(int j = 0; j < MAX_FEATURE_ID; j++) {
            if(features[j] == 1) {
                prev_frames[i][prev_frames_num_features[i]] = j;
                prev_frames_num_features[i]++;
            }
        }
    }

    int features[MAX_FEATURE_ID] = {0};
    int seed = MAX_FEATURE_ID * SOME_SEED;
    for(int j = 0; j < FEATURES_PER_FRAME; j++) {
        seed = (seed * SOME_PRIME + SOME_PRIME2) % MAX_FEATURE_ID;
        features[seed] = 1;
    }
    for(int j = 0; j < MAX_FEATURE_ID; j++) {
        if(features[j] == 1) {
            cur_frame[cur_frame_num_features] = j;
            cur_frame_num_features++;
        }
    }

    int num_matched_features[NUM_FRAMES] = {0};

    /************** START MEASURING FROM HERE **************/
    for(int i = 0; i < NUM_FRAMES; i++) {
        int num_matching_features = 0;

        int idx1 = 0, idx2 = 0;

        while(idx1 < prev_frames_num_features[i] && idx2 < cur_frame_num_features) {
            if(prev_frames[i][idx1] == cur_frame[idx2]) {
                num_matching_features++;
                idx1++;
                idx2++;
            }
            else if(prev_frames[i][idx1] < cur_frame[idx2]) {
                idx1++;
            }
            else {
                idx2++;
            }
        }
        num_matched_features[i] = num_matching_features;
    }

    /************** END MEASURING HERE **************/

    for(int i = 0; i < NUM_FRAMES; i++) {
        printf("Frame %d: %d %d\n", i, prev_frames_num_features[i], num_matched_features[i]);
    }

}
