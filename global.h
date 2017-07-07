#ifndef GLOBAL_H
#define GLOBAL_H



#define TUM_DATA



#define FRAME_MATCH_HAMMING_THRESHOLD 50
#define SPARSE_MATCH_HAMMING_THRESHOLD 50

#define USE_TSDF 0
#define SHOW_FLAG 0
                  // parameters for hash matching
#define similarity_threshold	3
#define salient_threshold		2
#define show_detected_features  0				// show detected features or not
#define use_rectify				0				// rectify or not
#define hamming_covariance		(900.0f)		// used in calculating feature similarity
#define save_features			0				// save features into output or not.
#define use_intrisic			1				// use intel_sse functions
#define MAX_FEATURE_NUM			800
#define use_early_terminate		1
#define BURSTINESS_HANDLING		1
                  // parameters for optimizing reprojection error
#define depth_value_threshold 5.5
#define minimum_3d_correspondence 40
#define huber_threshould_square (0.0001f)
#define outlier_3d_threshold (1e-2)
#define frame_valid_pts_maximum (200)

#ifdef TUM_DATA
#define DEPTH_MAP_SCALE (5000.f)
#else
#define DEPTH_MAP_SCALE (1000.f)
#endif

#endif
