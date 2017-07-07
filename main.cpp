#include <iostream>
#include <opencv/cv.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>



#include "frame.h"
#include "MILD/loop_closure_detector.h"
#include "MILD/BayesianFilter.hpp"
#include "global.h"
using namespace std;
using namespace cv;




bool DirectoryExists( const char* pzPath )
{
    if ( pzPath == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (pzPath);

    if (pDir != NULL)
    {
        bExists = true;
        (void) closedir (pDir);
    }

    return bExists;
}


void  LoadRGBFrame(string fileName, int index, Frame &t, string tag)
{
    t.frame_index = index;
    t.keypoints.clear();
    t.descriptor.release();
    t.depth_value.clear();
    t.rgb.release();
    t.depth.release();
    t.rgb = imread(fileName);
}

void extractFeatures(Frame &t,string tag,int maximum_feature_num)
{
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(maximum_feature_num);
    orb->detectAndCompute(t.rgb, cv::noArray(), t.keypoints, t.descriptor);
    t.rgb.release();			// save memory space.
}

void test_mild(string folder,
               int maximum_feature_num = 800,
               float probability_threshold = 0.5,
               int non_loop_closure_threshold = 4,
               float min_shared_score_threshold = 4,
               int min_distance = 60)
{


    // output files
    string tag = folder.substr(folder.find_last_of("/") + 1, folder.find_last_of(".") - folder.find_last_of("/") - 1);
    string output_folder = "output/" + tag ;
    string output_feature_folder = output_folder + "/feature/";
    string frame_feature_num_file = output_folder + "/lcd_frame_feature_num.bin";
    string shared_score_file = output_folder + "/lcd_shared_score_mild.bin";
    string visit_probability_file = output_folder + "/lcd_shared_probability.bin";
    string visit_flag_file = output_folder + "/lcd_shared_flag.bin";
    string relocalization_time_per_frame_file = output_folder + "/relocalization_time_per_frame.bin";

    // clock setting
    clock_t start, end;
    int dir_err;

    if(!DirectoryExists("output"))
    {    dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            printf("Error creating directory !  %s\n","output");
            exit(1);
        }
    }

    if(!DirectoryExists(output_folder.c_str()))
    {    dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            printf("Error creating directory !  %s\n",output_folder.c_str());
            exit(1);
        }
    }
    if(!DirectoryExists(output_folder.c_str()))
    {
        dir_err = mkdir(output_feature_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }
    }
    int runFrameNum;
    std::vector<std::string> fileName;
    std::vector<Frame> frame_list;
    fstream fin;
    char line[256];
    fin.open(folder, ios::in);
    while (fin.getline(line, sizeof(line), '\n'))
    {
        string input = line;
        memset(line, '\0', sizeof(line));
        fileName.push_back(input);
    }
    runFrameNum = fileName.size();
    for (int i = 0; i < runFrameNum; i++)
    {
        Frame curFrame;
        if (i % 100 == 0)
        {
            cout << "loading " << fileName[i] << endl;
        }
        LoadRGBFrame(fileName[i], i + 1, curFrame, tag);
        frame_list.push_back(curFrame);
    }
    start = clock();
    for (int i = 0; i < frame_list.size(); i++)
    {
        if (i % 100 == 0)
        {
            cout << "extracting features: " << i << endl;
        }
        extractFeatures(frame_list[i],tag,maximum_feature_num);
    }
    end = clock();
    double duration_loadFrame = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "feature extraction per frame : " << duration_loadFrame / frame_list.size() * 1000.0f << "ms" << endl;
    float *p_shared_score = new float[runFrameNum * runFrameNum];
    memset(p_shared_score, 0, sizeof(float)* runFrameNum * runFrameNum);
    float *p_visit_probability = new float[runFrameNum * runFrameNum];
    memset(p_visit_probability, 0, sizeof(float)* runFrameNum * runFrameNum);
    float *p_visit_flag = new float[runFrameNum * runFrameNum];
    memset(p_visit_flag, 0, sizeof(float)* runFrameNum * runFrameNum);
    float *time_cost_per_frame = new float[runFrameNum];
    memset(time_cost_per_frame, 0, sizeof(float)* runFrameNum);
    // relocalization

    cout << "begin loop closure detection " << endl;

    MILD::LoopClosureDetector lcd(FEATURE_TYPE_ORB, 16,0);
    lcd.displayParameters();
    start = clock();

    Eigen::VectorXf previous_visit_probability(1);
    previous_visit_probability << 0.1;
    std::vector<Eigen::VectorXf> privious_visit_flag;
    MILD::BayesianFilter spatial_filter(probability_threshold,non_loop_closure_threshold,min_shared_score_threshold, min_distance);
    for (int k = 0; k < frame_list.size(); k++)
    {
        if (k % 100 == 0)
        {
            cout << "loop closure detection: " << k << endl;
        }
        std::vector<float > similarity_score;
        similarity_score.clear();
        lcd.insert_and_query_database(frame_list[k].descriptor, similarity_score);

        spatial_filter.filter(similarity_score,
            previous_visit_probability,
            privious_visit_flag);
        for (int i = 0; i < similarity_score.size(); i++)
        {
            p_shared_score[i + k * runFrameNum] = similarity_score[i];
        }
        for (int i = 0; i < previous_visit_probability.size(); i++)
        {
            p_visit_probability[i + k * runFrameNum] = previous_visit_probability[i];
        }
        if (privious_visit_flag.size() >= 4)
        {
            for (int i = 0; i < privious_visit_flag[privious_visit_flag.size() - 4].size(); i++)
            {
                p_visit_flag[i + (k - 3) * runFrameNum] = privious_visit_flag[privious_visit_flag.size() - 4][i];
            }
        }
    }

    if (privious_visit_flag.size() >= 4)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < privious_visit_flag[privious_visit_flag.size() - (3-j)].size(); i++)
            {
                p_visit_flag[i + (runFrameNum - (3-j)) * runFrameNum] = privious_visit_flag[privious_visit_flag.size() - (3-j)][i];
            }

        }

    }
    end = clock();
    double duration_lcd = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "loop closure detection time per frame: " << duration_lcd / frame_list.size() * 1000.0f << "ms" << endl;
    float frame_num = lcd.features_descriptor.size();
    float average_feature_per_frame = lcd.count_feature_in_database() / frame_num;
    cout << "hamming distance calculation per feature : " << (float)(lcd.statistics_num_distance_calculation) / frame_num / average_feature_per_frame << endl;
    cout << "feature count : " << average_feature_per_frame << "features,	" << frame_num << "frames" << endl;
    int * frame_feature_num = new int[frame_list.size()];
    for (int i = 0; i < frame_list.size(); i++)
    {
        frame_feature_num[i] = frame_list[i].keypoints.size();
    }
    FILE * fp;
    fp = fopen(frame_feature_num_file.c_str(), "wb+");
    fwrite(frame_feature_num, sizeof(int), frame_list.size(), fp);
    fclose(fp);
    fp = fopen(shared_score_file.c_str(), "wb+");
    fwrite(p_shared_score, sizeof(float), runFrameNum * runFrameNum, fp);
    fclose(fp);
    fp = fopen(visit_probability_file.c_str(), "wb+");
    fwrite(p_visit_probability, sizeof(float), runFrameNum * runFrameNum, fp);
    fclose(fp);
    fp = fopen(visit_flag_file.c_str(), "wb+");
    fwrite(p_visit_flag, sizeof(float), runFrameNum * runFrameNum, fp);
    fclose(fp);

    system("pause");
    return;
}


int main(int argc, char *argv[])
{
    cv::FileStorage fSettings;
    string data_folder;
    if (argc == 3)
    {
        data_folder = argv[1];
        fSettings = cv::FileStorage(argv[2], cv::FileStorage::READ);
    }
    else
    {
        cout << "please check your input!" << endl;
        cout << "standard input: ./mild imageList.txt configure.yaml" << endl;
        cout << "imageList.txt: indicats the path of each input RGB image per line" << endl;
        cout << "configure.yaml: indicats the parameters used in loop closure detection" << endl;
        return 0;
    }

    float probability_threshold = fSettings["probability_threshold"];
    int non_loop_closure_threshold = fSettings["non_loop_closure_threshold"];
    float min_shared_score_threshold = fSettings["min_shared_score_threshold"];
    int min_distance = fSettings["min_distance"];
    int maximum_feature_num = fSettings["maximum_feature_num"];
    test_mild(data_folder,maximum_feature_num,probability_threshold,non_loop_closure_threshold,min_shared_score_threshold,min_distance);
    system("pause");
    return 0;
}
